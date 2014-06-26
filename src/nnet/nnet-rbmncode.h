// nnet/nnet-rbmncode.h

// Copyright 2014 Bo Li

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.


#ifndef KALDI_NNET_NNET_RBMNCODE_H_
#define KALDI_NNET_NNET_RBMNCODE_H_


#include "nnet/nnet-component.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-various.h"
#include "cudamatrix/cu-math.h"
#include "nnet/nnet-rbm.h"

namespace kaldi {
namespace nnet1 {

/*
 * The code vector and transform cannot be initialized to zero.
 * Only the code transform matrix is stored in the model.
 * The code is stored separately.
 *
 */
class RbmNCode : public RbmBase {
 public:
  RbmNCode(int32 dim_in, int32 dim_out)
   : RbmBase(dim_in, dim_out), code_dim_(0), update_weight(true)
  { } 
  RbmNCode(const Rbm &rbm, int32 code_dim)
   : RbmBase(rbm.InputDim(), rbm.OutputDim()), update_weight(true)
  {
    Matrix<BaseFloat> mat(output_dim_, input_dim_);
    rbm.GetWeight(&mat);
    vis_hid_ = mat;
    Vector<BaseFloat> vis(input_dim_), hid(output_dim_);
    rbm.GetVisibleBias(&vis);
    vis_bias_ = vis;
    rbm.GetHiddenBias(&hid);
    hid_bias_ = hid;
    InitCode(code_dim);
  }
  ~RbmNCode()
  { }  
  
  Component* Copy() const { return new RbmNCode(*this); }
  ComponentType GetType() const { return kRbmNCode; }

  void InitData(std::istream &is) {
    // define options
    int32 dim = 0;
    std::string vis_type;
    std::string hid_type;
    float vis_bias_mean = 0.0, vis_bias_range = 0.0, 
          hid_bias_mean = 0.0, hid_bias_range = 0.0, 
          param_stddev = 0.1;
    std::string vis_bias_cmvn_file; // initialize biases to logit(p_active)
    // parse config
    std::string token; 
    while (!is.eof()) {
      ReadToken(is, false, &token); 
      /**/  if (token == "<CodeDim>") ReadBasicType(is, false, &dim);
      else if (token == "<VisibleType>") ReadToken(is, false, &vis_type);
      else if (token == "<HiddenType>") ReadToken(is, false, &hid_type);
      else if (token == "<VisibleBiasMean>") ReadBasicType(is, false, &vis_bias_mean);
      else if (token == "<VisibleBiasRange>") ReadBasicType(is, false, &vis_bias_range);
      else if (token == "<HiddenBiasMean>") ReadBasicType(is, false, &hid_bias_mean);
      else if (token == "<HiddenBiasRange>") ReadBasicType(is, false, &hid_bias_range);
      else if (token == "<ParamStddev>") ReadBasicType(is, false, &param_stddev);
      else if (token == "<VisibleBiasCmvnFilename>") ReadToken(is, false, &vis_bias_cmvn_file);
      else KALDI_ERR << "Unknown token " << token << " Typo in config?";
      is >> std::ws; // eat-up whitespace
    }

    //
    // initialize
    //
    if (vis_type == "bern" || vis_type == "Bernoulli") vis_type_ = RbmBase::Bernoulli;
    else if (vis_type == "gauss" || vis_type == "Gaussian") vis_type_ = RbmBase::Gaussian;
    else KALDI_ERR << "Wrong <VisibleType>" << vis_type;
    //
    if (hid_type == "bern" || hid_type == "Bernoulli") hid_type_ = RbmBase::Bernoulli;
    else if (hid_type == "gauss" || hid_type == "Gaussian") hid_type_ = RbmBase::Gaussian;
    else KALDI_ERR << "Wrong <HiddenType>" << hid_type;
    // visible-hidden connections
    Matrix<BaseFloat> mat(output_dim_, input_dim_);
    for (int32 r=0; r<output_dim_; r++) {
      for (int32 c=0; c<input_dim_; c++) {
        mat(r,c) = param_stddev * RandGauss(); // 0-mean Gauss with given std_dev
      }
    }
    vis_hid_ = mat;
    // hidden-bias
    Vector<BaseFloat> vec(output_dim_);
    for (int32 i=0; i<output_dim_; i++) {
      // +/- 1/2*bias_range from bias_mean:
      vec(i) = hid_bias_mean + (RandUniform() - 0.5) * hid_bias_range; 
    }
    hid_bias_ = vec;
    // visible-bias
    if (vis_bias_cmvn_file == "") {
      Vector<BaseFloat> vec2(input_dim_);
      for (int32 i=0; i<input_dim_; i++) {
        // +/- 1/2*bias_range from bias_mean:
        vec2(i) = vis_bias_mean + (RandUniform() - 0.5) * vis_bias_range; 
      }
      vis_bias_ = vec2;
    } else {
      KALDI_LOG << "Initializing from <VisibleBiasCmvnFilename> " << vis_bias_cmvn_file;
      Nnet cmvn;
      cmvn.Read(vis_bias_cmvn_file);
      // getting probablity that neuron fires:
      Vector<BaseFloat> p(dynamic_cast<AddShift&>(cmvn.GetComponent(0)).GetShiftVec());
      p.Scale(-1.0);
      // compute logit:
      Vector<BaseFloat> logit_p(p.Dim());
      for(int32 d = 0; d < p.Dim(); d++) {
        if(p(d) < 0.0001) p(d) = 0.0001;
        if(p(d) > 0.9999) p(d) = 0.9999;
        logit_p(d) = log(p(d)) - log(1.0 - p(d));
      }
      vis_bias_ = logit_p;
      KALDI_ASSERT(vis_bias_.Dim() == InputDim());
    }
    //
    if (dim > 0) {
      InitCode(dim);
    }
  }

  void InitCode(int32 dim) {
    KALDI_ASSERT(dim > 0);
    code_dim_ = dim;

    // code to hidden connections
    Matrix<BaseFloat> mat(output_dim_, code_dim_);
    for(int32 r=0; r<output_dim_; ++r) {
      for(int32 c=0; c<output_dim_; ++c) {
        mat(r, c) = 0.01 * RandGauss(); // hard-coded
      }
    }
    code_hid_ = mat;

    // code vector
    Vector<BaseFloat> vec(code_dim_);
    for(int32 i=0; i<code_dim_; ++i) {
      vec(i) = 0.01 * (RandUniform() - 0.5); // hard-coded
    }
    code_vec_ = vec;

  }

  void ReadData(std::istream &is, bool binary) {
    std::string vis_node_type, hid_node_type;
    ReadToken(is, binary, &vis_node_type);
    ReadToken(is, binary, &hid_node_type);
    
    if(vis_node_type == "bern") {
      vis_type_ = RbmBase::Bernoulli;
    } else if(vis_node_type == "gauss") {
      vis_type_ = RbmBase::Gaussian;
    }
    if(hid_node_type == "bern") {
      hid_type_ = RbmBase::Bernoulli;
    } else if(hid_node_type == "gauss") {
      hid_type_ = RbmBase::Gaussian;
    }

    int32 dim;
    ReadBasicType(is, binary, &dim);
    KALDI_ASSERT(dim > 0);
    code_dim_ = dim;

    vis_hid_.Read(is, binary);
    vis_bias_.Read(is, binary);
    hid_bias_.Read(is, binary);

    code_hid_.Read(is, binary);

    KALDI_ASSERT(vis_hid_.NumRows() == output_dim_);
    KALDI_ASSERT(vis_hid_.NumCols() == input_dim_);
    KALDI_ASSERT(vis_bias_.Dim() == input_dim_);
    KALDI_ASSERT(hid_bias_.Dim() == output_dim_);
    KALDI_ASSERT(code_hid_.NumRows() == output_dim_);
    KALDI_ASSERT(code_hid_.NumCols() == code_dim_);
  }
  
  void WriteData(std::ostream &os, bool binary) const {
    switch (vis_type_) {
      case Bernoulli : WriteToken(os,binary,"bern"); break;
      case Gaussian  : WriteToken(os,binary,"gauss"); break;
      default : KALDI_ERR << "Unknown type " << vis_type_;
    }
    switch (hid_type_) {
      case Bernoulli : WriteToken(os,binary,"bern"); break;
      case Gaussian  : WriteToken(os,binary,"gauss"); break;
      default : KALDI_ERR << "Unknown type " << hid_type_;
    }
    WriteBasicType(os, binary, code_dim_);

    vis_hid_.Write(os, binary);
    vis_bias_.Write(os, binary);
    hid_bias_.Write(os, binary);

    code_hid_.Write(os, binary);
  }


  // Component API
  void PropagateFnc(const CuMatrix<BaseFloat> &in, CuMatrix<BaseFloat> *out) {
    // precopy bias
    out->AddVecToRows(1.0, hid_bias_, 0.0);
    // multiply by weights^t
    out->AddMatMat(1.0, in, kNoTrans, vis_hid_, kTrans, 1.0);
    // optionally apply sigmoid
    if (hid_type_ == RbmBase::Bernoulli) {
      out->Sigmoid(*out);
    }
  }

  // RBM training API
  void Reconstruct(const CuMatrix<BaseFloat> &hid_state, CuMatrix<BaseFloat> *vis_probs) {
    // check the dim
    if (output_dim_ != hid_state.NumCols()) {
      KALDI_ERR << "Nonmatching dims, component:" << output_dim_ << " data:" << hid_state.NumCols();
    }
    // optionally allocate buffer
    if (input_dim_ != vis_probs->NumCols() || hid_state.NumRows() != vis_probs->NumRows()) {
      vis_probs->Resize(hid_state.NumRows(), input_dim_);
    }

    ////////////////////////////////////////////////////////////
    // compute the hidden offset caused by the code vector
    if(code_hid_offset_vec_.Dim() != output_dim_) {
      code_hid_offset_vec_.Resize(output_dim_);
    }
    code_hid_offset_vec_.AddMatVec(1.0, code_hid_, kNoTrans, code_vec_, 0.0);

    if(code_hid_offset_mat_.NumRows()!=hid_state.NumRows()) {
      code_hid_offset_mat_.Resize(hid_state.NumRows(), code_hid_offset_vec_.Dim());
    }
    code_hid_offset_mat_.AddVecToRows(1.0, code_hid_offset_vec_, 0.0);

    // precopy bias
    vis_probs->AddVecToRows(1.0, vis_bias_, 0.0);
    // multiply by weights
    vis_probs->AddMatMat(1.0, hid_state, kNoTrans, vis_hid_, kNoTrans, 1.0);
    // add the visible offset caused by the code vector
    vis_probs->AddMatMat(1.0, code_hid_offset_mat_, kNoTrans, vis_hid_, kNoTrans, 1.0);
    // optionally apply sigmoid
    if (vis_type_ == RbmBase::Bernoulli) {
      vis_probs->Sigmoid(*vis_probs);
    }
  }
  
  void RbmUpdate(const CuMatrix<BaseFloat> &pos_vis, const CuMatrix<BaseFloat> &pos_hid, const CuMatrix<BaseFloat> &neg_vis, const CuMatrix<BaseFloat> &neg_hid) {

    KALDI_ASSERT(pos_vis.NumRows() == pos_hid.NumRows() &&
           pos_vis.NumRows() == neg_vis.NumRows() &&
           pos_vis.NumRows() == neg_hid.NumRows() &&
           pos_vis.NumCols() == neg_vis.NumCols() &&
           pos_hid.NumCols() == neg_hid.NumCols() &&
           pos_vis.NumCols() == input_dim_ &&
           pos_hid.NumCols() == output_dim_);

    //lazy initialization of buffers
    if ( vis_hid_corr_.NumRows() != vis_hid_.NumRows() ||
         vis_hid_corr_.NumCols() != vis_hid_.NumCols() ||
         vis_bias_corr_.Dim()    != vis_bias_.Dim()    ||
         hid_bias_corr_.Dim()    != hid_bias_.Dim()    ||
         code_hid_corr_.NumRows()!= code_hid_.NumRows()||
         code_hid_corr_.NumCols()!= code_hid_.NumCols()||
         code_vec_corr_.Dim()        != code_vec_.Dim()){
      vis_hid_corr_.Resize(vis_hid_.NumRows(),vis_hid_.NumCols(),kSetZero);
      vis_bias_corr_.Resize(vis_bias_.Dim());
      hid_bias_corr_.Resize(hid_bias_.Dim());
      code_hid_corr_.Resize(code_hid_.NumRows(), code_hid_.NumCols(), kSetZero);
      code_vec_corr_.Resize(code_vec_.Dim());
    }

    if(code_mat_.NumRows() != pos_vis.NumRows()) {
      code_mat_.Resize(pos_vis.NumRows(), code_vec_.Dim());
    }
    code_mat_.AddVecToRows(1.0, code_vec_, 0.0);

    if(code_hid_offset_mat_.NumRows() != pos_vis.NumRows()) {
      code_hid_offset_mat_.Resize(pos_vis.NumRows(), code_hid_offset_vec_.Dim());
    }
    code_hid_offset_mat_.AddVecToRows(1.0, code_hid_offset_vec_, 0.0);

    //  We use these training hyper-parameters
    //
    const BaseFloat lr = rbm_opts_.learn_rate;
    const BaseFloat mmt = rbm_opts_.momentum;
    const BaseFloat l2 = rbm_opts_.l2_penalty;
    
    BaseFloat N = static_cast<BaseFloat>(pos_vis.NumRows());
    if(update_weight) {
      //  UPDATE vishid matrix
      //
      //  vishidinc = momentum*vishidinc + ...
      //              epsilonw*( (posprods-negprods)/numcases - weightcost*vishid);
      //
      //  vishidinc[t] = -(epsilonw/numcases)*negprods + momentum*vishidinc[t-1]
      //                 +(epsilonw/numcases)*posprods
      //                 -(epsilonw*weightcost)*vishid[t-1]
      //
      vis_hid_corr_.AddMatMat(-lr/N, neg_hid, kTrans, neg_vis, kNoTrans, mmt);
      vis_hid_corr_.AddMatMat(+lr/N, pos_hid, kTrans, pos_vis, kNoTrans, 1.0);
      // code diff
      vis_hid_corr_.AddMatMat(-lr/N, code_hid_offset_mat_, kTrans, neg_vis, kNoTrans, 1.0);
      vis_hid_corr_.AddMatMat(+lr/N, code_hid_offset_mat_, kTrans, pos_vis, kNoTrans, 1.0);
      vis_hid_corr_.AddMat(-lr*l2, vis_hid_);
      vis_hid_.AddMat(1.0, vis_hid_corr_);

      //  UPDATE hidbias vector
      //
      // hidbiasinc = momentum*hidbiasinc + (epsilonhb/numcases)*(poshidact-neghidact);
      //
      // as Tz is constant in the pos and neg steps, it is equal to using neg_hid and pos_hid.
      hid_bias_corr_.AddRowSumMat(-lr/N, neg_hid, mmt);
      hid_bias_corr_.AddRowSumMat(+lr/N, pos_hid, 1.0);
      hid_bias_.AddVec(1.0, hid_bias_corr_, 1.0);

      //  UPDATE visbias vector
      //
      //  visbiasinc = momentum*visbiasinc + (epsilonvb/numcases)*(posvisact-negvisact);
      //
      vis_bias_corr_.AddRowSumMat(-lr/N, neg_vis, mmt);
      vis_bias_corr_.AddRowSumMat(+lr/N, pos_vis, 1.0);
      vis_bias_.AddVec(1.0, vis_bias_corr_, 1.0);
    }
    
    // UPDATE code transform
    //
    // codehidinc = momentum*codehidinc + (epsilon/numcases)*(poshid*z-neghid*z)
    //
    code_hid_corr_.AddMatMat(-lr/N, neg_hid, kTrans, code_mat_, kNoTrans, mmt);
    code_hid_corr_.AddMatMat(+lr/N, pos_hid, kTrans, code_mat_, kNoTrans, 1.0);

    // UPDATE code vector
    //
    // codevecinc= momentum*codevecinc + (epsilon/numcases)*(poshid*T-neghid*T)
    code_mat_.AddMatMat(-1.0, neg_hid, kNoTrans, code_hid_, kNoTrans, 0.0);
    code_mat_.AddMatMat(+1.0, pos_hid, kNoTrans, code_hid_, kNoTrans, 1.0);
    code_vec_corr_.AddRowSumMat(lr/N, code_mat_, mmt);

    // the gradients of code transform and vector depends on each other,
    // hence the updates are done after all the gradients are computed
    code_hid_.AddMat(1.0, code_hid_corr_);
    code_vec_.AddVec(1.0, code_vec_corr_, 1.0);

  }

  RbmNodeType VisType() const { 
    return vis_type_; 
  }

  RbmNodeType HidType() const { 
    return hid_type_; 
  }

  void WriteAsNnet(std::ostream& os, bool binary) const {
    //header
    WriteToken(os,binary,Component::TypeToMarker(Component::kAffineTransform));
    WriteBasicType(os,binary,OutputDim());
    WriteBasicType(os,binary,InputDim());
    if(!binary) os << "\n";
    //data
    vis_hid_.Write(os,binary);
    hid_bias_.Write(os,binary);
    //optionally sigmoid activation
    if(HidType() == Bernoulli) {
      WriteToken(os,binary,Component::TypeToMarker(Component::kSigmoid));
      WriteBasicType(os,binary,OutputDim());
      WriteBasicType(os,binary,OutputDim());
    }
    if(!binary) os << "\n";
  }

  void GetWeight(Matrix<BaseFloat> *weight) {
    vis_hid_.CopyToMat(weight);
  }

  void SetWeight(const Matrix<BaseFloat> &weight) {
    vis_hid_.CopyFromMat(weight);
  }

  void GetVisibleBias(Vector<BaseFloat> *bias) {
    vis_bias_.CopyToVec(bias);
  }

  void SetVisibleBias(const Vector<BaseFloat> &bias) {
    vis_bias_.CopyFromVec(bias);
  }

  void GetHiddenBias(Vector<BaseFloat> *bias) {
    hid_bias_.CopyToVec(bias);
  }

  void SetHiddenBias(const Vector<BaseFloat> &bias) {
    hid_bias_.CopyFromVec(bias);
  }

  void GetCodeXform(Matrix<BaseFloat> *xform) {
    code_hid_.CopyToMat(xform);
  }

  void SetCodeXform(const Matrix<BaseFloat> &xform) {
    code_hid_.CopyFromMat(xform);
  }

  void GetCode(Vector<BaseFloat> *code) {
    code_vec_.CopyToVec(code);
  }

  void SetCode(const Vector<BaseFloat> &code) {
    code_vec_.CopyFromVec(code);
  }

  void SetWeightUpdateFlag(bool flag) {
    update_weight=flag;
  }

protected:
  CuMatrix<BaseFloat> vis_hid_;        ///< Matrix with neuron weights
  CuMatrix<BaseFloat> code_hid_;       ///< Matrix connecting code to hidden
  CuVector<BaseFloat> vis_bias_;       ///< Vector with biases
  CuVector<BaseFloat> hid_bias_;       ///< Vector with biases


  CuVector<BaseFloat> code_vec_;       ///< Noise code vector
  CuMatrix<BaseFloat> code_mat_;       ///< Matrix with num frames of code vector


  CuVector<BaseFloat> code_hid_offset_vec_; ///< Vector = code_hid_ * code_;
  CuMatrix<BaseFloat> code_hid_offset_mat_; ///< Matrix with num frames of hid offset

  CuMatrix<BaseFloat> vis_hid_corr_;   ///< Matrix for linearity updates
  CuMatrix<BaseFloat> code_hid_corr_;  ///< Matrix for code xform updates
  CuVector<BaseFloat> vis_bias_corr_;  ///< Vector for bias updates
  CuVector<BaseFloat> hid_bias_corr_;  ///< Vector for bias updates
  CuVector<BaseFloat> code_vec_corr_;      ///< Vector for code updates

  RbmNodeType vis_type_;
  RbmNodeType hid_type_;

  int32 code_dim_;

  bool update_weight;                 ///< if false, only update code vector

};



} // namespace nnet1
} // namespace kaldi

#endif
