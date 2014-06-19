// nnetbin/rbm-uttbias-forward.cc

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

#include "nnet/nnet-trnopts.h"
#include "nnet/nnet-rbm.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-randomizer.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/timer.h"
#include "cudamatrix/cu-device.h"
#include "cudamatrix/cu-rand.h"


int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::nnet1;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Forward through the RBM with utt dependent bias.\n"
        "Usage:  rbm-uttbias-forward [options] <model-in> <feature-rspecifier> "
        "<hidbias-rspecifier> <act-wspecifier>\n"
        "e.g.: \n"
        " rbm-uttbias-forward 1.rbm scp:train.scp ark:hidbias1.ark ark:out.ark\n";

    ParseOptions po(usage);

    bool apply_log = false;
    po.Register("apply-log", &apply_log, "Apply log to activations");

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform, "Feature transform in Nnet format");

    std::string use_gpu="yes";
    po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA"); 

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        hidbias_rspecifier = po.GetArg(3),
        act_wspecifier = po.GetArg(4);

     
    using namespace kaldi;
    using namespace kaldi::nnet1;
    typedef kaldi::int32 int32;

#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
    CuDevice::Instantiate().DisableCaching();
#endif

    Nnet rbm_transf;
    if(feature_transform != "") {
      rbm_transf.Read(feature_transform);
    }

    // Read nnet, extract the RBM
    Nnet nnet;
    nnet.Read(model_filename);
    KALDI_ASSERT(nnet.NumComponents()==1);
    KALDI_ASSERT(nnet.GetComponent(0).GetType() == Component::kRbm);
    RbmBase &rbm = dynamic_cast<RbmBase&>(nnet.GetComponent(0));

    kaldi::int64 total_frames = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessBaseFloatVectorReader hidbias_reader(hidbias_rspecifier);

    BaseFloatMatrixWriter act_writer(act_wspecifier);
    
    CuMatrix<BaseFloat> feats, feats_transf, acts;
    Matrix<BaseFloat> acts_host;

    // hidden biases
    Vector<BaseFloat> hidbias(rbm.OutputDim());

    Timer time;
    KALDI_LOG << "RBM FORWARD STARTED";

    int32 num_done = 0, num_other_error = 0;

    for ( ; !feature_reader.Done(); feature_reader.Next()) {
      std::string utt = feature_reader.Key();
      KALDI_VLOG(3) << utt;

      // read
      const Matrix<BaseFloat> &mat = feature_reader.Value();
      KALDI_VLOG(2) << "Processing utterance " << num_done+1
                    << ", " << feature_reader.Key()
                    << ", " << mat.NumRows() << "frm";

      //check for NaN/inf
      for (int32 r = 0; r<mat.NumRows(); r++) {
        for (int32 c = 0; c<mat.NumCols(); c++) {
          BaseFloat val = mat(r,c);
          if (val != val) KALDI_ERR << "NaN in features of : " << feature_reader.Key();
          if (val == std::numeric_limits<BaseFloat>::infinity())
            KALDI_ERR << "inf in features of : " << feature_reader.Key();
        }
      }

      // setup the RBM model
      if(hidbias_reader.IsOpen()) {
        if(!hidbias_reader.HasKey(utt)) {
          KALDI_WARN << "Utterance " << utt <<": Skipped because no visbias found in "
              << hidbias_rspecifier << ".";
          num_other_error++;
          continue;
        }
        hidbias.CopyFromVec(hidbias_reader.Value(utt));
      }
      rbm.SetHiddenBias(hidbias);

      // push features to GPU
      feats.Resize(mat.NumRows(),mat.NumCols());
      feats.CopyFromMat(mat);
      // apply optional feature transform
      rbm_transf.Feedforward(feats, &feats_transf);
      // forward pass
      rbm.Propagate(feats_transf, &acts);

      if(apply_log) {
        acts.ApplyLog();
      }

      // download to CPU
      acts_host.Resize(acts.NumRows(), acts.NumCols());
      acts.CopyToMat(&acts_host);

      //check for NaN/inf
      for (int32 r = 0; r < acts_host.NumRows(); r++) {
        for (int32 c = 0; c < acts_host.NumCols(); c++) {
          BaseFloat val = acts_host(r,c);
          if (val != val) KALDI_ERR << "NaN in NNet output of : " << feature_reader.Key();
          if (val == std::numeric_limits<BaseFloat>::infinity())
            KALDI_ERR << "inf in NNet coutput of : " << feature_reader.Key();
        }
      }

      // write
      act_writer.Write(feature_reader.Key(), acts_host);

      // progress log
      num_done++;
      if (num_done % 100 == 0) {
        double time_now = time.Elapsed();
        KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
                      << time_now/60 << " min; processed " << total_frames/time_now
                      << " frames per second.";
      }
      total_frames += mat.NumRows();
    }

    // final message
    KALDI_LOG << "Done " << num_done << " files"
              << " in " << time.Elapsed()/60 << "min,"
              << " (fps " << total_frames/time.Elapsed() << ")";

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
