// nnetbin/rbm-uttbias-train.cc

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
        "Train RBM with utt dependent bias by Contrastive Divergence alg. with 1 step of "
        "Markov Chain Monte-Carlo.\n"
        "The tool can perform several iterations (--num-iters) "
        "or it can subsample the training dataset (--drop-data)\n"
        "Usage:  rbm-uttbias-train [options] <model-in> <feature-rspecifier> \""
        "<visbias-[rspecifier|rxfilename]> <hidbias-[rspecifier|rxfilename]> "
        "<model-out> <visbias-wspecifier> <hidbias-wspecifier>\n"
        "e.g.: \n"
        " rbm-uttbias-train 1.rbm.init scp:train.scp ark:visbias1.ark ark:hidbias1.ark "
        "1.rbm ark:visbias2.ark ark:hidbias2.ark \n";

    ParseOptions po(usage);

    RbmTrainOptions trn_opts, trn_opts_rbm;
    trn_opts.Register(&po);

    bool binary = false; 
    po.Register("binary", &binary, "Write output in binary mode");

    bool with_bug = true; 
    po.Register("with-bug", &with_bug, "Apply bug which led to better results (set-initial-momentum-to-max)");
    
    int32 num_iters = 1; 
    po.Register("num-iters", &num_iters, 
                "Number of iterations (smaller datasets should have more iterations, "
                "iterating within tool becase of linear momentum scheduling)");

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform, "Feature transform in Nnet format");

    kaldi::int32 max_frames = 6000; // Allow segments maximum of 30 seconds by default
    po.Register("max-frames",&max_frames, "Maximum number of frames a segment can have to be processed");
    
    BaseFloat drop_data = 0.0; 
    po.Register("drop-data", &drop_data, "Threshold for random dropping of the data (0 no-drop, 1 drop-all)");

    std::string use_gpu="yes";
    po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA"); 

    po.Read(argc, argv);

    if (po.NumArgs() != 7) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        visbias_rspecifier_or_rxfilename = po.GetArg(3),
        hidbias_rspecifier_or_rxfilename = po.GetArg(4),
        target_model_filename = po.GetArg(5),
        visbias_wspecifier = po.GetArg(6),
        hidbias_wspecifier = po.GetArg(7);

     
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

    // Configure the RBM,
    // make some constants accessible, will use them later:
    const BaseFloat& learn_rate = trn_opts.learn_rate;
    const BaseFloat& momentum = trn_opts.momentum;
    const BaseFloat& momentum_max = trn_opts.momentum_max;
    const int32& momentum_steps = trn_opts.momentum_steps;
    const int32& momentum_step_period = trn_opts.momentum_step_period;
    // trn_opts_rbm is for RBM, copy the opts
    trn_opts_rbm = trn_opts;
    trn_opts_rbm.learn_rate = learn_rate*(1-momentum); // keep `effective' learning rate constant
    // pass options to RBM
    rbm.SetRbmTrainOptions(trn_opts_rbm);

    kaldi::int64 total_frames = 0;

    SequentialBaseFloatMatrixReader feature_reader;
    RandomAccessBaseFloatVectorReader visbias_reader, hidbias_reader;

    BaseFloatVectorWriter visbias_writer(visbias_wspecifier);
    BaseFloatVectorWriter hidbias_writer(hidbias_wspecifier);

    CuRand<BaseFloat> cu_rand; // parallel random number generator
    Mse mse;
    
    CuMatrix<BaseFloat> feats, pos_vis,
                        pos_hid, pos_hid_aux, 
                        neg_vis, neg_hid;
    CuMatrix<BaseFloat> dummy_mse_mat;

    // initial biases
    Vector<BaseFloat> visbias(rbm.InputDim()), hidbias(rbm.OutputDim());
    // process visible bias
    if(ClassifyRspecifier(visbias_rspecifier_or_rxfilename, NULL, NULL)
        != kNoRspecifier) {
      visbias_reader.Open(visbias_rspecifier_or_rxfilename);
    }else{
      bool binary;
      Input ki(visbias_rspecifier_or_rxfilename, &binary);
      visbias.Read(ki.Stream(), binary);
    }
    // process hidden bias
    if(ClassifyRspecifier(hidbias_rspecifier_or_rxfilename, NULL, NULL)
        != kNoRspecifier) {
      hidbias_reader.Open(hidbias_rspecifier_or_rxfilename);
    }else{
      bool binary;
      Input ki(hidbias_rspecifier_or_rxfilename, &binary);
      hidbias.Read(ki.Stream(), binary);
    }
    // newly estimated biases
    Vector<BaseFloat> new_visbias(rbm.InputDim()), new_hidbias(rbm.OutputDim());


    Timer time;
    KALDI_LOG << "RBM TRAINING STARTED";

    int32 iter = 1;
    KALDI_LOG << "Iteration " << iter << "/" << num_iters;

    int32 num_done = 0, num_other_error = 0;
    for (int32 iter=1; iter <= num_iters ; ++iter) {
      KALDI_LOG << "Iteration " << iter << "/" << num_iters;
      feature_reader.Open(feature_rspecifier);
      for ( ; !feature_reader.Done(); feature_reader.Next()) {
        std::string utt = feature_reader.Key();
        KALDI_VLOG(3) << utt;
        // get feature matrix
        const Matrix<BaseFloat> &mat = feature_reader.Value();
        // skip too long segments (avoid runinning out of memory)
        if (mat.NumRows() > max_frames) {
          KALDI_WARN << "Utterance " << utt << ": Skipped because it has " << mat.NumRows() << 
            " frames, which is more than " << max_frames << ".";
          num_other_error++;
          continue;
        }

        // setup the RBM model
        if(visbias_reader.IsOpen()) {
          if(!visbias_reader.HasKey(utt)) {
            KALDI_WARN << "Utterance " << utt <<": Skipped because no visbias found in "
                << visbias_rspecifier_or_rxfilename << ".";
            num_other_error++;
            continue;
          }
          visbias.CopyFromVec(visbias_reader.Value(utt));
        }
        if(hidbias_reader.IsOpen()) {
          if(!hidbias_reader.HasKey(utt)) {
            KALDI_WARN << "Utterance " << utt <<": Skipped because no visbias found in "
                << hidbias_rspecifier_or_rxfilename << ".";
            num_other_error++;
            continue;
          }
          hidbias.CopyFromVec(hidbias_reader.Value(utt));
        }
        rbm.SetVisibleBias(visbias);
        rbm.SetHiddenBias(hidbias);

        // push features to GPU
        feats.Resize(mat.NumRows(),mat.NumCols());
        feats.CopyFromMat(mat);
        // apply optional feature transform
        rbm_transf.Feedforward(feats, &pos_vis);
        // subsample training data to get faster epochs on large datasets
        if(drop_data > 0.0) {
          Matrix<BaseFloat> mat2(pos_vis.NumRows(), pos_vis.NumCols(),
                                 kUndefined);
          pos_vis.CopyToMat(&mat2);
          for(int32 r=mat2.NumRows()-1; r >= 0; r--) {
            if(RandUniform() < drop_data) {
              mat2.RemoveRow(r);
            }
          }
          if(mat2.NumRows() == 0) continue;
          pos_vis.Resize(mat2.NumRows(),mat2.NumCols());
          pos_vis.CopyFromMat(mat2);
        }

        // get the dims 
        int32 num_frames = pos_vis.NumRows(),
              dim_hid = rbm.OutputDim();

        // TRAIN with CD1
        // forward pass
        rbm.Propagate(pos_vis, &pos_hid);

        // alter the hidden values, so we can generate negative example
        if (rbm.HidType() == Rbm::Bernoulli) {
          pos_hid_aux.Resize(num_frames, dim_hid);
          cu_rand.BinarizeProbs(pos_hid, &pos_hid_aux);
        } else {
          // assume HidType Rbm::GAUSSIAN
          pos_hid_aux.Resize(num_frames, dim_hid);
          pos_hid_aux.CopyFromMat(pos_hid);
          cu_rand.AddGaussNoise(&pos_hid_aux);
        }

        // reconstruct pass
        rbm.Reconstruct(pos_hid_aux, &neg_vis);
        // propagate negative examples
        rbm.Propagate(neg_vis, &neg_hid);
        // update step
        rbm.RbmUpdate(pos_vis, pos_hid, neg_vis, neg_hid);
        // evaluate mean square error
        mse.Eval(neg_vis, pos_vis, &dummy_mse_mat);

        // write out the newly estimated biases
        rbm.GetVisibleBias(&new_visbias);
        rbm.GetHiddenBias(&new_hidbias);
        visbias_writer.Write(utt, new_visbias);
        hidbias_writer.Write(utt, new_hidbias);

        total_frames += num_frames;

        // change the momentum progressively per 0.5million samples of the data
        {
          static int32 n_prev = -1;
          BaseFloat step = (momentum_max - momentum) / momentum_steps;
          int32 n = total_frames / momentum_step_period; //change every momentum_step_period data
          BaseFloat momentum_actual;
          if(n > momentum_steps) {
            momentum_actual = momentum_max;
          } else {
            momentum_actual = momentum + n*step;
          }
          if(n - n_prev > 0) {
            n_prev = n;
            BaseFloat learning_rate_actual = learn_rate*(1-momentum_actual);
            KALDI_VLOG(1) << "Setting momentum " << (with_bug ? momentum_max : momentum_actual)
                          << " and learning rate " << learning_rate_actual
                          << " after processing " 
                          << static_cast<double>(total_frames)/360000 << "h";
            // pass values to rbm
            trn_opts_rbm.momentum = (with_bug ? momentum_max : momentum_actual);
            trn_opts_rbm.learn_rate = learning_rate_actual;
            rbm.SetRbmTrainOptions(trn_opts_rbm);
          }
        }

        // report the speed
        if (num_done % 5000 == 0) {
          double time_now = time.Elapsed();
          KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
                        << time_now/60 << " min; processed " << total_frames/time_now
                        << " frames per second.";
        }
      }

      feature_reader.Close();
    }

    nnet.Write(target_model_filename, binary);
    
    KALDI_LOG << "Done " << iter << " iterations, " << num_done << " files, "
              << "skipped " << num_other_error << " files. "
              << "[" << time.Elapsed()/60 << " min, fps" << total_frames/time.Elapsed() 
              << "]";

    KALDI_LOG << mse.Report();

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
