/*
 * copy-feat-to-matlab.cc
 *
 *  Created on: Nov 2, 2012
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 *
 * Write the archieve feature file to two files:
 * 1) Index file: filename start_frame(0 indexed) end_frame(not included)
 * 2) Data file: each line is a frame
 *
 */

#include <fstream>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/kaldi-matrix.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Copy features for Matlab use.\n"
            "Usage: copy-feats-to-matlab [options] in-rspecifier out-index-file out-data-file\n";

    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string rspecifier = po.GetArg(1);
    std::string out_index_file = po.GetArg(2);
    std::string out_data_file = po.GetArg(3);

    std::ofstream fidx(out_index_file.c_str());
    std::ofstream fdat(out_data_file.c_str());
    int32 total_frames = 0;

    SequentialBaseFloatMatrixReader kaldi_reader(rspecifier);
    for (; !kaldi_reader.Done(); kaldi_reader.Next()) {

      std::string key = kaldi_reader.Key();
      const Matrix<BaseFloat> &feat = kaldi_reader.Value();

      fidx << key << " " << total_frames << " " << total_frames + feat.NumRows()
          << std::endl;
      for(int32 r=0; r<feat.NumRows(); ++r){
        for (int32 c=0; c<feat.NumCols(); ++c){
          fdat << feat(r,c) << " ";
        }
        fdat << std::endl;
      }
      total_frames += feat.NumRows();
    }

    fidx.close();
    fdat.close();

    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

