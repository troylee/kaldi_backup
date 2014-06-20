#!/bin/bash
# Author: Bo Li (li-bo@outlook.com)
#
# This is a shell script, but it's recommended that you run the commands one by
# one by copying and pasting into the shell.
#

set -e           #Exit on non-zero return code from any command
set -o pipefail  #Exit if any of the commands in the pipeline will 
                 #return non-zero return code
set -u           #Fail on an undefined variable

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. ./path.sh ## update system path

log_start(){
  echo "#####################################################################"
  echo "Spawning *** $1 *** on" `date` `hostname`
  echo ---------------------------------------------------------------------
}

log_end(){
  echo ---------------------------------------------------------------------
  echo "Done *** $1 *** on" `date` `hostname` 
  echo "#####################################################################"
}

###########################################
# Model training - multi
#

# Note: the --boost-silence option should probably be omitted by default
# for normal setups.  It doesn't always help. [it's to discourage non-silence
# models from modeling silence.]

train_multi_tri1a(){
  log_start "mono [train]"
  steps/aurora4/train_mono.sh --boost-silence 1.25 --nj 4 --norm_vars true \
    feat/mfcc/train_multi data/lang exp_multi/mono || exit 1;
  log_end "mono [train]"

  log_start "mono [align]"
  steps/aurora4/align_si.sh --boost-silence 1.25 --nj 4  \
     feat/mfcc/train_multi data/lang exp_multi/mono exp_multi/mono_ali || exit 1;
  log_end "mono [align]"

  log_start "tri1a [train]"
  steps/aurora4/train_deltas.sh --boost-silence 1.25 --norm_vars true \
      4200 55000 feat/mfcc/train_multi data/lang exp_multi/mono_ali exp_multi/tri1a || exit 1;
  utils/mkgraph.sh data/lang_bcb05cnp exp_multi/tri1a exp_multi/tri1a/graph_bg || exit 1;
  log_end "tri1a [train]"
}
#train_multi_tri1a

decode_multi_tri1a(){
  log_start "tri1a [decode]"
  for i in `seq -f "%02g" 1 14`; do
    x=test${i}
    steps/aurora4/decode_deltas.sh --nj 4 --srcdir exp_multi/tri1a exp_multi/tri1a/graph_bg feat/mfcc/${x} exp_multi/tri1a/decode/decode_bg_${x} || exit 1;
  done
  # write out the average WER results
  local/average_wer.sh 'exp_multi/tri1a/decode/decode_bg_test*' | tee exp_multi/tri1a/decode/decode_bg_test.avgwer
  log_end "tri1a [decode]"
}
#decode_multi_tri1a

align_multi_tri1a(){
  # align multi-style data with multi-trained model, needs a larger beam
  log_start "tri1a [align-train-multi]"
  steps/aurora4/align_si.sh --nj 4 --retry-beam 60 feat/mfcc/train_multi data/lang exp_multi/tri1a exp_multi/tri1a_ali/train_multi || exit 1;
  log_end "tri1a [align-train-multi]"

  log_start "tri1a [align-dev-multi]"
  steps/aurora4/align_si.sh --nj 4 --retry-beam 80 feat/mfcc/dev_multi data/lang exp_multi/tri1a exp_multi/tri1a_ali/dev_multi || exit 1;
  log_end "tri1a [align-dev-multi]"

  # align clean data with multi-trained model
  log_start "tri1a [align-train-clean]"
  steps/aurora4/align_si.sh --nj 4 feat/mfcc/train_clean data/lang exp_multi/tri1a exp_multi/tri1a_ali/train_clean || exit 1;
  log_end "tri1a [align-train-clean]"

  log_start "tri1a [align-dev-clean]"
  steps/aurora4/align_si.sh --nj 4 feat/mfcc/dev_clean data/lang exp_multi/tri1a exp_multi/tri1a_ali/dev_clean || exit 1;
  log_end "tri1a [align-dev-clean]"

  # additional processing of the clean data alignments for used as multi labels
  dir=exp_multi/tri1a_ali/train_clean
  mkdir -p ${dir}/ori
  for j in {1..4}; do 
    mv ${dir}/ali.${j}.gz ${dir}/ori/ali.${j}.gz
    utils/convert_ali_names.py feat/mfcc/train_multi/feats.scp ${dir}/ori/ali.${j}.gz ${dir}/ali.${j}.gz
  done

  dir=exp_multi/tri1a_ali/dev_clean
  mkdir -p ${dir}/ori
  for j in {1..4}; do 
    mv ${dir}/ali.${j}.gz ${dir}/ori/ali.${j}.gz
    utils/convert_ali_names.py feat/mfcc/dev_multi/feats.scp ${dir}/ori/ali.${j}.gz ${dir}/ali.${j}.gz
  done

  # sanity check for the genreated clean frame alignment
  ./utils/alignment_frame_checking.sh exp_multi/tri1a_ali/train_clean/ exp_multi/tri1a_ali/train_multi/
  ./utils/alignment_frame_checking.sh exp_multi/tri1a_ali/dev_clean/ exp_multi/tri1a_ali/dev_multi/
}
#align_multi_tri1a

train_multi_spr_tri1b(){
  log_start "tri1b [train]"
  steps/aurora4/singlepass_retrain.sh feat/mfcc/train_multi exp_multi/tri1a_ali/train_multi exp_multi/tri1b || exit 1;
  utils/mkgraph.sh data/lang_bcb05cnp exp_multi/tri1b exp_multi/tri1b/graph_bg || exit 1;
  log_end "tri1b [train]"
}
#train_multi_spr_tri1b

###############################################
#Now begin train DNN systems on multi data

pretrain(){
  #RBM pretrain
  log_start "tri2a [pretrain]"
  dir=exp_multi/tri2a_dnn_pretrain
  mkdir -p $dir/log
  steps/aurora4/pretrain_dbn.sh --nn-depth 7 --rbm-iter 20 --norm-vars true --splice 5 feat/fbank/train_multi $dir
  log_end "tri2a [pretrain]"
}
#pretrain

###############################################
# DNN fine-tuning with multi-style aligned labels
#

train_tri3a(){
  log_start "tri3a [train]"
  dir=exp_multi/tri3a_dnn
  ali=exp_multi/tri1a_ali/train_multi
  ali_dev=exp_multi/tri1a_ali/dev_multi
  dbn=exp_multi/tri2a_dnn_pretrain/7.dbn
  mkdir -p $dir/log
  steps/aurora4/nnet_train.sh --norm-vars true --dbn $dbn --hid-layers 0 --learn-rate 0.008 --use-gpu-id 0 \
    feat/fbank/train_multi feat/fbank/dev_multi data/lang $ali $ali_dev $dir || exit 1;
  utils/mkgraph.sh data/lang_bcb05cnp exp_multi/tri3a_dnn exp_multi/tri3a_dnn/graph_bg || exit 1;
  log_end "tri3a [train]" 
}
train_tri3a

decode_tri3a(){
  log_start "tri3a [decode]"
  for i in `seq -f "%02g" 1 14`; do
    x=test${i}
    steps/aurora4/nnet_decode.sh --nj 4 --acwt 0.10 --config conf/decode_dnn.config --srcdir exp_multi/tri3a_dnn exp_multi/tri3a_dnn/graph_bg feat/fbank/${x} exp_multi/tri3a_dnn/decode/decode_bg_${x} || exit 1;
  done
  local/average_wer.sh 'exp_multi/tri3a_dnn/decode/decode_bg_test*' | tee exp_multi/tri3a_dnn/decode/decode_bg_test.avgwer
  log_end "tri3a [decode]"
}
#decode_tri3a

align_tri3a(){
  #nnet realignments
  log_start "tri3a [realign-train_multi]"
  steps/aurora4/nnet_align.sh --nj 4 --retry-beam 60 feat/fbank/train_multi data/lang exp_multi/tri3a_dnn exp_multi/tri3a_dnn_ali/train_multi || exit 1;
  log_end "tri3a [realign-train_multi]"

  log_start "tri3a [realign-dev_multi]"
  steps/aurora4/nnet_align.sh --nj 4 --retry-beam 80 feat/fbank/dev_multi data/lang exp_multi/tri3a_dnn exp_multi/tri3a_dnn_ali/dev_multi || exit 1;
  log_end "tri3a [realign-dev_multi]"
}
align_tri3a

train_tri3b(){
  log_start "tri3b [train]"
  dir=exp_multi/tri3b_dnn
  ali=exp_multi/tri3a_dnn_ali/train_multi
  ali_dev=exp_multi/tri3a_dnn_ali/dev_multi
  mlp_init=exp_multi/tri3a_dnn/nnet_7.dbn_dnn.init
  mkdir -p $dir/log
  steps/aurora4/nnet_train.sh --norm-vars true --mlp-init $mlp_init --hid-layers 0 --learn-rate 0.008 --use-gpu-id 0 \
    feat/fbank/train_multi feat/fbank/dev_multi data/lang $ali $ali_dev $dir || exit 1;
  utils/mkgraph.sh data/lang_bcb05cnp exp_multi/tri3b_dnn exp_multi/tri3b_dnn/graph_bg || exit 1;
  log_end "tri3b [train]"
}
train_tri3b

align_tri3b(){
  #nnet realignments
  log_start "tri3b [realign-train_multi]"
  steps/aurora4/nnet_align.sh --nj 4 --retry-beam 60 feat/fbank/train_multi data/lang exp_multi/tri3b_dnn exp_multi/tri3b_dnn_ali/train_multi || exit 1;
  log_end "tri3b [realign-train_multi]"

  log_start "tri3b [realign-dev_multi]"
  steps/aurora4/nnet_align.sh --nj 4 --retry-beam 80 feat/fbank/dev_multi data/lang exp_multi/tri3b_dnn exp_multi/tri3b_dnn_ali/dev_multi || exit 1;
  log_end "tri3b [realign-dev_multi]"
}
align_tri3b

train_tri3c(){
  log_start "tri3c [train]"
  dir=exp_multi/tri3c_dnn
  ali=exp_multi/tri3b_dnn_ali/train_multi
  ali_dev=exp_multi/tri3b_dnn_ali/dev_multi
  mlp_init=exp_multi/tri3a_dnn/nnet_7.dbn_dnn.init
  mkdir -p $dir/log
  steps/aurora4/nnet_train.sh --norm-vars true --mlp-init $mlp_init --hid-layers 0 --learn-rate 0.008 --use-gpu-id 0 \
    feat/fbank/train_multi feat/fbank/dev_multi data/lang $ali $ali_dev $dir || exit 1;
  utils/mkgraph.sh data/lang_bcb05cnp exp_multi/tri3c_dnn exp_multi/tri3c_dnn/graph_bg || exit 1;
  log_end "tri3c [train]"
}
train_tri3c

align_tri3c(){
  #nnet realignments
  log_start "tri3c [realign-train_multi]"
  steps/aurora4/nnet_align.sh --nj 4 --retry-beam 60 feat/fbank/train_multi data/lang exp_multi/tri3c_dnn exp_multi/tri3c_dnn_ali/train_multi || exit 1;
  log_end "tri3c [realign-train_multi]"

  log_start "tri3c [realign-dev_multi]"
  steps/aurora4/nnet_align.sh --nj 4 --retry-beam 80 feat/fbank/dev_multi data/lang exp_multi/tri3c_dnn exp_multi/tri3c_dnn_ali/dev_multi || exit 1;
  log_end "tri3c [realign-dev_multi]"
}
align_tri3c

train_tri3d(){
  log_start "tri3d [train]"
  dir=exp_multi/tri3d_dnn
  ali=exp_multi/tri3c_dnn_ali/train_multi
  ali_dev=exp_multi/tri3c_dnn_ali/dev_multi
  mlp_init=exp_multi/tri3a_dnn/nnet_7.dbn_dnn.init
  mkdir -p $dir/log
  steps/aurora4/nnet_train.sh --norm-vars true --mlp-init $mlp_init --hid-layers 0 --learn-rate 0.008 --use-gpu-id 0 \
    feat/fbank/train_multi feat/fbank/dev_multi data/lang $ali $ali_dev $dir || exit 1;
  utils/mkgraph.sh data/lang_bcb05cnp exp_multi/tri3d_dnn exp_multi/tri3d_dnn/graph_bg || exit 1;
  log_end "tri3d [train]"
}
train_tri3d

