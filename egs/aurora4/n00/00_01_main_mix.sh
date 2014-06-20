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

###############################################
# DNN fine-tuning with labels obtained from aligning the corresponding clean data
# 
# Requires exp_multi/tri2a_dnn_pretrain be finished.
#

train_tri4a(){
  # DNN fine-tuning with clean aligned labels
  log_start "tri4a [train]"
  dir=exp_multi/tri4a_dnn
  ali=exp_multi/tri1a_ali/train_clean
  ali_dev=exp_multi/tri1a_ali/dev_clean
  dbn=exp_multi/tri2a_dnn_pretrain/7.dbn
  mkdir -p $dir/log
  steps/aurora4/nnet_train.sh --norm-vars true --dbn $dbn --hid-layers 0 --learn-rate 0.008 --use-gpu-id 0 \
    feat/fbank/train_multi feat/fbank/dev_multi data/lang $ali $ali_dev $dir || exit 1;
  utils/mkgraph.sh data/lang_bcb05cnp exp_multi/tri4a_dnn exp_multi/tri4a_dnn/graph_bg || exit 1;
  log_end "tri4a [train]"
}
train_tri4a

decode_tri4a(){
  log_start "tri4a [decode]"
  for i in `seq -f "%02g" 1 14`; do
    x=test${i}
    steps/aurora4/nnet_decode.sh --nj 4 --acwt 0.10 --config conf/decode_dnn.config --srcdir exp_multi/tri4a_dnn exp_multi/tri4a_dnn/graph_bg feat/fbank/${x} exp_multi/tri4a_dnn/decode/decode_bg_${x} || exit 1;
  done
  local/average_wer.sh 'exp_multi/tri4a_dnn/decode/decode_bg_test*' | tee exp_multi/tri4a_dnn/decode/decode_bg_test.avgwer
  log_end "tri4a [decode]"
}
#decode_tri4a

align_tri4a(){
  #nnet realignments
  log_start "tri4a [realign-train_clean]"
  steps/aurora4/nnet_align.sh --nj 4 feat/fbank/train_clean data/lang exp_multi/tri4a_dnn exp_multi/tri4a_dnn_ali/train_clean || exit 1;
  log_end "tri4a [realign-train_clean]"

  log_start "tri4a [realign-dev_clean]"
  steps/aurora4/nnet_align.sh --nj 4 feat/fbank/dev_clean data/lang exp_multi/tri4a_dnn exp_multi/tri4a_dnn_ali/dev_clean || exit 1;
  log_end "tri4a [realign-dev_clean]"

  # additional processing of the clean data alignments for used as multi labels
  dir=exp_multi/tri4a_dnn_ali/train_clean
  mkdir -p ${dir}/ori
  for j in {1..4}; do
    mv ${dir}/ali.${j}.gz ${dir}/ori/ali.${j}.gz
    utils/convert_ali_names.py feat/mfcc/train_multi/feats.scp ${dir}/ori/ali.${j}.gz ${dir}/ali.${j}.gz
  done

  dir=exp_multi/tri4a_dnn_ali/dev_clean
  mkdir -p ${dir}/ori
  for j in {1..4}; do
    mv ${dir}/ali.${j}.gz ${dir}/ori/ali.${j}.gz
    utils/convert_ali_names.py feat/mfcc/dev_multi/feats.scp ${dir}/ori/ali.${j}.gz ${dir}/ali.${j}.gz
  done

  # sanity check for the genreated clean frame alignment
  ./utils/alignment_frame_checking.sh exp_multi/tri4a_dnn_ali/train_clean/ exp_multi/tri1a_ali/train_multi/
  ./utils/alignment_frame_checking.sh exp_multi/tri4a_dnn_ali/dev_clean/ exp_multi/tri1a_ali/dev_multi/
}
align_tri4a

train_tri4b(){
  # DNN fine-tuning with clean aligned labels
  log_start "tri4b [train]"
  dir=exp_multi/tri4b_dnn
  ali=exp_multi/tri4a_dnn_ali/train_clean
  ali_dev=exp_multi/tri4a_dnn_ali/dev_clean
  mlp_init=exp_multi/tri4a_dnn/nnet_7.dbn_dnn.init
  mkdir -p $dir/log
  steps/aurora4/nnet_train.sh --norm-vars true --mlp-init $mlp_init --hid-layers 0 --learn-rate 0.008 --use-gpu-id 0 \
    feat/fbank/train_multi feat/fbank/dev_multi data/lang $ali $ali_dev $dir || exit 1;
  utils/mkgraph.sh data/lang_bcb05cnp exp_multi/tri4b_dnn exp_multi/tri4b_dnn/graph_bg || exit 1;
  log_end "tri4b [train]"
}
train_tri4b

align_tri4b(){
  #nnet realignments
  log_start "tri4b [realign-train_clean]"
  steps/aurora4/nnet_align.sh --nj 4 feat/fbank/train_clean data/lang exp_multi/tri4b_dnn exp_multi/tri4b_dnn_ali/train_clean || exit 1;
  log_end "tri4b [realign-train_clean]"

  log_start "tri4b [realign-dev_clean]"
  steps/aurora4/nnet_align.sh --nj 4 feat/fbank/dev_clean data/lang exp_multi/tri4b_dnn exp_multi/tri4b_dnn_ali/dev_clean || exit 1;
  log_end "tri4b [realign-dev_clean]"

  # additional processing of the clean data alignments for used as multi labels
  dir=exp_multi/tri4b_dnn_ali/train_clean
  mkdir -p ${dir}/ori
  for j in {1..4}; do
    mv ${dir}/ali.${j}.gz ${dir}/ori/ali.${j}.gz
    utils/convert_ali_names.py feat/mfcc/train_multi/feats.scp ${dir}/ori/ali.${j}.gz ${dir}/ali.${j}.gz
  done

  dir=exp_multi/tri4b_dnn_ali/dev_clean
  mkdir -p ${dir}/ori
  for j in {1..4}; do
    mv ${dir}/ali.${j}.gz ${dir}/ori/ali.${j}.gz
    utils/convert_ali_names.py feat/mfcc/dev_multi/feats.scp ${dir}/ori/ali.${j}.gz ${dir}/ali.${j}.gz
  done

  # sanity check for the genreated clean frame alignment
  ./utils/alignment_frame_checking.sh exp_multi/tri4b_dnn_ali/train_clean/ exp_multi/tri1a_ali/train_multi/
  ./utils/alignment_frame_checking.sh exp_multi/tri4b_dnn_ali/dev_clean/ exp_multi/tri1a_ali/dev_multi/
}
align_tri4b

train_tri4c(){
  log_start "tri4c [train]"
  dir=exp_multi/tri4c_dnn
  ali=exp_multi/tri4b_dnn_ali/train_clean
  ali_dev=exp_multi/tri4b_dnn_ali/dev_clean
  mlp_init=exp_multi/tri4a_dnn/nnet_7.dbn_dnn.init
  mkdir -p $dir/log
  steps/aurora4/nnet_train.sh --norm-vars true --mlp-init $mlp_init --hid-layers 0 --learn-rate 0.008 --use-gpu-id 0 \
    feat/fbank/train_multi feat/fbank/dev_multi data/lang $ali $ali_dev $dir || exit 1;
  utils/mkgraph.sh data/lang_bcb05cnp exp_multi/tri4c_dnn exp_multi/tri4c_dnn/graph_bg || exit 1;
  log_end "tri4c [train]"
}
train_tri4c

align_tri4c(){
  #nnet realignments
  log_start "tri4c [realign-train_clean]"
  steps/aurora4/nnet_align.sh --nj 4 feat/fbank/train_clean data/lang exp_multi/tri4c_dnn exp_multi/tri4c_dnn_ali/train_clean || exit 1;
  log_end "tri4c [realign-train_clean]"

  log_start "tri4c [realign-dev_clean]"
  steps/aurora4/nnet_align.sh --nj 4 feat/fbank/dev_clean data/lang exp_multi/tri4c_dnn exp_multi/tri4c_dnn_ali/dev_clean || exit 1;
  log_end "tri4c [realign-dev_clean]"

  # additional processing of the clean data alignments for used as multi labels
  dir=exp_multi/tri4c_dnn_ali/train_clean
  mkdir -p ${dir}/ori
  for j in {1..4}; do
    mv ${dir}/ali.${j}.gz ${dir}/ori/ali.${j}.gz
    utils/convert_ali_names.py feat/mfcc/train_multi/feats.scp ${dir}/ori/ali.${j}.gz ${dir}/ali.${j}.gz
  done

  dir=exp_multi/tri4c_dnn_ali/dev_clean
  mkdir -p ${dir}/ori
  for j in {1..4}; do
    mv ${dir}/ali.${j}.gz ${dir}/ori/ali.${j}.gz
    utils/convert_ali_names.py feat/mfcc/dev_multi/feats.scp ${dir}/ori/ali.${j}.gz ${dir}/ali.${j}.gz
  done

  # sanity check for the genreated clean frame alignment
  ./utils/alignment_frame_checking.sh exp_multi/tri4c_dnn_ali/train_clean/ exp_multi/tri1a_ali/train_multi/
  ./utils/alignment_frame_checking.sh exp_multi/tri4c_dnn_ali/dev_clean/ exp_multi/tri1a_ali/dev_multi/
}
align_tri4c

train_tri4d(){
  log_start "tri4d [train]"
  dir=exp_multi/tri4d_dnn
  ali=exp_multi/tri4c_dnn_ali/train_clean
  ali_dev=exp_multi/tri4c_dnn_ali/dev_clean
  mlp_init=exp_multi/tri4a_dnn/nnet_7.dbn_dnn.init
  mkdir -p $dir/log
  steps/aurora4/nnet_train.sh --norm-vars true --mlp-init $mlp_init --hid-layers 0 --learn-rate 0.008 --use-gpu-id 0 \
    feat/fbank/train_multi feat/fbank/dev_multi data/lang $ali $ali_dev $dir || exit 1;
  utils/mkgraph.sh data/lang_bcb05cnp exp_multi/tri4d_dnn exp_multi/tri4d_dnn/graph_bg || exit 1;
  log_end "tri4d [train]"
}
train_tri4d

