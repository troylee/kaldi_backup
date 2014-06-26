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

###############################################
# DNN fine-tuning with multi-style aligned labels
# 6H

train_tri5a(){
  log_start "tri5a [train]"
  dir=exp_multi/tri5a_dnn
  ali=exp_multi/tri1a_ali/train_multi
  ali_dev=exp_multi/tri1a_ali/dev_multi
  dbn=exp_multi/tri2a_dnn_pretrain/6.dbn
  mkdir -p $dir/log
  steps/aurora4/nnet_train_frmacc.sh --norm-vars true --dbn $dbn --hid-layers 0 --learn-rate 0.015 --use-gpu-id 0 \
    feat/fbank/train_multi feat/fbank/dev_multi data/lang $ali $ali_dev $dir || exit 1;
  utils/mkgraph.sh data/lang_bcb05cnp exp_multi/tri5a_dnn exp_multi/tri5a_dnn/graph_bg || exit 1;
  log_end "tri5a [train]" 
}
train_tri5a

align_tri5a(){
  #nnet realignments
  log_start "tri5a [realign-train_multi]"
  steps/aurora4/nnet_align.sh --nj 4 --retry-beam 60 feat/fbank/train_multi data/lang exp_multi/tri5a_dnn exp_multi/tri5a_dnn_ali/train_multi || exit 1;
  log_end "tri5a [realign-train_multi]"

  log_start "tri5a [realign-dev_multi]"
  steps/aurora4/nnet_align.sh --nj 4 --retry-beam 80 feat/fbank/dev_multi data/lang exp_multi/tri5a_dnn exp_multi/tri5a_dnn_ali/dev_multi || exit 1;
  log_end "tri5a [realign-dev_multi]"
}
#align_tri5a

train_tri5b(){
  log_start "tri5b [train]"
  dir=exp_multi/tri5b_dnn
  ali=exp_multi/tri5a_dnn_ali/train_multi
  ali_dev=exp_multi/tri5a_dnn_ali/dev_multi
  mlp_init=exp_multi/tri5a_dnn/nnet_6.dbn_dnn.init
  mkdir -p $dir/log
  steps/aurora4/nnet_train.sh --norm-vars true --mlp-init $mlp_init --hid-layers 0 --learn-rate 0.008 --use-gpu-id 0 \
    feat/fbank/train_multi feat/fbank/dev_multi data/lang $ali $ali_dev $dir || exit 1;
  utils/mkgraph.sh data/lang_bcb05cnp $dir $dir/graph_bg || exit 1;
  log_end "tri5b [train]"
}
#train_tri5b

align_tri5b(){
  #nnet realignments
  log_start "tri5b [realign-train_multi]"
  steps/aurora4/nnet_align.sh --nj 4 --retry-beam 60 feat/fbank/train_multi data/lang exp_multi/tri5b_dnn exp_multi/tri5b_dnn_ali/train_multi || exit 1;
  log_end "tri5b [realign-train_multi]"

  log_start "tri5b [realign-dev_multi]"
  steps/aurora4/nnet_align.sh --nj 4 --retry-beam 80 feat/fbank/dev_multi data/lang exp_multi/tri5b_dnn exp_multi/tri5b_dnn_ali/dev_multi || exit 1;
  log_end "tri5b [realign-dev_multi]"
}
#align_tri5b

train_tri5c(){
  log_start "tri5c [train]"
  dir=exp_multi/tri5c_dnn
  ali=exp_multi/tri5b_dnn_ali/train_multi
  ali_dev=exp_multi/tri5b_dnn_ali/dev_multi
  mlp_init=exp_multi/tri5a_dnn/nnet_6.dbn_dnn.init
  mkdir -p $dir/log
  steps/aurora4/nnet_train.sh --norm-vars true --mlp-init $mlp_init --hid-layers 0 --learn-rate 0.008 --use-gpu-id 0 \
    feat/fbank/train_multi feat/fbank/dev_multi data/lang $ali $ali_dev $dir || exit 1;
  utils/mkgraph.sh data/lang_bcb05cnp $dir $dir/graph_bg || exit 1;
  log_end "tri5c [train]"
}
#train_tri5c

align_tri5c(){
  #nnet realignments
  log_start "tri5c [realign-train_multi]"
  steps/aurora4/nnet_align.sh --nj 4 --retry-beam 60 feat/fbank/train_multi data/lang exp_multi/tri5c_dnn exp_multi/tri5c_dnn_ali/train_multi || exit 1;
  log_end "tri5c [realign-train_multi]"

  log_start "tri5c [realign-dev_multi]"
  steps/aurora4/nnet_align.sh --nj 4 --retry-beam 80 feat/fbank/dev_multi data/lang exp_multi/tri5c_dnn exp_multi/tri5c_dnn_ali/dev_multi || exit 1;
  log_end "tri5c [realign-dev_multi]"
}
#align_tri5c

train_tri5d(){
  log_start "tri5d [train]"
  dir=exp_multi/tri5d_dnn
  ali=exp_multi/tri5c_dnn_ali/train_multi
  ali_dev=exp_multi/tri5c_dnn_ali/dev_multi
  mlp_init=exp_multi/tri5a_dnn/nnet_6.dbn_dnn.init
  mkdir -p $dir/log
  steps/aurora4/nnet_train.sh --norm-vars true --mlp-init $mlp_init --hid-layers 0 --learn-rate 0.008 --use-gpu-id 0 \
    feat/fbank/train_multi feat/fbank/dev_multi data/lang $ali $ali_dev $dir || exit 1;
  utils/mkgraph.sh data/lang_bcb05cnp $dir $dir/graph_bg || exit 1;
  log_end "tri5d [train]"
}
#train_tri5d

