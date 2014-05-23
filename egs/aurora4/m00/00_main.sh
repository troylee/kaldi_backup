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
# data preparations
#

aurora4=/hpctmp/asr390305000/aurora4
#we need lm, trans, from WSJ0 CORPUS
wsj0=/hpctmp/asr390305000/WSJ0

pre(){

#the data preparations and feature extractions are done
# on speech5.

log_start "data preparation"
local/aurora4_data_prep.sh $aurora4 $wsj0
log_end "data preparation"

log_start "dictionary preparation"
local/wsj_prepare_dict.sh || exit 1;
log_end "dictionary preparation"

log_start "lang preparation"
utils/prepare_lang.sh data/local/dict "<SPOKEN_NOISE>" data/local/lang_tmp data/lang || exit 1;
log_end "lang preparation"

log_start "format data"
local/aurora4_format_data.sh || exit 1;
log_end "format_data"

# Now make MFCC features.
# mfccdir should be some place with a largish disk where you
# want to store MFCC features.
log_start "MFCC extraction"
mfccdir=data/mfcc
mkdir -p $mfccdir
for x in train_si84_clean train_si84_multi dev_0330 dev_1206 test_eval92 ; do 
  cp -r data/$x $mfccdir/$x
  steps/make_mfcc.sh  --nj 4 $mfccdir/$x exp/make_mfcc/$x $mfccdir/params || exit 1;
  steps/compute_cmvn_stats.sh $mfccdir/$x exp/make_mfcc/$x $mfccdir/params || exit 1;
done
log_end "MFCC extraction"

# make fbank features
log_start "FBank extraction"
fbankdir=data/fbank
mkdir -p $fbankdir
for x in train_si84_clean train_si84_multi dev_0330 dev_1206 test_eval92 ; do
  cp -r data/$x $fbankdir/$x
  steps/make_fbank.sh --nj 4 $fbankdir/$x exp/make_fbank/$x $fbankdir/params || exit 1;
  steps/compute_cmvn_stats.sh $fbankdir/$x exp/make_fbank/$x $fbankdir/params || exit 1;
done
log_end "FBank extraction"

###########################################
# Model training
#

# Note: the --boost-silence option should probably be omitted by default
# for normal setups.  It doesn't always help. [it's to discourage non-silence
# models from modeling silence.]

log_start "mono [train]"
steps/train_mono.sh --boost-silence 1.25 --nj 4  \
  data/mfcc/train_si84_multi data/lang exp/mono0a_multi || exit 1;
log_end "mono [train]"

log_start "mono [align]"
steps/align_si.sh --boost-silence 1.25 --nj 4  \
   data/mfcc/train_si84_multi data/lang exp/mono0a_multi exp/mono0a_multi_ali || exit 1;
log_end "mono [align]"

log_start "tri1 [train]"
steps/train_deltas.sh --boost-silence 1.25 \
    2000 10000 data/mfcc/train_si84_multi data/lang exp/mono0a_multi_ali exp/tri1_multi || exit 1;
log_end "tri1 [train]"

log_start "tri1 [align]"
steps/align_si.sh --nj 4 \
  data/mfcc/train_si84_multi data/lang exp/tri1_multi exp/tri1_multi_ali_si84 || exit 1;
log_end "tri1 [align]"

log_start "tri2a [train]"
steps/train_deltas.sh  \
  2500 15000 data/mfcc/train_si84_multi data/lang exp/tri1_multi_ali_si84 exp/tri2a_multi || exit 1;
log_end "tri2a [train]"

log_start "tri2b [train]"
steps/train_lda_mllt.sh \
   --splice-opts "--left-context=3 --right-context=3" \
   2500 15000 data/mfcc/train_si84_multi data/lang exp/tri1_multi_ali_si84 exp/tri2b_multi || exit 1;
log_end "tri2b [train]"

log_start "tri2b [decode]"
utils/mkgraph.sh data/lang_test_tgpr exp/tri2b_multi exp/tri2b_multi/graph_tgpr || exit 1;
steps/decode.sh --nj 4 \
  exp/tri2b_multi/graph_tgpr data/mfcc/test_eval92 exp/tri2b_multi/decode_tgpr_eval92 || exit 1;
log_end "tri2b [decode]"


# Align tri2b system with si84 multi-condition data.
log_start "tri2b [align-train]"
steps/align_si.sh  --nj 4 \
  --use-graphs true data/mfcc/train_si84_multi data/lang exp/tri2b_multi exp/tri2b_multi_ali_si84  || exit 1;
log_end "tri2b [align-train]"

log_start "tri2b [align-dev_0330]"
steps/align_si.sh  --nj 4 \
  data/mfcc/dev_0330 data/lang exp/tri2b_multi exp/tri2b_multi_ali_dev_0330 || exit 1;
log_end "tri2b [align-dev_0330]"

log_start "tri2b [align-dev_1206]"
steps/align_si.sh  --nj 4 \
  data/mfcc/dev_1206 data/lang exp/tri2b_multi exp/tri2b_multi_ali_dev_1206 || exit 1;
log_end "tri2b [align-dev_1206]"

###############################################
#Now begin train DNN systems on multi data

#RBM pretrain
dir=exp/tri3a_dnn_pretrain
mkdir -p $dir/log

log_start "tri3a [pretrain]"
steps/nnet/pretrain_dbn.sh --nn-depth 7 --rbm-iter 3 data/fbank/train_si84_multi $dir
log_end "tri3a [pretrain]"
}

dir=exp/tri3a_dnn
ali=exp/tri2b_multi_ali_si84
ali_dev=exp/tri2b_multi_ali_dev_0330
feature_transform=exp/tri3a_dnn_pretrain/final.feature_transform
dbn=exp/tri3a_dnn_pretrain/7.dbn
mkdir -p $dir/log
log_start "tri3a [train]"
steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.008 --use-gpu-id 0 \
  data/fbank/train_si84_multi data/fbank/dev_0330 data/lang $ali $ali_dev $dir || exit 1;
log_end "tri3a [train]"

log_start "tri3a [decode]"
utils/mkgraph.sh data/lang_test_tgpr exp/tri3a_dnn exp/tri3a_dnn/graph_tgpr || exit 1;
dir=exp/tri3a_dnn
steps/nnet/decode.sh --nj 4 --acwt 0.10 --config conf/decode_dnn.config \
  exp/tri3a_dnn/graph_tgpr data/fbank/test_eval92 $dir/decode_tgpr_eval92 || exit 1;
log_end "tri3a [decode]"

#realignments
srcdir=exp/tri3a_dnn
log_start "tri3a [realign-train]"
steps/nnet/align.sh --nj 4 \
  data/fbank/train_si84_multi data/lang $srcdir ${srcdir}_ali_si84_multi || exit 1;
log_end "tri3a [realign-train]"

log_start "tri3a [realign-dev_0330]"
steps/nnet/align.sh --nj 4 \
  data/fbank/dev_0330 data/lang $srcdir ${srcdir}_ali_dev_0330 || exit 1;
log_end "tri3a [realign-dev_0330]"

#train system again 

dir=exp/tri4a_dnn
ali=exp/tri3a_dnn_ali_si84_multi
ali_dev=exp/tri3a_dnn_ali_dev_0330
feature_transform=exp/tri3a_dnn_pretrain/final.feature_transform
dbn=exp/tri3a_dnn_pretrain/7.dbn
mkdir -p $dir/log
log_start "tri4a [train]"
steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.008 \
  data/fbank/train_si84_multi data/fbank/dev_0330 data/lang $ali $ali_dev $dir || exit 1;
log_end "tri4a [train]"

log_start "tri4a [decode]"
utils/mkgraph.sh data/lang_test_tgpr exp/tri4a_dnn exp/tri4a_dnn/graph_tgpr || exit 1;
dir=exp/tri4a_dnn
steps/nnet/decode.sh --nj 4 --acwt 0.10 --config conf/decode_dnn.config \
  exp/tri4a_dnn/graph_tgpr data/fbank/test_eval92 $dir/decode_tgpr_eval92 || exit 1;
log_end "tri4a [decode]"


# DNN Sequential DT training
#......

