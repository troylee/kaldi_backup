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

aurora4=/media/research/corpus/aurora4
#we need lm, trans, from WSJ0 CORPUS
wsj0=/media/research/corpus/WSJ0

prepare_basic(){
log_start "data preparation"
local/aurora4_prep_data.sh $aurora4 $wsj0
log_end "data preparation"

log_start "dictionary preparation"
local/aurora4_prep_dict.sh || exit 1;
log_end "dictionary preparation"

log_start "lang preparation"
utils/prepare_lang.sh --position-dependent-phones false data/local/dict "<SPOKEN_NOISE>" data/local/lang_tmp data/lang || exit 1;
log_end "lang preparation"

log_start "format data"
local/aurora4_prep_testlm.sh || exit 1;
log_end "format_data"

# Now make MFCC features.
# mfccdir should be some place with a largish disk where you
# want to store MFCC features.
log_start "MFCC extraction"
mfccdir=feat/mfcc
mkdir -p $mfccdir
for x in train_clean train_multi dev_clean dev_multi dev{01..14} test{01..14} ; do 
  if [ -d $mfccdir/$x ]; then
    rm -r $mfccdir/$x
  fi
  cp -r data/$x $mfccdir/$x
  steps/make_mfcc.sh --nj 1 $mfccdir/$x exp/make_mfcc/$x $mfccdir/params || exit 1;
  steps/compute_cmvn_utt_stats.sh $mfccdir/$x exp/make_mfcc/$x $mfccdir/params || exit 1;
done
log_end "MFCC extraction"

# make fbank features
log_start "FBank extraction"
fbankdir=feat/fbank
mkdir -p $fbankdir
for x in train_clean train_multi dev_clean dev_multi dev{01..14} test{01..14} ; do
  if [ -d $fbankdir/$x ]; then
    rm -r $fbankdir/$x
  fi
  cp -r data/$x $fbankdir/$x
  steps/make_fbank.sh --nj 1 $fbankdir/$x exp/make_fbank/$x $fbankdir/params || exit 1;
  steps/compute_cmvn_utt_stats.sh $fbankdir/$x exp/make_fbank/$x $fbankdir/params || exit 1;
done
log_end "FBank extraction"
}

clean_train(){
###########################################
# Model training - clean
#

# Note: the --boost-silence option should probably be omitted by default
# for normal setups.  It doesn't always help. [it's to discourage non-silence
# models from modeling silence.]

log_start "mono [train]"
steps/aurora4_train_mono.sh --boost-silence 1.25 --nj 4 --norm_vars true \
  feat/mfcc/train_clean data/lang exp_clean/mono || exit 1;
log_end "mono [train]"

log_start "mono [align]"
steps/aurora4_align_si.sh --boost-silence 1.25 --nj 4  \
   feat/mfcc/train_clean data/lang exp_clean/mono exp_clean/mono_ali || exit 1;
log_end "mono [align]"

log_start "tri1a [train]"
steps/aurora4_train_deltas.sh --boost-silence 1.25 --norm_vars true \
    4200 35000 feat/mfcc/train_clean data/lang exp_clean/mono_ali exp_clean/tri1a || exit 1;
log_end "tri1a [train]"

log_start "tri1a [decode]"
utils/mkgraph.sh data/lang_bcb05cnp exp_clean/tri1a exp_clean/tri1a/graph_bg || exit 1;
for x in test{01..14} ; do
  steps/aurora4_decode_deltas.sh --nj 4 exp_clean/tri1a/graph_bg feat/mfcc/${x} exp_clean/tri1a/decode_bg_${x} || exit 1;
done
log_end "tri1a [decode]"
}

multi_train(){
###########################################
# Model training - multi
#

# Note: the --boost-silence option should probably be omitted by default
# for normal setups.  It doesn't always help. [it's to discourage non-silence
# models from modeling silence.]

log_start "mono [train]"
steps/aurora4_train_mono.sh --boost-silence 1.25 --nj 4 --norm_vars true \
  feat/mfcc/train_multi data/lang exp_multi/mono || exit 1;
log_end "mono [train]"

log_start "mono [align]"
steps/aurora4_align_si.sh --boost-silence 1.25 --nj 4  \
   feat/mfcc/train_multi data/lang exp_multi/mono exp_multi/mono_ali || exit 1;
log_end "mono [align]"

log_start "tri1a [train]"
steps/aurora4_train_deltas.sh --boost-silence 1.25 --norm_vars true \
    4200 35000 feat/mfcc/train_multi data/lang exp_multi/mono_ali exp_multi/tri1a || exit 1;
log_end "tri1a [train]"

log_start "tri1a [decode]"
utils/mkgraph.sh data/lang_bcb05cnp exp_multi/tri1a exp_multi/tri1a/graph_bg || exit 1;

for x in test{01..14} ; do
  steps/aurora4_decode_deltas.sh --nj 4 exp_multi/tri1a/graph_bg feat/mfcc/${x} exp_multi/tri1a/decode_bg_${x} || exit 1;
done
log_end "tri1a [decode]"

# align multi-style data with multi-trained model, needs a larger beam
log_start "tri1a [align-train-multi]"
steps/aurora4_align_si.sh --nj 4 --retry-beam 60 feat/mfcc/train_multi data/lang exp_multi/tri1a exp_multi/tri1a_ali/train_multi || exit 1;
log_end "tri1a [align-train-multi]"

log_start "tri1a [align-dev-multi]"
steps/aurora4_align_si.sh --nj 4 --retry-beam 80 feat/mfcc/dev_multi data/lang exp_multi/tri1a exp_multi/tri1a_ali/dev_multi || exit 1;
log_end "tri1a [align-dev-multi]"

# align clean data with multi-trained model
log_start "tri1a [align-train-clean]"
steps/aurora4_align_si.sh --nj 4 feat/mfcc/train_clean data/lang exp_multi/tri1a exp_multi/tri1a_ali/train_clean || exit 1;
log_end "tri1a [align-train-clean]"

log_start "tri1a [align-dev-clean]"
steps/aurora4_align_si.sh --nj 4 feat/mfcc/dev_clean data/lang exp_multi/tri1a exp_multi/tri1a_ali/dev_clean || exit 1;
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

###############################################
#Now begin train DNN systems on multi data

#RBM pretrain
log_start "tri2a [pretrain]"
dir=exp_multi/tri2a_dnn_pretrain
mkdir -p $dir/log
steps/aurora4_pretrain_dbn.sh --nn-depth 7 --rbm-iter 3 --norm-vars true feat/fbank/train_multi $dir
log_end "tri2a [pretrain]"
