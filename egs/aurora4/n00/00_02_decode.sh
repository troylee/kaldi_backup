#!/bin/bash
# Author: Bo Li (li-bo@outlook.com)
#
# This script is specific to our servers for decoding, which is also included in the 
# main script. 
#
cwd=~/tools/kaldi/egs/aurora4/n00
cd $cwd

nodes=( compg2 compg5 compg6 compg7 compg8 compg9 compg10 )

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

decode_clean_tri1a(){
  # decode exp_clean/tri1a
  log_start "tri1a [decode]"
  for i in {0..6} ; do 
    printf -v x 'test%02g' $((i+1))
    echo ${nodes[$i]} $x
    ( ssh ${nodes[$i]} "cd $cwd; steps/aurora4/decode_deltas.sh --nj 5 --srcdir exp_clean/tri1a exp_clean/tri1a/graph_bg feat/mfcc/${x} exp_clean/tri1a/decode/decode_bg_${x}" ) &
    
    printf -v x 'test%02g' $((i+8))
    echo ${nodes[$i]} $x
    ( ssh ${nodes[$i]} "cd $cwd; steps/aurora4/decode_deltas.sh --nj 5 --srcdir exp_clean/tri1a exp_clean/tri1a/graph_bg feat/mfcc/${x} exp_clean/tri1a/decode/decode_bg_${x}" ) &
  done
  wait;
  # write out the average WER results
  local/average_wer.sh 'exp_clean/tri1a/decode/decode_bg_test*' | tee exp_clean/tri1a/decode/decode_bg_test.avgwer
  log_end "tri1a [decode]"
}
#decode_clean_tri1a

decode_clean_tri1b(){
  log_start "tri1b [decode]"
  for i in {0..6} ; do
    printf -v x 'test%02g' $((i+1))
    echo ${nodes[$i]} $x
    ( ssh ${nodes[$i]} "cd $cwd; steps/aurora4/decode_deltas.sh --nj 5 --srcdir exp_clean/tri1b exp_clean/tri1b/graph_bg feat/mfcc/${x} exp_clean/tri1b/decode/decode_bg_${x}" ) &

    printf -v x 'test%02g' $((i+8))
    echo ${nodes[$i]} $x
    ( ssh ${nodes[$i]} "cd $cwd; steps/aurora4/decode_deltas.sh --nj 5 --srcdir exp_clean/tri1b exp_clean/tri1b/graph_bg feat/mfcc/${x} exp_clean/tri1b/decode/decode_bg_${x}" ) &
  done
  wait;
  # write out the average WER results
  local/average_wer.sh 'exp_clean/tri1b/decode/decode_bg_test*' | tee exp_clean/tri1b/decode/decode_bg_test.avgwer
  log_end "tri1b [decode]"
}
#decode_clean_tri1b

decode_clean_tri1b_vtsmodel(){
  log_start "tri1b [vtsmodel decode]"
  for i in {0..6} ; do
    printf -v x 'test%02g' $((i+1))
    echo ${nodes[$i]} $x
    ( ssh ${nodes[$i]} "cd $cwd; steps/aurora4/decode_vts_model.sh --nj 5 --srcdir exp_clean/tri1b exp_clean/tri1b/graph_bg feat/mfcc/${x} exp_clean/tri1b/decode_vts_model/decode_bg_${x}" ) &

    printf -v x 'test%02g' $((i+8))
    echo ${nodes[$i]} $x
    ( ssh ${nodes[$i]} "cd $cwd; steps/aurora4/decode_vts_model.sh --nj 5 --srcdir exp_clean/tri1b exp_clean/tri1b/graph_bg feat/mfcc/${x} exp_clean/tri1b/decode_vts_model/decode_bg_${x}" ) &
  done
  wait;
  # write out the average WER results
  local/average_wer.sh 'exp_clean/tri1b/decode_vts_model/decode_bg_test*' | tee exp_clean/tri1b/decode_vts_model/decode_bg_test.avgwer
  log_end "tri1b [vtsmodel decode]"
}
#decode_clean_tri1b_vtsmodel


decode_multi_tri1a(){
  # decode exp_multi/tri1a
  log_start "tri1a [decode]"
  for i in {0..6} ; do 
    printf -v x 'test%02g' $((i+1))
    echo ${nodes[$i]} $x
    ( ssh ${nodes[$i]} "cd $cwd; steps/aurora4/decode_deltas.sh --nj 5 --srcdir exp_multi/tri1a exp_multi/tri1a/graph_bg feat/mfcc/${x} exp_multi/tri1a/decode/decode_bg_${x}" ) &
    
    printf -v x 'test%02g' $((i+8))
    echo ${nodes[$i]} $x
    ( ssh ${nodes[$i]} "cd $cwd; steps/aurora4/decode_deltas.sh --nj 5 --srcdir exp_multi/tri1a exp_multi/tri1a/graph_bg feat/mfcc/${x} exp_multi/tri1a/decode/decode_bg_${x}" ) &
  done
  wait;
  # write out the average WER results
  local/average_wer.sh 'exp_multi/tri1a/decode/decode_bg_test*' | tee exp_multi/tri1a/decode/decode_bg_test.avgwer
  log_end "tri1a [decode]"
}
#decode_multi_tri1a

decode_multi_tri1b(){
  log_start "tri1b [decode]"
  for i in {0..6} ; do
    printf -v x 'test%02g' $((i+1))
    echo ${nodes[$i]} $x
    ( ssh ${nodes[$i]} "cd $cwd; steps/aurora4/decode_deltas.sh --nj 5 --srcdir exp_multi/tri1b exp_multi/tri1b/graph_bg feat/mfcc/${x} exp_multi/tri1b/decode/decode_bg_${x}" ) &

    printf -v x 'test%02g' $((i+8))
    echo ${nodes[$i]} $x
    ( ssh ${nodes[$i]} "cd $cwd; steps/aurora4/decode_deltas.sh --nj 5 --srcdir exp_multi/tri1b exp_multi/tri1b/graph_bg feat/mfcc/${x} exp_multi/tri1b/decode/decode_bg_${x}" ) &
  done
  wait;
  # write out the average WER results
  local/average_wer.sh 'exp_multi/tri1b/decode/decode_bg_test*' | tee exp_multi/tri1b/decode/decode_bg_test.avgwer
  log_end "tri1b [decode]"
}
#decode_multi_tri1b

decode_multi_tri1b_vtsmodel(){
  log_start "tri1b [vtsmodel decode]"
  for i in {0..6} ; do
    printf -v x 'test%02g' $((i+1))
    echo ${nodes[$i]} $x
    ( ssh ${nodes[$i]} "cd $cwd; steps/aurora4/decode_vts_model.sh --nj 5 --srcdir exp_multi/tri1b exp_multi/tri1b/graph_bg feat/mfcc/${x} exp_multi/tri1b/decode_vts_model/decode_bg_${x}" ) &

    printf -v x 'test%02g' $((i+8))
    echo ${nodes[$i]} $x
    ( ssh ${nodes[$i]} "cd $cwd; steps/aurora4/decode_vts_model.sh --nj 5 --srcdir exp_multi/tri1b exp_multi/tri1b/graph_bg feat/mfcc/${x} exp_multi/tri1b/decode_vts_model/decode_bg_${x}" ) &
  done
  wait;
  # write out the average WER results
  local/average_wer.sh 'exp_multi/tri1b/decode_vts_model/decode_bg_test*' | tee exp_multi/tri1b/decode_vts_model/decode_bg_test.avgwer
  log_end "tri1b [vtsmodel decode]"
}
#decode_multi_tri1b_vtsmodel

decode_multi_tri3a(){
  log_start "tri3a [decode]"
  dir=exp_multi/tri3a_dnn
  for i in {0..6} ; do 
    printf -v x 'test%02g' $((i+1))
    echo ${nodes[$i]} $x
    ( ssh ${nodes[$i]} "cd $cwd; steps/aurora4/nnet_decode.sh --nj 5 --srcdir $dir $dir/graph_bg feat/fbank/${x} $dir/decode/decode_bg_${x}" ) &
    
    printf -v x 'test%02g' $((i+8))
    echo ${nodes[$i]} $x
    ( ssh ${nodes[$i]} "cd $cwd; steps/aurora4/nnet_decode.sh --nj 5 --srcdir $dir $dir/graph_bg feat/fbank/${x} $dir/decode/decode_bg_${x}" ) &
  done
  wait;
  # write out the average WER results
  local/average_wer.sh --min-lmwt 9 --max-lmwt 20 "$dir/decode/decode_bg_test*" | tee $dir/decode/decode_bg_test.avgwer
  log_end "tri3a [decode]"
}
#decode_multi_tri3a

decode_multi_tri4a(){
  log_start "tri4a [decode]"
  dir=exp_multi/tri4a_dnn
  for i in {0..6} ; do
    printf -v x 'test%02g' $((i+1))
    echo ${nodes[$i]} $x
    ( ssh ${nodes[$i]} "cd $cwd; steps/aurora4/nnet_decode.sh --nj 5 --srcdir $dir $dir/graph_bg feat/fbank/${x} $dir/decode/decode_bg_${x}" ) &

    printf -v x 'test%02g' $((i+8))
    echo ${nodes[$i]} $x
    ( ssh ${nodes[$i]} "cd $cwd; steps/aurora4/nnet_decode.sh --nj 5 --srcdir $dir $dir/graph_bg feat/fbank/${x} $dir/decode/decode_bg_${x}" ) &
  done
  wait;
  # write out the average WER results
  local/average_wer.sh --min-lmwt 9 --max-lmwt 20 "$dir/decode/decode_bg_test*" | tee $dir/decode/decode_bg_test.avgwer
  log_end "tri4a [decode]"
}
#decode_multi_tri4a

decode_multi_tri5a(){
  log_start "tri5a [decode]"
  dir=exp_multi/tri5a_dnn
  for i in {0..6} ; do
    printf -v x 'test%02g' $((i+1))
    echo ${nodes[$i]} $x
    ( ssh ${nodes[$i]} "cd $cwd; steps/aurora4/nnet_decode.sh --nj 5 --srcdir $dir $dir/graph_bg feat/fbank/${x} $dir/decode/decode_bg_${x}" ) &

    printf -v x 'test%02g' $((i+8))
    echo ${nodes[$i]} $x
    ( ssh ${nodes[$i]} "cd $cwd; steps/aurora4/nnet_decode.sh --nj 5 --srcdir $dir $dir/graph_bg feat/fbank/${x} $dir/decode/decode_bg_${x}" ) &
  done
  wait;
  # write out the average WER results
  local/average_wer.sh --min-lmwt 9 --max-lmwt 20 "$dir/decode/decode_bg_test*" | tee $dir/decode/decode_bg_test.avgwer
  log_end "tri5a [decode]"
}
decode_multi_tri5a

decode_multi_tri3b(){
  log_start "tri3b [decode]"
  dir=exp_multi/tri3b_dnn
  for i in {0..6} ; do
    printf -v x 'test%02g' $((i+1))
    echo ${nodes[$i]} $x
    ( ssh ${nodes[$i]} "cd $cwd; steps/aurora4/nnet_decode.sh --nj 5 --srcdir $dir $dir/graph_bg feat/fbank/${x} $dir/decode/decode_bg_${x}" ) &

    printf -v x 'test%02g' $((i+8))
    echo ${nodes[$i]} $x
    ( ssh ${nodes[$i]} "cd $cwd; steps/aurora4/nnet_decode.sh --nj 5 --srcdir $dir $dir/graph_bg feat/fbank/${x} $dir/decode/decode_bg_${x}" ) &
  done
  wait;
  # write out the average WER results
  local/average_wer.sh --min-lmwt 9 --max-lmwt 20 "$dir/decode/decode_bg_test*" | tee $dir/decode/decode_bg_test.avgwer
  log_end "tri3b [decode]"
}
#decode_multi_tri3b

decode_multi_tri4b(){
  log_start "tri4b [decode]"
  dir=exp_multi/tri4b_dnn
  for i in {0..6} ; do
    printf -v x 'test%02g' $((i+1))
    echo ${nodes[$i]} $x
    ( ssh ${nodes[$i]} "cd $cwd; steps/aurora4/nnet_decode.sh --nj 5 --srcdir $dir $dir/graph_bg feat/fbank/${x} $dir/decode/decode_bg_${x}" ) &

    printf -v x 'test%02g' $((i+8))
    echo ${nodes[$i]} $x
    ( ssh ${nodes[$i]} "cd $cwd; steps/aurora4/nnet_decode.sh --nj 5 --srcdir $dir $dir/graph_bg feat/fbank/${x} $dir/decode/decode_bg_${x}" ) &
  done
  wait;
  # write out the average WER results
  local/average_wer.sh --min-lmwt 9 --max-lmwt 20 "$dir/decode/decode_bg_test*" | tee $dir/decode/decode_bg_test.avgwer
  log_end "tri4b [decode]"
}
#decode_multi_tri4b

decode_multi_tri5b(){
  log_start "tri5b [decode]"
  dir=exp_multi/tri5b_dnn
  for i in {0..6} ; do
    printf -v x 'test%02g' $((i+1))
    echo ${nodes[$i]} $x
    ( ssh ${nodes[$i]} "cd $cwd; steps/aurora4/nnet_decode.sh --nj 5 --srcdir $dir $dir/graph_bg feat/fbank/${x} $dir/decode/decode_bg_${x}" ) &

    printf -v x 'test%02g' $((i+8))
    echo ${nodes[$i]} $x
    ( ssh ${nodes[$i]} "cd $cwd; steps/aurora4/nnet_decode.sh --nj 5 --srcdir $dir $dir/graph_bg feat/fbank/${x} $dir/decode/decode_bg_${x}" ) &
  done
  wait;
  # write out the average WER results
  local/average_wer.sh --min-lmwt 9 --max-lmwt 20 "$dir/decode/decode_bg_test*" | tee $dir/decode/decode_bg_test.avgwer
  log_end "tri5b [decode]"
}
#decode_multi_tri5b

decode_multi_tri3c(){
  log_start "tri3c [decode]"
  dir=exp_multi/tri3c_dnn
  for i in {0..6} ; do
    printf -v x 'test%02g' $((i+1))
    echo ${nodes[$i]} $x
    ( ssh ${nodes[$i]} "cd $cwd; steps/aurora4/nnet_decode.sh --nj 5 --srcdir $dir $dir/graph_bg feat/fbank/${x} $dir/decode/decode_bg_${x}" ) &

    printf -v x 'test%02g' $((i+8))
    echo ${nodes[$i]} $x
    ( ssh ${nodes[$i]} "cd $cwd; steps/aurora4/nnet_decode.sh --nj 5 --srcdir $dir $dir/graph_bg feat/fbank/${x} $dir/decode/decode_bg_${x}" ) &
  done
  wait;
  # write out the average WER results
  local/average_wer.sh --min-lmwt 9 --max-lmwt 20 "$dir/decode/decode_bg_test*" | tee $dir/decode/decode_bg_test.avgwer
  log_end "tri3c [decode]"
}
#decode_multi_tri3c

decode_multi_tri4c(){
  log_start "tri4c [decode]"
  dir=exp_multi/tri4c_dnn
  for i in {0..6} ; do
    printf -v x 'test%02g' $((i+1))
    echo ${nodes[$i]} $x
    ( ssh ${nodes[$i]} "cd $cwd; steps/aurora4/nnet_decode.sh --nj 5 --srcdir $dir $dir/graph_bg feat/fbank/${x} $dir/decode/decode_bg_${x}" ) &

    printf -v x 'test%02g' $((i+8))
    echo ${nodes[$i]} $x
    ( ssh ${nodes[$i]} "cd $cwd; steps/aurora4/nnet_decode.sh --nj 5 --srcdir $dir $dir/graph_bg feat/fbank/${x} $dir/decode/decode_bg_${x}" ) &
  done
  wait;
  # write out the average WER results
  local/average_wer.sh --min-lmwt 9 --max-lmwt 20 "$dir/decode/decode_bg_test*" | tee $dir/decode/decode_bg_test.avgwer
  log_end "tri4c [decode]"
}
#decode_multi_tri4c

decode_multi_tri3d(){
  log_start "tri3d [decode]"
  dir=exp_multi/tri3d_dnn
  for i in {0..6} ; do
    printf -v x 'test%02g' $((i+1))
    echo ${nodes[$i]} $x
    ( ssh ${nodes[$i]} "cd $cwd; steps/aurora4/nnet_decode.sh --nj 5 --srcdir $dir $dir/graph_bg feat/fbank/${x} $dir/decode/decode_bg_${x}" ) &

    printf -v x 'test%02g' $((i+8))
    echo ${nodes[$i]} $x
    ( ssh ${nodes[$i]} "cd $cwd; steps/aurora4/nnet_decode.sh --nj 5 --srcdir $dir $dir/graph_bg feat/fbank/${x} $dir/decode/decode_bg_${x}" ) &
  done
  wait;
  # write out the average WER results
  local/average_wer.sh --min-lmwt 9 --max-lmwt 20 "$dir/decode/decode_bg_test*" | tee $dir/decode/decode_bg_test.avgwer
  log_end "tri3d [decode]"
}
#decode_multi_tri3d

decode_multi_tri4d(){
  log_start "tri4d [decode]"
  dir=exp_multi/tri4d_dnn
  for i in {0..6} ; do
    printf -v x 'test%02g' $((i+1))
    echo ${nodes[$i]} $x
    ( ssh ${nodes[$i]} "cd $cwd; steps/aurora4/nnet_decode.sh --nj 5 --srcdir $dir $dir/graph_bg feat/fbank/${x} $dir/decode/decode_bg_${x}" ) &

    printf -v x 'test%02g' $((i+8))
    echo ${nodes[$i]} $x
    ( ssh ${nodes[$i]} "cd $cwd; steps/aurora4/nnet_decode.sh --nj 5 --srcdir $dir $dir/graph_bg feat/fbank/${x} $dir/decode/decode_bg_${x}" ) &
  done
  wait;
  # write out the average WER results
  local/average_wer.sh --min-lmwt 9 --max-lmwt 20 "$dir/decode/decode_bg_test*" | tee $dir/decode/decode_bg_test.avgwer
  log_end "tri4d [decode]"
}
#decode_multi_tri4d
