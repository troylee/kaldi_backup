#!/bin/bash
# Author: Bo Li (li-bo@outlook.com)
#
# This script is specific to our servers for decoding, which is also included in the 
# main script. 
#
cwd=~/tools/kaldi/egs/aurora4/n00
cd $cwd

nodes=( compg2 compg3 compg4 compg5 compg6 compg7 compg8 )

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
  ( ssh ${nodes[$i]} "cd $cwd; steps/aurora4_decode_deltas.sh --nj 5 --model exp_clean/tri1a/final.mdl exp_clean/tri1a/graph_bg feat/mfcc/${x} exp_clean/tri1a/decode/decode_bg_${x}" ) &
  
  printf -v x 'test%02g' $((i+8))
  echo ${nodes[$i]} $x
  ( ssh ${nodes[$i]} "cd $cwd; steps/aurora4_decode_deltas.sh --nj 5 --model exp_clean/tri1a/final.mdl exp_clean/tri1a/graph_bg feat/mfcc/${x} exp_clean/tri1a/decode/decode_bg_${x}" ) &
done
wait;
# write out the average WER results
local/average_wer.sh 'exp_clean/tri1a/decode/decode_bg_test*' | tee exp_clean/tri1a/decode/decode_bg_test.avgwer
log_end "tri1a [decode]"

}

# decode exp_multi/tri1a
log_start "tri1a [decode]"
for i in {0..6} ; do 
  printf -v x 'test%02g' $((i+1))
  echo ${nodes[$i]} $x
  ( ssh ${nodes[$i]} "cd $cwd; steps/aurora4_decode_deltas.sh --nj 5 --model exp_multi/tri1a/final.mdl exp_multi/tri1a/graph_bg feat/mfcc/${x} exp_multi/tri1a/decode/decode_bg_${x}" ) &
  
  printf -v x 'test%02g' $((i+8))
  echo ${nodes[$i]} $x
  ( ssh ${nodes[$i]} "cd $cwd; steps/aurora4_decode_deltas.sh --nj 5 --model exp_multi/tri1a/final.mdl exp_multi/tri1a/graph_bg feat/mfcc/${x} exp_multi/tri1a/decode/decode_bg_${x}" ) &
done
wait;
# write out the average WER results
local/average_wer.sh 'exp_multi/tri1a/decode/decode_bg_test*' | tee exp_multi/tri1a/decode/decode_bg_test.avgwer
log_end "tri1a [decode]"

