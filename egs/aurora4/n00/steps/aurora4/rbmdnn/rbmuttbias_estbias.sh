#!/bin/bash
# Copyright 2014 Bo Li

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

# Begin configuration.
#
rbm_drop_data=0.0     #sample the training set, 1.0 drops all the data, 0.0 keeps all
# pre-training opts
rbm_lrate_low=0.01    #lower RBM learning rate (for Gaussian units)
rbm_l2penalty=0.0002  #L2 penalty (increases RBM-mixing rate)
rbm_extra_opts=
# data processing config
# feature config
srcdir=
num_iter=10

uttbias_use_zbias=false # use zero biases
# misc.
verbose=1 # enable per-cache reports
# End configuration.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh;
. parse_options.sh || exit 1;


if [ $# != 3 ]; then
   echo "Usage: $0 <model> <data> <exp-dir>"
   echo " e.g.: $0 exp/rbm_pretrain/1.rbm data/dev_multi exp/rbm_pretrain/biases/dev_multi"
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>           # config containing options"
   echo ""
   echo "  --srcdir <string>                # training dir"
   echo "  --num-iter <N>                   # number of CD-1 iterations"
   echo "  --rbm-drop-data <float>          # probability of frame-dropping,"
   echo "                                   # can be used to subsample large datasets"
   echo "  --rbm-lrate_low <float>          # learning-rate for Gaussian-Bernoulli RBM"
   echo ""
   exit 1;
fi

model=$1
data=$2
dir=$3
[ -z $srcdir ] && srcdir=`dirname $dir`; # Default model directory one level up from decoding directory.


for f in $data/feats.scp; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

echo "# INFO"
echo "$0 : Estimating utt-biases for RBM"
printf "\t dir       : $dir \n"
printf "\t data      : $data \n"
printf "\t rbm       : $model \n"

[ -e $dir/rbm_visbias.ark ] && [ -e $dir/rbm_hidbias.ark ] && echo "$0 Skipping, already have $dir/rbm_visbias.ark and $dir/rbm_hidbias.ark" && exit 0

mkdir -p $dir/log

###### PREPARE FEATURES ######
echo
echo "# PREPARING FEATURES"
# print the list size
wc -l $data/feats.scp

###### PREPARE FEATURE PIPELINE ######

#prepare features, add delta
feats="ark:add-deltas --delta-order=2 --delta-window=3 scp:$data/feats.scp ark:- |"
# do utt-cmvn
# add CMVN
if [ -f $srcdir/norm_vars ]; then
  norm_vars=$(cat $srcdir/norm_vars 2>/dev/null)
  [ ! -f $data/cmvn_0_d_a.utt.scp ] && echo "$0: cannot find cmvn stats $data/cmvn_0_d_a.utt.scp" && exit 1
  feats="$feats apply-cmvn --norm-vars=$norm_vars scp:$data/cmvn_0_d_a.utt.scp ark:- ark:- |"
fi
# splicing 
if [ -f $srcdir/splice ]; then
  splice=$(cat $srcdir/splice 2>/dev/null)
  feats="$feats splice-feats --left-context=${splice} --right-context=${splice} ark:- ark:- |"
fi

#get feature dim
echo -n "Getting feature dim : "
feat_dim=$(feat-to-dim --print-args=false scp:$dir/train.scp -)
echo $feat_dim

if [ -f $srcdir/final.feature_transform ]; then
  # transform the features
  feats="${feats} transform-feats $srcdir/final.feature_transform ark:- ark:- |"
fi

###### GET THE DIMENSIONS ######
num_fea=$(feat-to-dim --print-args=false "$feats" - 2>/dev/null)

###### PERFORM THE TRAINING ######
echo
echo "# Estimating biases"
visbias=$dir/rbm_visbias.ark
hidbias=$dir/rbm_hidbias.ark

#pre-train
echo "# Estimating biases ($num_iter iters)"
rbm-uttbias-train --learn-rate=$rbm_lrate_low --l2-penalty=$rbm_l2penalty \
  --num-iters=$num_iter --drop-data=$rbm_drop_data --verbose=$verbose \
  --use-zero-init-bias=$uttbias_use_zbias $rbm_extra_opts \
  $model "$feats" ark:$visbias ark:$hidbias 2>$dir/log/rbm_uttbias.log || exit 1
# clean up the intermediate bias estimates
rm $dir/rbm_visbias.ark.[0-9]*
rm $dir/rbm_hidbias.ark.[0-9]*

echo
echo "# REPORT"
echo "# Training progress"
grep progress $dir/log/rbm_uttbias.log
echo 

echo "RBM utt bias estimation finished."

sleep 3
exit 0

