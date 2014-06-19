#!/bin/bash

# Copyright 2014 Bo Li
# Apache 2.0

# Begin configuration.
config=            # config, which is also sent to all other scripts

# NETWORK INITIALIZATION
mlp_init=          # select initialized MLP (override initialization)
mlp_proto=         # select network prototype (initialize it)
proto_opts=        # non-default options for 'make_nnet_proto.py'
feature_transform= # provide feature transform (=splice,rescaling,...) (don't build new one)
#
hid_layers=4       # nr. of hidden layers (prior to sotfmax or bottleneck)
hid_dim=1024       # select hidden dimension
bn_dim=            # set a value to get a bottleneck network
dbn=               # select DBN to prepend to the MLP initialization
#
init_opts=         # options, passed to the initialization script

# FEATURE PROCESSING
# feature config (applies always)
norm_vars=true # use variance normalization?
# feature_transform:
splice=5         # temporal splicing
feat_type=plain

# LABELS
labels=            # use these labels to train (override deafault pdf alignments) 
num_tgt=           # force to use number of outputs in the MLP (default is autodetect)

# TRAINING SCHEDULER
learn_rate=0.008   # initial learning rate
train_opts=        # options, passed to the training script
train_tool=        # optionally change the training tool
frame_weights=     # per-frame weights for gradient weighting

# OTHER
use_gpu_id= # manually select GPU id to run on, (-1 disables GPU)
seed=777    # seed value used for training data shuffling and initialization
# End configuration.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh; 


. parse_options.sh || exit 1;


if [ $# != 6 ]; then
   echo "Usage: $0 <data-train> <data-dev> <lang-dir> <ali-train> <ali-dev> <exp-dir>"
   echo " e.g.: $0 data/train data/cv data/lang exp/mono_ali exp/mono_ali_cv exp/mono_nnet"
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>  # config containing options"
   exit 1;
fi

data=$1
data_cv=$2
lang=$3
alidir=$4
alidir_cv=$5
dir=$6

silphonelist=`cat $lang/phones/silence.csl` || exit 1;


for f in $alidir/final.mdl $alidir/ali.1.gz $alidir_cv/ali.1.gz $data/feats.scp $data_cv/feats.scp; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

echo
echo "# INFO"
echo "$0 : Training Neural Network"
printf "\t dir       : $dir \n"
printf "\t Train-set : $data $alidir \n"
printf "\t CV-set    : $data_cv $alidir_cv \n"

mkdir -p $dir/{log,nnet}

# skip when already trained
[ -e $dir/final.nnet ] && printf "\nSKIPPING TRAINING... ($0)\nnnet already trained : $dir/final.nnet ($(readlink $dir/final.nnet))\n\n" && exit 0

###### PREPARE ALIGNMENTS ######
echo
echo "# PREPARING ALIGNMENTS"
if [ ! -z $labels ]; then
  echo "Using targets '$labels' (by force)"
  labels_tr="$labels"
  labels_cv="$labels"
else
  echo "Using PDF targets from dirs '$alidir' '$alidir_cv'"
  # define pdf-alignment rspecifiers
  labels_tr="ark:ali-to-pdf $alidir/final.mdl \"ark:gunzip -c $alidir/ali.*.gz |\" ark:- | ali-to-post ark:- ark:- |"
  labels_cv="ark:ali-to-pdf $alidir/final.mdl \"ark:gunzip -c $alidir_cv/ali.*.gz |\" ark:- | ali-to-post ark:- ark:- |"
  # 
  labels_tr_pdf="ark:ali-to-pdf $alidir/final.mdl \"ark:gunzip -c $alidir/ali.*.gz |\" ark:- |" # for analyze-counts.
  labels_tr_phn="ark:ali-to-phones --per-frame=true $alidir/final.mdl \"ark:gunzip -c $alidir/ali.*.gz |\" ark:- |"

  # get pdf-counts, used later to post-process DNN posteriors
  analyze-counts --verbose=1 --binary=false "$labels_tr_pdf" $dir/ali_train_pdf.counts 2>$dir/log/analyze_counts_pdf.log || exit 1
  # copy the old transition model, will be needed by decoder
  copy-transition-model --binary=false $alidir/final.mdl $dir/final.mdl || exit 1
  # copy the tree
  cp $alidir/tree $dir/tree || exit 1

  # make phone counts for analysis
  analyze-counts --verbose=1 --symbol-table=$lang/phones.txt "$labels_tr_phn" /dev/null 2>$dir/log/analyze_counts_phones.log || exit 1
fi

###### PREPARE FEATURES ######
echo
echo "# PREPARING FEATURES"
# shuffle the list
echo "Preparing train/cv lists :"
cat $data/feats.scp | utils/shuffle_list.pl --srand ${seed:-777} > $dir/train.scp
cp $data_cv/feats.scp $dir/cv.scp
# print the list sizes
wc -l $dir/train.scp $dir/cv.scp

###### PREPARE FEATURE PIPELINE ######

# prepare the features, add delta
feats_tr="ark:add-deltas --delta-order=2 --delta-window=3 scp:$dir/train.scp ark:- |"
feats_cv="ark:add-deltas --delta-order=2 --delta-window=3 scp:$dir/cv.scp ark:- |"

# CMVN:
cmvn_tr=$data/cmvn_0_d_a.utt.scp
cmvn_cv=$data_cv/cmvn_0_d_a.utt.scp
echo "Will use CMVN statistics : ${cmvn_tr}, ${cmvn_cv}"
[ ! -r ${cmvn_tr} ] && echo "Cannot find cmvn stats $cmvn_tr" && exit 1;
[ ! -r ${cmvn_cv} ] && echo "Cannot find cmvn stats $cmvn_cv" && exit 1;
feats_tr="$feats_tr apply-cmvn --print-args=false --norm-vars=$norm_vars scp:$cmvn_tr ark:- ark:- |"
feats_cv="$feats_cv apply-cmvn --print-args=false --norm-vars=$norm_vars scp:$cmvn_cv ark:- ark:- |"
# keep track of norm_vars option
echo "$norm_vars" >$dir/norm_vars 

# Splicing:
feats_tr="$feats_tr splice-feats --left-context=${splice} --right-context=${splice} ark:- ark:- |"
feats_cv="$feats_cv splice-feats --left-context=${splice} --right-context=${splice} ark:- ark:- |"
echo "${splice}" > $dir/splice

# get feature dim
echo "Getting feature dim : "
feat_dim=$(feat-to-dim --print-args=false "$feats_tr" -)
echo "Feature dim is : $feat_dim"

# Now we will start building complex feature_transform which will 
# be forwarded in CUDA to have fast run-time.
#
# We will use 1GPU for both feature_transform and MLP training in one binary tool. 
# This is against the kaldi spirit to have many independent small processing units, 
# but it is necessary because of compute exclusive mode, where GPU cannot be shared
# by multiple processes.

if [ ! -z "$feature_transform" ]; then
  echo "Using pre-computed feature-transform : '$feature_transform'"
  cp $feature_transform $dir/final.feature_transform
  # transform the features
  feats_tr="$feats_tr transform-feats ${feature_transform} ark:- ark:- |"
  feats_cv="$feats_cv transform-feats ${feature_transform} ark:- ark:- |"
fi

###### INITIALIZE THE NNET ######
echo 
echo "# NN-INITIALIZATION"
[ ! -z "$mlp_init" ] && echo "Using pre-initialized network '$mlp_init'";
if [ ! -z "$mlp_proto" ]; then 
  echo "Initializing using network prototype '$mlp_proto'";
  mlp_init=$dir/nnet.init; log=$dir/log/nnet_initialize.log
  nnet-initialize $mlp_proto $mlp_init 2>$log || { cat $log; exit 1; } 
fi
if [[ -z "$mlp_init" && -z "$mlp_proto" ]]; then
  echo "Getting input/output dims :"
  #initializing the MLP, get the i/o dims...
  #input-dim
  num_fea=$(feat-to-dim "$feats_tr" - )
  { #optioanlly take output dim of DBN
    [ ! -z $dbn ] && num_fea=$(nnet-forward $dbn "$feats_tr" ark:- | feat-to-dim ark:- -)
    [ -z "$num_fea" ] && echo "Getting nnet input dimension failed!!" && exit 1
  }

  #output-dim
  [ -z $num_tgt ] && num_tgt=$(hmm-info --print-args=false $alidir/final.mdl | grep pdfs | awk '{ print $NF }')

  # make network prototype
  mlp_proto=$dir/nnet.proto
  echo "Genrating network prototype $mlp_proto"
  utils/nnet/make_nnet_proto.py $proto_opts \
    ${bn_dim:+ --bottleneck-dim=$bn_dim} \
    $num_fea $num_tgt $hid_layers $hid_dim >$mlp_proto || exit 1
  # initialize
  mlp_init=$dir/nnet.init; log=$dir/log/nnet_initialize.log
  echo "Initializing $mlp_proto -> $mlp_init"
  nnet-initialize $mlp_proto $mlp_init 2>$log || { cat $log; exit 1; }

  #optionally prepend dbn to the initialization
  if [ ! -z $dbn ]; then
    mlp_init_old=$mlp_init; mlp_init=$dir/nnet_$(basename $dbn)_dnn.init
    nnet-concat $dbn $mlp_init_old $mlp_init || exit 1 
  fi
fi


###### TRAIN ######
echo
echo "# RUNNING THE NN-TRAINING SCHEDULER"
steps/nnet_scheduler/train_scheduler_newbob.sh \
  ${feature_transform:+ --feature-transform "$feature_transform"} \
  --learn-rate $learn_rate \
  --randomizer-seed $seed \
  ${train_opts} \
  ${train_tool:+ --train-tool "$train_tool"} \
  ${frame_weights:+ --frame-weights "$frame_weights"} \
  ${config:+ --config $config} \
  $mlp_init "$feats_tr" "$feats_cv" "$labels_tr" "$labels_cv" $dir || exit 1


echo "$0 successfuly finished.. $dir"

sleep 3
exit 0

