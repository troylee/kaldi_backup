-----

### Date sets

#### Training Data
* train\_si84\_clean: for clean training
* train\_si84\_multi: for multi-style training

#### Development Data
* dev\_0330\_{01...14}
* dev\_1206\_{01...14}

We keep both the two sets just to be consistent with the Kaldi's original setup.

#### Test Data
* test\_eval92\_{01...14}

We only use the standard test sets (each has 330 utterances).

-----

### Language Model

Only the 5K close vocabulary LM is used. Three of them are involved.

* Bigram
* Trigram
* Pruned Trigram

-----

### Experimental Results

#### Pruned Trigram LM

##### Triphone GMM-HMM (exp/tri2b_multi/decode_tgpr_eval92)

```
compute-wer --text --mode=present ark:exp/tri2b_multi/decode_tgpr_eval92/scoring/test_filt.txt ark,p:- 
%WER 19.72 [ 14778 / 74942, 1282 ins, 3531 del, 9965 sub ]
%SER 70.24 [ 3245 / 4620 ]
Scored 4620 sentences, 0 not present in hyp.
```

##### DNN-HMM (exp/tri3a_dnn/decode_tgpr_eval92)

GMM-HMM aligned frame labels.

```
compute-wer --text --mode=present ark:exp/tri3a_dnn/decode_tgpr_eval92/scoring/test_filt.txt ark,p:- 
%WER 13.91 [ 10422 / 74942, 884 ins, 2142 del, 7396 sub ]
%SER 58.03 [ 2681 / 4620 ]
Scored 4620 sentences, 0 not present in hyp.
```

#### DNN-HMM (exp/tri4a_dnn/decode_tgpr_eval92)

DNN-HMM aligned frame labels.

```
compute-wer --text --mode=present ark:exp/tri4a_dnn/decode_tgpr_eval92/scoring/test_filt.txt ark,p:- 
%WER 13.20 [ 9890 / 74942, 850 ins, 2077 del, 6963 sub ]
%SER 57.34 [ 2649 / 4620 ]
Scored 4620 sentences, 0 not present in hyp.
```





