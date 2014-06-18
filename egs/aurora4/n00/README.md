#My Aurora4 Setup.

##exp_clean

Models trained using clean training data.

### tri1a

1. 39D MFCC_0_D_A feature 
2. Per-utterance CMVN
3. Tied triphone states

```
Avg.WER [lmwt=19] 34.6014 %
```

### tri1b

Single pass retrained from tri1a using features without CMVN.

```
Avg.WER [lmwt=15] 26.9182
```


##exp_multi

Models trained using multi-style training data.

### tri1a

1. 39D MFCC_0_D_A feature
2. Per-utterance CMVN
3. Tied triphone states

```
Avg.WER [lmwt=17] 22.2465 %
```

### tri1b

Single pass retrained from tri1a using features without CMVN.

```
Avg.WER [lmwt=18] 54.1712 %
```

### tri2a_dnn

1. 26D FBank_D_A feature
2. Per-utterance CMVN
3. Frame labels obtained by aligning the multi-style data

```
Avg.WER [lmwt=13] 14.4472 %
```

### tri2b_dnn

1. 26D FBank_D_A feature
2. Per-utterance CMVN
3. Frame labels obtained by aligning the **clean** data

```
Avg.WER [lmwt=15] 13.8761 %
```

