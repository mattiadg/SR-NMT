#!/bin/bash

hyp=$1
tgt=$2
tok_gold_targets=$3

mosesdecoder=/hltsrv1/software/moses/moses-20150228_kenlm_cmph_xmlrpc_irstlm_master/

sed -e "s/@@ //g" < $hyp | $mosesdecoder/scripts/tokenizer/detokenizer.perl $tgt  | $mosesdecoder/scripts/recaser/detruecase.perl > $hyp.tmp
# Tokenize.
perl $mosesdecoder/scripts/tokenizer/tokenizer.perl -l $tgt < $hyp.tmp > $hyp.tok

# Put compounds in ATAT format (comparable to papers like GNMT, ConvS2S).
# See https://nlp.stanford.edu/projects/nmt/ :
# 'Also, for historical reasons, we split compound words, e.g.,
#    "rich-text format" --> rich ##AT##-##AT## text format."'
perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < $tok_gold_targets > $tok_gold_targets.atat
perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < $hyp.tok > $hyp.atat

# Get BLEU.
perl $mosesdecoder/scripts/generic/multi-bleu.perl $tok_gold_targets.atat < $hyp.atat
