#!/bin/bash

echo "processing data for fine-tuning"
ner_corpus=(/nfs/storages/bio_corpus/ner/*)
MAX_LENGTH=128
BERT_MODEL=dmis-lab/biobert-v1.1
for u in "${ner_corpus[@]}"
do
    rm  $u/*.tmp $u/*.txt
    echo "processing data at ($u)****************start"
    cat $u/train_dev.tsv | tr '\t' ' ' > $u/train.txt.tmp
    cat $u/devel.tsv | tr '\t' ' ' > $u/dev.txt.tmp
    cat $u/test.tsv| tr '\t' ' ' > $u/test.txt.tmp

    python3 scripts/preprocess.py $u/train.txt.tmp $BERT_MODEL $MAX_LENGTH > $u/train.txt
    python3 scripts/preprocess.py $u/dev.txt.tmp $BERT_MODEL $MAX_LENGTH > $u/dev.txt
    python3 scripts/preprocess.py $u/test.txt.tmp $BERT_MODEL $MAX_LENGTH > $u/test.txt
    cat $u/train.txt $u/dev.txt $u/test.txt | cut -d " " -f 2 | grep -v "^$"| sort | uniq > $u/labels.txt
    echo "processing data at ($u)****************end"
done
