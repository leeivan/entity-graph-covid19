#!/bin/bash

BIOBERT_MODEL=dmis-lab/biobert-v1.1
echo "setting biober pre-trained model:$BIOBERT_MODEL"
MAX_LENGTH=128
BATCH_SIZE=32
NUM_EPOCHS=3
LEARNING_RATE=2e-5 
re_corpus=(/nfs/storages/bio_corpus/re_all_in_one/*)
for u in "${re_corpus[@]}"
do
    echo "$u is a bio corpus used by the RE task"
    OUTPUT_DIR=$u/re_outputs
    RE_DIR=$u
    echo "fine-tuning model saved to $OUTPUT_DIR"
    rm -rf $OUTPUT_DIR
    mkdir -p $OUTPUT_DIR
    start_time=`date +%s`
    python3 run_glue.py --data_dir $RE_DIR/ \
        --task_name SST-2 \
        --model_name_or_path $BIOBERT_MODEL \
        --output_dir $OUTPUT_DIR \
        --max_seq_length  $MAX_LENGTH \
        --num_train_epochs $NUM_EPOCHS \
        --per_device_train_batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --do_train \
        --do_predict
    echo "execution ($RE_DIR) time: $((($(date +%s)-$start_time))) seconds."
done
