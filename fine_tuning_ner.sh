#!/bin/bash

BIOBERT_MODEL=dmis-lab/biobert-v1.1
echo "setting biober pre-trained model:$BIOBERT_DIR"
MAX_LENGTH=128
BATCH_SIZE=32
NUM_EPOCHS=3
SAVE_STEPS=750
SEED=1 
ner_corpus=(/nfs/storages/bio_corpus/ner/*)
for u in "${ner_corpus[@]}"
do
    echo "$u a bio corpus used by the NER task"
    OUTPUT_DIR=$u/ner_outputs
    NER_DIR=$u
    echo "fine-tuning model saved to $OUTPUT_DIR"
    rm -rf $OUTPUT_DIR
    mkdir -p $OUTPUT_DIR
    start_time=`date +%s`
    python3 run_ner.py --data_dir $NER_DIR/ \
	--labels $NER_DIR/labels.txt \
	--model_name_or_path $BIOBERT_MODEL \
	--output_dir $OUTPUT_DIR \
	--max_seq_length  $MAX_LENGTH \
	--num_train_epochs $NUM_EPOCHS \
	--per_device_train_batch_size $BATCH_SIZE \
	--save_steps $SAVE_STEPS \
	--seed $SEED \
	--do_train \
	--do_eval \
	--do_predict
    echo "execution ($NER_DIR) time: $((($(date +%s)-$start_time))) seconds."
done
