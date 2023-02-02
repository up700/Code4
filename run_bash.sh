#!/bin/bash
GPUID=$1
echo "Run on GPU $GPUID"
TRAIN=$4
TEST=$5
# data
DATASET_SRC=$2
DATASET=$3
PROJECT_ROOT=$(dirname "$(readlink -f "$0")")
DATA_ROOT=$PROJECT_ROOT/dataset/

# model
TOKENIZER_TYPE=bert
SPAN_TYPE=bert
TYPE_TYPE=bert
TOKENIZER_NAME=bert-base-multilingual-cased

# params
LR=1e-5
WEIGHT_DECAY=1e-4
SEED=0

ADAM_EPS=1e-8
ADAM_BETA1=0.9
ADAM_BETA2=0.98
WARMUP=500

############ Progressive Contrastive Bridging ############

# model
MODEL_NAME=bert-base-multilingual-cased

EPOCH=3

TRAIN_BATCH=32
EVAL_BATCH=32

# output
OUTPUT=$PROJECT_ROOT/ptms_pro_contras_bri/$DATASET/

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPUID python3 -u pro_contras_bri.py --data_dir $DATA_ROOT \
  --model_name_or_path $MODEL_NAME \
  --output_dir $OUTPUT \
  --tokenizer_name_or_path $TOKENIZER_NAME \
  --cache_dir $PROJECT_ROOT/cached_models \
  --learning_rate $LR \
  --num_train_epochs $EPOCH \
  --warmup_steps $WARMUP \
  --per_gpu_train_batch_size $TRAIN_BATCH \
  --per_gpu_eval_batch_size $EVAL_BATCH \
  --do_train $TRAIN\
  --do_test $TEST \
  --evaluate_during_training \
  --overwrite_output_dir \
  --model_type $TOKENIZER_TYPE \
  --dataset $DATASET \
  --overwrite_cache \

######### Span Extraction ##########

SPAN_MODEL_NAME=ptms_pro_contras_bri/$DATASET/checkpoint-best-bpt

# params
EPOCH=50

TRAIN_BATCH=16
EVAL_BATCH=32

# output
OUTPUT=$PROJECT_ROOT/ptms_span/$DATASET/

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPUID python3 -u span_extraction_transfer.py --data_dir $DATA_ROOT \
  --span_model_name_or_path $SPAN_MODEL_NAME \
  --output_dir $OUTPUT \
  --tokenizer_name_or_path $TOKENIZER_NAME \
  --cache_dir $PROJECT_ROOT/cached_models \
  --learning_rate $LR \
  --num_train_epochs $EPOCH \
  --warmup_steps $WARMUP \
  --per_gpu_train_batch_size $TRAIN_BATCH \
  --per_gpu_eval_batch_size $EVAL_BATCH \
  --do_train $TRAIN\
  --do_test $TEST \
  --evaluate_during_training \
  --overwrite_output_dir \
  --model_type $TOKENIZER_TYPE \
  --dataset $DATASET \
  --src_dataset $DATASET_SRC \
  --overwrite_cache \

######### Type Prediction ##########
TYPE_MODEL_NAME=ptms_pro_contras_bri/$DATASET/checkpoint-best-bpt

# output
OUTPUT=$PROJECT_ROOT/ptms_type/$DATASET/

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPUID python3 -u type_prediction_transfer.py --data_dir $DATA_ROOT \
  --type_model_name_or_path $TYPE_MODEL_NAME \
  --output_dir $OUTPUT \
  --tokenizer_name_or_path $TOKENIZER_NAME \
  --cache_dir $PROJECT_ROOT/cached_models \
  --learning_rate $LR \
  --num_train_epochs $EPOCH \
  --warmup_steps $WARMUP \
  --per_gpu_train_batch_size $TRAIN_BATCH \
  --per_gpu_eval_batch_size $EVAL_BATCH \
  --do_train $TRAIN\
  --do_test $TEST \
  --evaluate_during_training \
  --overwrite_output_dir \
  --model_type $TOKENIZER_TYPE \
  --dataset $DATASET \
  --src_dataset $DATASET_SRC \
  --overwrite_cache \

######### Subtask Combination ##########

# TOKENIZER_NAME=bert-base-multilingual-cased
SPAN_MODEL_NAME=bert-base-multilingual-cased
# SPAN_MODEL_NAME=ptms_pro_contras_bri/$DATASET/checkpoint-best-bpt-sl
TYPE_MODEL_NAME=bert-base-multilingual-cased
# TYPE_MODEL_NAME=ptms_pro_contras_bri/$DATASET/checkpoint-best-bpt-sl

# output
OUTPUT=$PROJECT_ROOT/ptms/$DATASET/

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPUID python3 -u subtask_combination.py --data_dir $DATA_ROOT \
  --span_model_name_or_path $SPAN_MODEL_NAME \
  --type_model_name_or_path $TYPE_MODEL_NAME \
  --output_dir $OUTPUT \
  --tokenizer_name_or_path $TOKENIZER_NAME \
  --cache_dir $PROJECT_ROOT/cached_models \
  --learning_rate $LR \
  --num_train_epochs $EPOCH \
  --warmup_steps $WARMUP \
  --per_gpu_train_batch_size $TRAIN_BATCH \
  --per_gpu_eval_batch_size $EVAL_BATCH \
  --do_train $TRAIN\
  --do_test $TEST \
  --evaluate_during_training \
  --overwrite_output_dir \
  --model_type $TOKENIZER_TYPE \
  --dataset $DATASET \
  --src_dataset $DATASET_SRC \
  --overwrite_cache \
