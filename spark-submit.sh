#!/usr/bin/env bash
DATA_DIR=${1:-data}
MODEL_OUT=${2:-models/als_model}
RANK=${3:-20}
REG=${4:-0.1}
MAXITER=${5:-10}

spark-submit \
  --master ${SPARK_MASTER:-local[*]} \
  --conf spark.sql.shuffle.partitions=8 \
  src/train_als.py --data-path ${DATA_DIR} --model-out ${MODEL_OUT} --rank ${RANK} --reg ${REG} --maxIter ${MAXITER}