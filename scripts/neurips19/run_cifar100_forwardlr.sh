expname=$1
SAMPLING_MIN=$2
LR=$3


set -x

ulimit -n 2048
ulimit -a

EXP_PREFIX=$expname
NET="wideresnet"
BATCH_SIZE=64
DECAY=0.0005
SEED=1337
NUM_TRIALS=1
PROB_STRATEGY="relative-cubed"
LOSS="cross"

EXP_NAME=$EXP_PREFIX"_"$PROB_STRATEGY"_"$LOSS

mkdir "/proj/BigLearning/ahjiang/output/cifar100/"
OUTPUT_DIR="/proj/BigLearning/ahjiang/output/cifar100/"$EXP_NAME
mkdir $OUTPUT_DIR

for i in `seq 1 $NUM_TRIALS`
do

  OUTPUT_FILE="sampling_cifar100_"$NET"_"$SAMPLING_MIN"_"$BATCH_SIZE"_0.0_"$DECAY"_trial"$i"_seed"$SEED"_v2"

  echo $OUTPUT_DIR/$OUTPUT_FILE

  # WARNING: Many of these options are hardcoded in lib/SelectiveBackpropper.py
  time python -u train.py \
     --seed=$SEED \
     --batch_size=$BATCH_SIZE \
     --dataset=cifar100 \
     --model=$NET \
     --data_augmentation \
     --cutout \
     --length=8 \
     --epochs=650 \
     --output_dir=$OUTPUT_DIR \
     --sb \
     --lr_sched=$LR \
     --forwardlr \
     --sampling_min=$SAMPLING_MIN &> $OUTPUT_DIR/$OUTPUT_FILE

  let "SEED=SEED+1"

done
