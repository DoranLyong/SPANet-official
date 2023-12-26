DATA_PATH=/path/to/imagenet
CODE_PATH=/path/to/code/spanet # modify code path here


ALL_BATCH_SIZE=1024
NUM_GPU=4
GRAD_ACCUM_STEPS=4 # Adjust according to your GPU numbers and memory size.
let BATCH_SIZE=ALL_BATCH_SIZE/NUM_GPU/GRAD_ACCUM_STEPS

MODEL=spanet_mediumX # spanet_{small, medium, mediumX, base, baseX}
DROP_PATH=0.3 # drop path rates [0.1, 0.2, 0.3, 0.3, 0.4] responding to model [small, medium, mediumX, base, baseX]
LR=1e-3 

cd $CODE_PATH && sh distributed_train.sh $NUM_GPU $DATA_PATH \
--model $MODEL --opt adamw --lr $LR --warmup-epochs 5 \
-b $BATCH_SIZE --grad-accum-steps $GRAD_ACCUM_STEPS \
--drop-path $DROP_PATH --head-dropout 0.0 \
