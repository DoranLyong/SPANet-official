DATA_PATH=/path/to/imagenet
MODEL_NAME=spanet_medium
CKPT_PATH=./ckpt/spanet-medium.pth

python validate.py $DATA_PATH --model $MODEL_NAME -b 128 --checkpoint $CKPT_PATH
