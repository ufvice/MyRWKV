#!/bin/bash
#######################################################################################################################
#
# Run demo-training-prepare.sh with the same MODEL_TYPE & N_LAYER & N_EMBD first
# Or, rename your base model to rwkv-init.pth and put it in the output folder
#
# The trainer will load the last rwkv-*.pth in the folder, such that it can continue from a stopped run
# Therefore check the log (### Loading rwkv-xxx.pth... ###), and make sure you don't have extra rwkv-*.pth there
#
#######################################################################################################################
MODEL_TYPE="x070" # x070 => rwkv-7.0
N_LAYER="24"
N_EMBD="2048"
CTX_LEN="32768" # !!! change magic_prime if you change ctx_len !!!
TOKEN_NUM=9951649622
MAGIC_PRIME=303689
DATA_PATH="data/novel"
WANDB_PROJ="rwkv7-1.5B-novel-ctx32k"
PROJ_DIR="out/L"$N_LAYER"-D"$N_EMBD"-"$MODEL_TYPE # set output folder
M_BSZ="4" # takes ~9G VRAM here => reduce this to save VRAM, increase this for faster speed
LR_INIT="8e-4"
LR_FINAL="1e-4"
GRAD_CP=0 # 1 => slower, save VRAM; 0 => faster, more VRAM
EPOCH_SAVE=5 # save every 10 "miniepochs" (1 miniepoch = 40320 * ctx_len tokens) => decrease if your GPU is weak
N_NODE=1 # number of nodes
GPU_PER_NODE=1 # number of GPUs per node
DS_BUCKET_MB=2 # set to 2 for consumer GPUs, set to 200 for A100 / H100 (affects speed & vram usage)
#
python train.py --load_model "0" --wandb $WANDB_PROJ --proj_dir $PROJ_DIR --my_testing $MODEL_TYPE \
 --ctx_len $CTX_LEN --my_pile_stage 3 --load_partial 1 --epoch_count 999999 --epoch_begin 0 \
 --data_file $DATA_PATH --my_exit_tokens $TOKEN_NUM --magic_prime $MAGIC_PRIME \
 --num_nodes $N_NODE --micro_bsz $M_BSZ --n_layer $N_LAYER --n_embd $N_EMBD --pre_ffn 0 --head_qk 0 \
 --lr_init $LR_INIT --lr_final $LR_FINAL --warmup_steps 10 --beta1 0.9 --beta2 0.99 --adam_eps 1e-18 --my_pile_edecay 0 --data_type "binidx" --vocab_size 65536 \
 --weight_decay 0.001 --epoch_save $EPOCH_SAVE --head_size_a 64 \
 --accelerator gpu --devices $GPU_PER_NODE --precision bf16 --strategy deepspeed_stage_2 --grad_cp $GRAD_CP --enable_progress_bar True --ds_bucket_mb $DS_BUCKET_MB
