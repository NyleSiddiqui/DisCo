#!/bin/bash
#SBATCH --gres=gpu:1
##SBATCH --exclude c4-4
#SBATCH -c12
#SBATCH -p gpu
#SBATCH -C gmem80
#SBATCH --mem-per-cpu=14G
#SBATCH --job-name=ntu60-fixpose
#SBATCH --output=%x.out

source activate /home/siddiqui/.conda/envs/diffusion

AZFUSE_USE_FUSE=0 NCCL_ASYNC_ERROR_HANDLING=0 python finetune_sdm_yaml.py --cf config/ref_attn_clip_combine_controlnet/tiktok_S256L16_xformers_tsv.py --do_train --root_dir /home/siddiqui/DisCo/ --local_train_batch_size 256  --local_eval_batch_size 256  --log_dir exp/ntu60-fixpose --epochs 20 --deepspeed --eval_step 1000 --save_step 1000 --gradient_accumulate_steps 1 --learning_rate 2e-4 --fix_dist_seed --loss_target "noise" --train_yaml ../TSV_dataset/composite_offset/train_TiktokDance-poses-masks.yaml --val_yaml ../TSV_dataset/composite_offset/new10val_TiktokDance-poses-masks.yaml --unet_unfreeze_type "all" --refer_sdvae --ref_null_caption False --combine_clip_local --combine_use_mask --conds "poses" "masks" --stage1_pretrain_path mp_rank_00_model_states.pt --num_workers 12 --eval_before_train False

#ython3 dataloader.py
#python3 preprocessCharades.py
#python3 model.py