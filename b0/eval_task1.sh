#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/OWOBJ-main
/root/autodl-tmp/experiments/B0_PROB_OWOBJ/env/bin/python -u main_open_world.py \
    --data_root /root/autodl-tmp/datasets/OWOD_B0 --dataset TOWOD \
    --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 20 \
    --train_set owod_t1_train --test_set owod_all_task_test \
    --model_type sketch --obj_loss_coef 8e-4 --obj_temp 1.3 --obj_kl_div 0.1 \
    --batch_size 5 --num_workers 4 --wandb_project '' --eval \
    --pretrain /root/autodl-tmp/experiments/B0_PROB_OWOBJ/checkpoints/task1/checkpoint0040.pth \
    --output_dir /root/autodl-tmp/experiments/B0_PROB_OWOBJ/evaluations/final
