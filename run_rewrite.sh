#!/bin/bash
# remember to set this
exp_dir=$EXP_DIR
#
epsilons=(1000 750 500 400 300 200 100 50)
datasets=("atis" "imdb")

output_dir="${exp_dir}output/"
asset_dir="${exp_dir}asset/"
pruned_model_dir="${output_dir}dp_bart_pr_plus_pruning/"
pruning_index_path="${pruned_model_dir}k_prune_neurons_0_5.pt"
last_checkpoint_path="${pruned_model_dir}checkpoints/checkpoint_6.pt"

for epsilon in ${epsilons[@]}
do
    for dataset in ${datasets[@]}
    do
        python main.py \
            --mode rewrite \
            --model dp_bart \
            --dataset $dataset \
            --name rewrite_dp-bart-pr-plus_epsilon$epsilon \
            --pruning True \
            --gradually_increase_pruning False \
            --private True \
            --epsilon $epsilon \
            --delta 1e-05 \
            --max_seq_len 20 \
            --dp_mechanism gaussian \
            --dp_module clip_value \
            --batch_size 32 \
            --clipping_constant 5 \
            --transformer_type facebook/bart-base \
            --seed 42 \
            --output_dir $output_dir \
            --asset_dir $asset_dir \
            --pruning_index_path $pruning_index_path \
            --last_checkpoint_path $last_checkpoint_path
    done
done