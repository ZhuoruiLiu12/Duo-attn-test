cd ../eval/needle
weights="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"
types="loss_with_weighted_gain loss_with_weighted_minimize_number_of_changed_heads"

for weight in $weights; do
    for type in $types; do
        python visualize.py \
            --folder_path "results/Llama-3-8B-Instruct-Gradient-1048k_duo_attn-attn_pattern=attn_patterns_${type}_weight${weight}_0.5_32_16.json-sparsity=0.5/" \
            --pretrained_len 1048000
    done
done