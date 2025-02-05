loss_func="loss_with_weighted_gain"
weights="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"
for loss_fc in $loss_func; do
    for weight in $weights; do
        python search_optimal_combination.py --model_attn_path /private_space/liuzhuorui/work/DuoAttention/attn_patterns/Llama-3-8B-Instruct-Gradient-1048k/lr=0.02-reg=0.05-ctx=1000_32000-multi_passkey10 \
                                                --sparsity 0.5 \
                                                --total_layers 32 \
                                                --save_path /private_space/liuzhuorui/work/DuoAttention/combination_res_refined_with_weight \
                                                --loss_type $loss_fc \
                                                --loss_weight $weight 
    done
done