loss_func="loss_only loss_with_gain loss_minimize_number_of_changed_heads"
for loss_fc in $loss_func; do
    python search_optimal_combination.py --model_attn_path /disk12/liuzhuorui/works/duo-attention/attn_patterns/Llama-3-8B-Instruct-Gradient-1048k/lr=0.02-reg=0.05-ctx=1000_32000-multi_passkey10 \
                                            --sparsity 0.5 \
                                            --total_layers 32 \
                                            --save_path /disk12/liuzhuorui/works/duo-attention/optimal_combination_res_refined \
                                            --loss_type $loss_fc
done