# bash scripts/train.sh Llama-2-7B-32K-Instruct 1000 32000 0.05 0.02 10

# bash scripts/train.sh Llama-3-8B-Instruct-Gradient-1048k 1000 32000 0.05 0.02 10 loss_only /private_space/liuzhuorui/work/DuoAttention/combination_res_refined/attn_patterns_loss_only_0.5_32_16.json

# bash scripts/train.sh Llama-3-8B-Instruct-Gradient-1048k 1000 32000 0.05 0.02 10 loss_with_gain /private_space/liuzhuorui/work/DuoAttention/combination_res_refined/attn_patterns_loss_with_gain_0.5_32_16.json

bash scripts/train.sh Llama-3-8B-Instruct-Gradient-1048k 1000 32000 0.05 0.02 10 loss_with_weighted0.1_gain /private_space/liuzhuorui/work/DuoAttention/combination_res_refined_with_weight/attn_patterns_loss_with_weighted_gain_weight0.1_0.5_32_16.json

# bash scripts/train.sh Llama-3-8B-Instruct-Gradient-1048k 1000 32000 0.05 0.02 10 initial none

# bash scripts/train.sh Llama-3-8B-Instruct-Gradient-4194k 1000 32000 0.05 0.02 10

# bash scripts/train.sh Mistral-7B-Instruct-v0.2 1000 32000 0.05 0.02 10

# bash scripts/train.sh Mistral-7B-Instruct-v0.3 1000 32000 0.05 0.02 10
