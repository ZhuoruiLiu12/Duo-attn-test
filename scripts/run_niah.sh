# model="Llama-3-8B-Instruct-Gradient-1048k"
model="/workspace/Edge_Shared/zhuorui/models/Llama-3-8B-Instruct-Gradient-1048k"
model_provider=LLaMA
context_lengths_min=80000
pretrained_len=1048000
sparsity=0.5
attn_pattern="attn_patterns/Llama-3-8B-Instruct-Gradient-1048k/lr=0.02-reg=0.05-ctx=1000_32000-multi_passkey10"
weights="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"
# initializations="initial loss_only loss_with_gain loss_with_weighted_gain minimize_number_of_changed_heads"
initializations="loss_with_weighted_minimize_number_of_changed_heads loss_with_weighted_gain"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
attn_pattern="layer_attn_patterns/Llama-3-8B-Instruct-Gradient-1048k/loss_with_weighted0.1_gain/lr=0.02-reg=0.05-ctx=1000_32000-multi_passkey10"
bash scripts/niah.sh $attn_pattern $sparsity $model $context_lengths_min 0 $pretrained_len $model_provider
# for init in $initializations; do
#     for weight in $weights; do
#         attn_pattern="combination_res_refined_with_weight/attn_patterns_${init}_weight${weight}_0.5_32_16.json"
#         bash scripts/niah.sh $attn_pattern $sparsity $model $context_lengths_min 0 $pretrained_len $model_provider
#     done
# done


# model="Llama-2-7B-32K-Instruct"
# model_provider=LLaMA
# context_lengths_min=2000
# pretrained_len=32000
# sparsity=0.75
# attn_pattern="attn_patterns/Llama-2-7B-32K-Instruct/lr=0.02-reg=0.05-ctx=1000_32000-multi_passkey10"

# CUDA_VISIBLE_DEVICES=0 bash scripts/niah.sh $attn_pattern $sparsity $model $context_lengths_min 0 $pretrained_len $model_provider
