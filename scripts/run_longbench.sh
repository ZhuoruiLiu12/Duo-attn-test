attn_pattern_name="lr=0.02-reg=0.05-ctx=1000_32000-multi_passkey10"
models="Llama-3-8B-Instruct-Gradient-1048k"

# sparsities="0 0.5 0.75"
sparsities="0.5"
# initializations="initial loss_only loss_with_gain loss_with_weighted_gain minimize_number_of_changed_heads"
export CUDA_VISIBLE_DEVICES=6
initializations="loss_with_weighted_gain"
weight="0.9"

tasks="samsum narrativeqa qasper triviaqa hotpotqa multifieldqa_en multifieldqa_zh 2wikimqa musique dureader gov_report qmsum multi_news vcsum trec lsht passage_count passage_retrieval_en passage_retrieval_zh lcc repobench-p"

for model in $models; do
    for initialization in $initializations; do
        for task in $tasks; do
            for sparsity in $sparsities; do
                # bash scripts/longbench.sh $model $task "combination_res_refined_with_weight/attn_patterns_${initialization}_weight${weight}_0.5_32_16.json" $sparsity
                bash scripts/longbench.sh $model $task "/private_space/liuzhuorui/work/DuoAttention/layer_attn_patterns/Llama-3-8B-Instruct-Gradient-1048k/loss_with_weighted0.1_gain/lr=0.02-reg=0.05-ctx=1000_32000-multi_passkey10" $sparsity
            done
        done
    done
done

cd eval/LongBench
for model in $models; do
    python -u eval.py --model $model &
done
