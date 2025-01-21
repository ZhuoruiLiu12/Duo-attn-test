export CUDA_VISIBLE_DEVICES=1
export BUILD_DOC=1

length_list="512"
mode_list="duo"
for mode in $mode_list; do
    for gen_length in $length_list; do
        python test.py \
            --model-path /disk12/liuzhuorui/models/PLMs/Llama-3-8B-Instruct-Gradient-1048k \
            --attn-pattern-path /disk12/liuzhuorui/works/duo-attention/attn_patterns/Llama-3-8B-Instruct-Gradient-1048k/lr=0.02-reg=0.05-ctx=1000_32000-multi_passkey10/ \
            --sparsity 0.5 \
            --mode $mode \
            --gen_length $gen_length \
            --save_path /disk12/liuzhuorui/works/duo-attention/speed_res_refined_sparsity0.5/
    done
done


