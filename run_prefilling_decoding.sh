export CUDA_VISIBLE_DEVICES=1
export BUILD_DOC=1

decoding_length="1024 2048 4096 8192 16384 32768"
prefilling_length="1024 2048 4096 8192 16384 32768"
mode_list="duo layer-wise"
for dl in $decoding_length; do
    for pl in $prefilling_length; do
        for mode in $mode_list; do
            python prefilling_decoding_speed_test_res.py \
                --model-path /disk12/liuzhuorui/models/PLMs/Llama-3-8B-Instruct-Gradient-1048k \
                --attn-pattern-path /disk12/liuzhuorui/works/duo-attention/attn_patterns/Llama-3-8B-Instruct-Gradient-1048k/lr=0.02-reg=0.05-ctx=1000_32000-multi_passkey10/ \
                --sparsity 0.5 \
                --mode $mode \
                --decoding_length $dl \
                --prefilling_length $pl \
                --save_path /disk12/liuzhuorui/works/duo-attention/res/duo_layer-wise_combined_res_sparsity0.5
        done
    done
done