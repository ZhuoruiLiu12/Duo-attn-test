# export BUILD_DOC=1

mode=$1
prefilling_length="1024 2048 4096 8192 16384 32768"
decoding_length="1024 2048 4096 8192 16384 32768"
save_path=$2

for dl in $decoding_length; do
    for pl in $prefilling_length; do
        python prefilling_decoding_speed_test_res.py \
            --model-path /disk12/liuzhuorui/models/PLMs/Llama-3-8B-Instruct-Gradient-1048k \
            --attn-pattern-path /disk12/liuzhuorui/works/duo-attention/attn_patterns/Llama-3-8B-Instruct-Gradient-1048k/lr=0.02-reg=0.05-ctx=1000_32000-multi_passkey10/ \
            --sparsity 0.5 \
            --mode $mode \
            --decoding_length $dl \
            --prefilling_length $pl \
            --save_path $save_path
    done
done