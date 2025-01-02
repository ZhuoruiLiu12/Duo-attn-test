# Duo-attn-test

## Start
### Environment Setup
[ref original repo](https://github.com/mit-han-lab/duo-attention/tree/main?tab=readme-ov-file#installation-and-usage)

### Run Test
```
bash run_prefilling_decoding_speed_test.sh {mode} {save_path}
```
Change the model path and attn pattern path in this shell script to your own path.   
Only support sparsity is 0.5 now.
### Experiment Results
- Convert the result to markdown view
```
python statistics_res.py --data_path {directory_to_res} \
                         --save_path {path_to_save_markdown} 
```