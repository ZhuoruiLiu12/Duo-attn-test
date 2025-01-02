from email import iterators
from duo_attn.utils import load_attn_pattern, sparsify_attention_heads
from duo_attn.patch import enable_duo_attention_eval
import transformers
import torch
import numpy as np

import argparse
from tqdm import tqdm
import os
import json

from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--attn-pattern-path",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--sparsity",
        required=True,
        type=float
    )
    parser.add_argument(
        "--mode",
        choices=["duo", "layer-wise", "test"],
        required=True,
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=0
    )
    parser.add_argument(
        "--iter",
        type=int,
        default=10
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None
    )

    parser.add_argument("--sink_tokens", type=int, default=None)
    parser.add_argument("--recent_size", type=int, default=None)
    parser.add_argument("--prefilling_length", type=int, default=1024)
    parser.add_argument("--decoding_length", type=int, default=1024)

    args = parser.parse_args()
    return args

def main(args):

    # Load the model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )

    # Load the attention pattern
    attn_heads, sink_size, recent_size = load_attn_pattern(
        args.attn_pattern_path
    )

    # Sparsify attention heads
    attn_heads, sparsity = sparsify_attention_heads(attn_heads, sparsity=args.sparsity)

    # Test the same attention pattern in a layer
    if args.mode == 'layer-wise':
        attn_heads = np.full_like(attn_heads, 0)
        attn_heads[16:, :] = 1
    elif args.mode == "test":
        attn_heads = np.full_like(attn_heads, 0)
        attn_heads[:, 4:] = 1

    # Enable DuoAttention
    enable_duo_attention_eval(
        model,
        attn_heads,
        sink_size=64,
        recent_size=256,
        mode=args.mode,
    )


    # Move model to GPU
    model = model.cuda()

    # Ready for inference!
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path)

    # Test speed, use random tensor to replace the original dataset
    input_data = torch.randint(low=0, high=12799, size=(1, args.prefilling_length)).to(model.device)

    # warmup
    for _ in range(5):
        input_ids = input_data
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        past_key_values = None
        original_length = input_ids.shape[1]
        start_event.record()
        for _ in range(args.decoding_length):
            with torch.no_grad():
                outputs = model(input_ids, use_cache=True, past_key_values=past_key_values)
                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values

                gen_token = torch.argmax(logits, dim=-1).unsqueeze(1)
                # input_ids = torch.cat([input_ids, gen_token], dim=-1)
                input_ids = gen_token
            
            torch.cuda.empty_cache()
        # outputs = model(input_ids)
        # import pdb;pdb.set_trace()
        end_event.record()

        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        print(f"Warmup: {elapsed_time} ms, input length: {original_length}.")
        print(f"Generate sequence length: {input_ids.size(1) - original_length}.")

    # Test prefilling and decoding time.
    runing_times = []
    progress_bar = tqdm(range(args.iter), desc="Progress for total instances.")
    prefilling_start_event = torch.cuda.Event(enable_timing=True)
    prefilling_end_event = torch.cuda.Event(enable_timing=True)
    decoding_start_event = torch.cuda.Event(enable_timing=True)
    decoding_end_event = torch.cuda.Event(enable_timing=True)
    
    for _ in range(args.iter):
        progress_bar.update(1)
        input_ids = torch.randint(low=0, high=12799, size=(1, args.prefilling_length)).to(model.device)
        original_length = input_ids.shape[1]
        # Prefilling stage
        prefilling_start_event.record()
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

            gen_token = torch.argmax(logits, dim=-1).unsqueeze(1)
            # input_ids = torch.cat([input_ids, gen_token], dim=-1)
            input_ids = gen_token
        
        prefilling_end_event.record()
        torch.cuda.synchronize()
        prefilling_time = prefilling_start_event.elapsed_time(prefilling_end_event)

        # Decoding stage
        decoding_start_event.record()
        for _ in range(1, args.decoding_length):
            with torch.no_grad():
                outputs = model(input_ids, use_cache=True, past_key_values=past_key_values)
                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values

                gen_token = torch.argmax(logits, dim=-1).unsqueeze(1)
                # input_ids = torch.cat([input_ids, gen_token], dim=-1)
                input_ids = gen_token

            torch.cuda.empty_cache()

        decoding_end_event.record()

        torch.cuda.synchronize()
        elapsed_time = decoding_start_event.elapsed_time(decoding_end_event)
        runing_times.append((elapsed_time, original_length, prefilling_time))
    
    avg_time = sum([time[0] for time in runing_times]) / len(runing_times)
    avg_prefilling_time = sum([time[2] for time in runing_times]) / len(runing_times)
    avg_context_length = sum([time[1] for time in runing_times]) / len(runing_times)
    peak_memory = torch.cuda.max_memory_allocated()
    print(f"Avg time: {avg_time} ms, Avg prefilling time: {avg_prefilling_time}, Avg context length: {avg_context_length}.")
    print(f"Total instance: {args.iter}, Mode: {args.mode}")
    print(f"Peak memory footprint: {peak_memory / 1024**2:.2f} MB.")

    if args.save_path:
        os.makedirs(args.save_path, exist_ok=True)
        file_name = f"{args.mode}_prefilling{args.prefilling_length}_decoding{args.decoding_length}_{args.iter}_sparsity{args.sparsity}.json"
        file_path = os.path.join(args.save_path, file_name)
        res = {
            'Avg Decoding Time': avg_time,
            'Avg Prefilling Time': avg_prefilling_time,
            'Prefilling length': avg_context_length,
            'Decoding length': args.decoding_length,
            'Mode': args.mode,
            'Peak Memory (MB)': peak_memory / 1024**2,
            'Iter': args.iter,
            'Sparsity': args.sparsity,
        }
        with open(file_path, 'w') as f:
            json.dump(res, f, indent=4)


if __name__ == '__main__':
    args = parse_args()
    main(args)
