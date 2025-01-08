import argparse 
from itertools import combinations
from tqdm import tqdm
import math
import numpy as np
import json
import os

from duo_attn.utils import load_attn_pattern, sparsify_attention_heads


def main(args):
    baseline_attn_heads, sink_size, recent_size = load_attn_pattern(
        args.model_attn_path
    )
    model_name = args.model_attn_path.split('/')[-3]

    baseline_attn_heads_classification, sparsity = sparsify_attention_heads(baseline_attn_heads, sparsity=args.sparsity)
    selected_layers = math.ceil(args.total_layers * args.sparsity)

    min_loss = 100
    for combo in tqdm(combinations(range(args.total_layers), selected_layers), desc="Calculate progress."):
        # Calculate the difference of selected combination with baseline.
        curr_attn_heads = np.full_like(baseline_attn_heads_classification, 0)
        curr_attn_heads[combo, :] = 1
        
        if args.loss_type == "loss_only":
            mask = (baseline_attn_heads_classification == 1)
            diff = (curr_attn_heads != baseline_attn_heads_classification)
            diff = mask & diff 
            res = baseline_attn_heads[diff].sum()
        else:
            # calculate the gain simultaneously
            loss_mask = (baseline_attn_heads_classification == 1)
            gain_mask = (baseline_attn_heads_classification == 0)
            diff = (curr_attn_heads != baseline_attn_heads_classification)

            loss_index = loss_mask & diff
            gain_index = gain_mask & diff
            res = baseline_attn_heads[loss_index].sum() - baseline_attn_heads[gain_index].sum()


        if min_loss > res:
            min_loss = res
            final_combo = combo
            final_attn_heads = curr_attn_heads
    
    final_res = {
        "Min Loss": float(min_loss),
        "Combination": list(final_combo),
        "Attention Heads": final_attn_heads.tolist()
    }

    os.makedirs(args.save_path, exist_ok=True)
    file_name = f"{model_name}_{args.loss_type}_{sparsity}_{args.total_layers}_{selected_layers}.json"
    file_path = os.path.join(args.save_path, file_name)
    with open(file_path, 'w') as f:
        json.dump(final_res, f, indent=4)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_attn_path", type=str)
    parser.add_argument("--sparsity", type=float)
    parser.add_argument("--total_layers", type=int)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--loss_type", type=str, default="loss_only")

    args = parser.parse_args()
    main(args)