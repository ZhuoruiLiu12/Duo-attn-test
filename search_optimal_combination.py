import argparse 
from itertools import combinations
from tqdm import tqdm
import math
import numpy as np
import json
import os
from multiprocessing import Pool, cpu_count
from functools import partial

from duo_attn.utils import load_attn_pattern, sparsify_attention_heads


def loss_only(baseline_attn_heads_classification, curr_attn_heads, baseline_attn_heads, *args):
    full_heads_mask = (baseline_attn_heads_classification == 1)
    diff_mask = (curr_attn_heads != baseline_attn_heads_classification)
    target_indices = full_heads_mask & diff_mask

    return baseline_attn_heads[target_indices].sum()


def loss_with_gain(baseline_attn_heads_classification, curr_attn_heads, baseline_attn_heads, *args):
    full_heads_mask = (baseline_attn_heads_classification == 1)
    streaming_heads_mask = (baseline_attn_heads_classification == 0)
    diff_mask = (curr_attn_heads != baseline_attn_heads_classification)

    loss_indices = full_heads_mask & diff_mask
    gain_indices = streaming_heads_mask & diff_mask

    return baseline_attn_heads[loss_indices].sum() - baseline_attn_heads[gain_indices].sum()


def loss_minimize_number_of_changed_heads(baseline_attn_heads_classification, curr_attn_heads, baseline_attn_heads, *args):
    # The same as all attention heads have the same score.
    baseline_attn_heads = np.full_like(baseline_attn_heads, 1)
    diff_mask = (curr_attn_heads != baseline_attn_heads_classification)

    return baseline_attn_heads[diff_mask].sum()


def loss_with_wighted_gain(baseline_attn_heads_classification, curr_attn_heads, baseline_attn_heads, *args):
    # gain is not as important as loss, we multiply a weighted to reduce the importance of gain in final loss.
    full_heads_mask = (baseline_attn_heads_classification == 1)
    streaming_heads_mask = (baseline_attn_heads_classification == 0)
    diff_mask = (curr_attn_heads != baseline_attn_heads_classification)

    loss_indices = full_heads_mask & diff_mask
    gain_indices = streaming_heads_mask & diff_mask

    return baseline_attn_heads[loss_indices].sum() - baseline_attn_heads[gain_indices].sum() * args[0].loss_weight

loss_func = {
    "loss_only": loss_only,
    "loss_with_gain": loss_with_gain,
    "loss_minimize_number_of_changed_heads": loss_minimize_number_of_changed_heads,
    "loss_with_weighted_gain": loss_with_wighted_gain,
}

def calculate_loss(combo, baseline_attn_heads, baseline_attn_heads_classification, args):
    curr_attn_heads = np.full_like(baseline_attn_heads_classification, 0)
    curr_attn_heads[combo, :] = 1

    res = loss_func[args.loss_type](baseline_attn_heads_classification, curr_attn_heads, baseline_attn_heads, args)

    return res, combo, curr_attn_heads


def process_combo(combo, baseline_attn_heads, baseline_attn_heads_classification, args):
    min_loss = float('inf')
    part_final_heads = None
    part_final_combo = None
    for com in combo:
        part_loss, part_combo, part_attn_heads = calculate_loss(com, baseline_attn_heads, baseline_attn_heads_classification, args)
        if min_loss > part_loss:
            min_loss = part_loss
            part_final_combo = part_combo
            part_final_heads = part_attn_heads
    return min_loss, part_final_combo, part_final_heads


def main(args):
    baseline_attn_heads, sink_size, recent_size = load_attn_pattern(
        args.model_attn_path
    )
    model_name = args.model_attn_path.split('/')[-3]

    baseline_attn_heads_classification, sparsity = sparsify_attention_heads(baseline_attn_heads, sparsity=args.sparsity)
    selected_layers = math.ceil(args.total_layers * args.sparsity)

    min_loss = float('inf')
    final_combo = None
    final_attn_heads = None

    # Create a pool of workers
    with Pool(cpu_count() // 4) as pool:
        # Generation all combinations
        combos = list(combinations(range(args.total_layers), selected_layers))

        chunk_size = len(combos) // 10000
        combos = [combos[i: i + chunk_size] for i in range(0, len(combos), chunk_size)]

        # Use imap to process combinations in parallel
        bounded_process_combo = partial(process_combo, baseline_attn_heads=baseline_attn_heads, baseline_attn_heads_classification=baseline_attn_heads_classification, args=args)
        results = tqdm(pool.imap(
            bounded_process_combo,
            combos
        ), total=len(combos), desc="Calculate progress.")

        for res, combo, curr_attn_heads in results:
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
    """
    Command line:
    python search_optimal_combination.py --model_attn_path /disk12/liuzhuorui/works/duo-attention/attn_patterns/Llama-3-8B-Instruct-Gradient-1048k/lr=0.02-reg=0.05-ctx=1000_32000-multi_passkey10 \
                                         --sparsity 0.5 \
                                         --total_layers 32 \
                                         --save_path /disk12/liuzhuorui/works/duo-attention/optimal_combination_res_refined
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_attn_path", type=str)
    parser.add_argument("--sparsity", type=float)
    parser.add_argument("--total_layers", type=int)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--loss_type", type=str, default="loss_only")

    parser.add_argument("--loss_weight", type=float, default=0.5)
    
    args = parser.parse_args()
    main(args)