import argparse
import glob
import os
import json

import pandas as pd
import torch


def preprocess_data(file_list):
    df = pd.DataFrame(columns=["1k", "2k", "4k", "8k", "16k", "32k"], index=["1k", "2k", "4k", "8k", "16k", "32k"])
    for file in file_list:
        with open(file, "r") as f:
            data = json.load(f)
            row_idx = f"{data['Prefilling length'] // 1024:.0f}k"
            col_idx = f"{data['Decoding length'] // 1024:.0f}k"
            df.at[row_idx, col_idx] = (data['Avg Prefilling Time'], data['Avg Decoding Time'])
    return df
    
    
def save_result_file(save_path, data, mode):
    save_path = os.path.join(save_path)
    os.makedirs(save_path, exist_ok=True)
    save_file_path = os.path.join(save_path, mode + "_result.md")
    with open(save_file_path, "w") as f:
        f.write(data)
    

def main(args):
    duo_res_file_list = glob.glob(args.data_path + "/duo*")
    layer_wise_res_file_list = glob.glob(args.data_path + "/layer*")

    duo_res = preprocess_data(duo_res_file_list)
    layer_wise_res = preprocess_data(layer_wise_res_file_list)

    duo_markdown_res, layer_wise_markdown_res = duo_res.to_markdown(), layer_wise_res.to_markdown()
    print(duo_markdown_res)
    print(layer_wise_markdown_res)

    if args.save_path:
        save_result_file(args.save_path, duo_markdown_res, "duo")
        save_result_file(args.save_path, layer_wise_markdown_res, "layer-wise")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_path", type=str)
    
    args = parser.parse_args()
    main(args)