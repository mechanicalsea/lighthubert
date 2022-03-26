"""
Usage:
python search_results.py --search-results ${result_file} --max-params 1e8
"""

import argparse
import os
import torch
import numpy

def process(args):
    if not os.path.exists(args.search_results):
        raise FileNotFoundError(args.search_results)
    print(f"\nSubnet search less than {args.max_params:.0f} parameters from {args.search_results}\n")
    results = torch.load(args.search_results, map_location="cpu")["result"]
    best_id = numpy.argmin([
        r["loss"] if r["loss"] is not None and r["params"] <= args.max_params else float("inf") for r in results
    ])
    print(f"{'='*20} Reference Subnet {'='*20}")
    print(f"{results[0]}".replace("'", '"'))
    print(f"{'='*20} Minimal Subnet {'='*20}")
    print(f"{results[1]}".replace("'", '"'))
    print(f"{'='*20} Maximal Subnet {'='*20}")
    print(f"{results[2]}".replace("'", '"'))
    if best_id != 0:
        print(f"{'='*20} Best Subnet {'='*20}")
        print(f"{results[best_id]}".replace("'", '"'))
    else:
        print(f"{'='*20} Best Subnet {'='*20}")
        print("No available subnets found, please increase the maximum parameters.")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-results", required=True, type=str)
    parser.add_argument("--max-params", required=True, type=float)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    process(get_args())
