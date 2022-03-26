# import hf_env
# hf_env.set_env('202111')
# import sys
# sys.path=["/ceph-jd/pub/jupyter/xiongzhx/notebooks/xzx/site-packages/"] + sys.path
# sys.path=["/ceph-jd/pub/jupyter/xiongzhx/notebooks/xzx/fairseq/fairseq/"] + sys.path

import argparse
import logging
import os
import sys

import torch
from collections.abc import Iterable
from examples.compress_hubert.search_algorithm import RandomHubertSupernet, EAHubertSupernet
from fairseq import checkpoint_utils, utils, tasks
from fairseq.models.compress_hubert import HubertPruner
from fairseq.logging import progress_bar
from fairseq.utils import reset_logging
from omegaconf import OmegaConf
from pathlib import Path


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("search.subnet")

def clever_format(nums, format="%.2f"):
    """
    Copy from link:
    https://github.com/Lyken17/pytorch-OpCounter/blob/master/thop/utils.py#L28
    """
    if not isinstance(nums, Iterable):
        nums = [nums]
    clever_nums = []

    for num in nums:
        if num > 1e12:
            clever_nums.append(format % (num / 1e12) + "T")
        elif num > 1e9:
            clever_nums.append(format % (num / 1e9) + "G")
        elif num > 1e6:
            clever_nums.append(format % (num / 1e6) + "M")
        elif num > 1e3:
            clever_nums.append(format % (num / 1e3) + "K")
        else:
            clever_nums.append(format % num + "B")

    clever_nums = clever_nums[0] if len(clever_nums) == 1 else (*clever_nums,)

    return clever_nums

def main(args):
    reset_logging()

    if not os.path.exists(args.result_path):
        os.mkdir(args.result_path)
    if args.result_file is None:
        args.result_file = f"subnet_losses_{args.search_algorithm}_last.pt"
    if args.max_params is None:
        args.max_params = float("inf")

    use_fp16 = True
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        torch.cuda.set_device(0)

    # Load task and model
    logger.info("loading model(s) from {}".format(args.path))
    state = checkpoint_utils.load_checkpoint_to_cpu(args.path)
    saved_cfg = state["cfg"]
    saved_cfg.task.data = args.data
    saved_cfg.task.label_dir = args.data

    if not hasattr(saved_cfg.model, "pruner_supernet"):
        OmegaConf.set_struct(saved_cfg.model, False)
        saved_cfg.model.pruner_supernet = ""
        OmegaConf.set_struct(saved_cfg.model, True)

    task = tasks.setup_task(saved_cfg.task)
    task.load_state_dict(state["task_state"])

    model = HubertPruner(saved_cfg.model, task.cfg, task.dictionaries)
    model.load_state_dict(state["model"])
    del state

    assert saved_cfg.task._name == "compress_hubert_pretraining", saved_cfg.task
    assert hasattr(model, "set_sample_config")

    # Move models to GPU
    model.eval()
    if use_cuda:
        model.cuda()
    if use_fp16:
        model.half()

    # Build criterion
    saved_cfg.criterion.teacher_from_model = args.teacher_path
    criterion = task.build_criterion(saved_cfg.criterion)
    criterion.eval()
    if use_cuda:
        criterion.cuda()
    if use_fp16:
        criterion.half()

    # Print args
    logger.info(saved_cfg)
    logger.info(args)

    # Build search algorithm
    assert args.search_algorithm in ["random", "ea"]
    if args.search_algorithm == "random":
        searcher = RandomHubertSupernet(Path(args.search_space_yaml), args.num_subnet_batch)
    else:
        searcher = EAHubertSupernet(Path(args.search_space_yaml))
    logger.info(f"{searcher.search_strategy}")

    # Load dataset
    subset = "valid"
    task.load_dataset(subset, combine=False, epoch=1, task_cfg=saved_cfg.task)
    dataset = task.dataset(subset)
    def _move_to_fp16(tensor):
        return tensor.half()

    # initialize subnet
    model.set_sample_config(searcher.max_subnet)
    
    # random search
    if args.search_algorithm == "random":
        # Prepare subnet, min subnet, and max subnet as the first validation
        bounded_subnets = [searcher.subnet, searcher.min_subnet, searcher.max_subnet]
        if not os.path.exists(os.path.join(args.result_path, args.result_file)):
            results_subnets = []
            start_iteration = -1
        else:
            tmp = torch.load(
                os.path.join(args.result_path, args.result_file),
                map_location="cpu",
            )
            results_subnets = tmp["result"]
            start_iteration = tmp["iteration"]
            logger.info(f"loaded results checkpoint from {start_iteration} iteration")
        for i in range(args.num_iteration + 1):
            batch_subnets = bounded_subnets if i == 0 else searcher.batch_search()
            if i <= start_iteration:
                # repeat randomly batch search in a given times
                continue
            for subnet in batch_subnets:
                # Set a subnet
                model.set_sample_config(subnet)
                model_params = model.calc_sampled_param_num()
                if model_params <= args.max_params:
                    # Initialize data iterator
                    itr = task.get_batch_iterator(
                        dataset=dataset,
                        max_sentences=1 if args.max_tokens is None else None,
                        max_tokens=args.max_tokens,
                        num_workers=0,
                        data_buffer_size=0,
                    ).next_epoch_itr(shuffle=False)
                    progress = progress_bar.progress_bar(
                        itr,
                        epoch=i,
                        prefix=f"{subnet} | {model_params:,} Params",
                        default_log_format="tqdm",
                    )
                    log_outputs = []
                    for j, sample in enumerate(progress):
                        if args.debug:
                            if j > 1: break
                        sample = utils.move_to_cuda(sample) if use_cuda else sample
                        sample = utils.apply_to_sample(_move_to_fp16, sample) if use_fp16 else sample
                        _loss, _sample_size, log_output = task.valid_step(sample, model, criterion)
                        progress.log(log_output, step=j)
                        log_output["loss"] *= sample["id"].numel()
                        log_outputs.append(log_output)
                    loss = sum(log_output["loss"] for log_output in log_outputs) / len(dataset)
                    log_output = {"loss": loss}
                    progress.print(log_output, tag=subset, step=i)
                else:
                    loss = None
                results_subnets.append({
                    "subnet": subnet,
                    "params": model_params,
                    "loss": loss,
                })
            if (i > 0 and i % args.save_interval == 0) or (i == args.num_iteration):
                pt_result_path = os.path.join(args.result_path, args.result_file)
                torch.save({
                    "result": results_subnets,
                    "iteration": i,
                }, pt_result_path) 
                logger.info(f"saved {i} iterations result at {pt_result_path}")
        logger.info(f"finished subnet search in {i} iteration.")
    else: 
        # evolutionary algorithm
        import geatpy as ea
        import numpy as np

        @ea.Problem.single
        def evalVars(Var):
            subnet = searcher.encode2subnet(Var)
            # Set a subnet
            model.set_sample_config(subnet)
            model_params = model.calc_sampled_param_num()
            if model_params <= args.max_params:
                # Initialize data iterator
                itr = task.get_batch_iterator(
                    dataset=dataset,
                    max_sentences=1 if args.max_tokens is None else None,
                    max_tokens=args.max_tokens,
                    num_workers=args.num_workers,
                    data_buffer_size=10,
                ).next_epoch_itr(shuffle=False)
                log_outputs = []
                log_sizes = []
                for j, sample in enumerate(itr):
                    if args.debug:
                        if j > 1: break
                    sample = utils.move_to_cuda(sample) if use_cuda else sample
                    sample = utils.apply_to_sample(_move_to_fp16, sample) if use_fp16 else sample
                    _loss, _sample_size, log_output = task.valid_step(sample, model, criterion)
                    log_outputs.append(log_output)
                    log_sizes.append(sample["id"].numel())
                loss = sum(
                    log_output["loss"] * log_size for log_output, log_size in zip(log_outputs, log_sizes)
                ) / len(dataset)
                logger.info(f"subnet | {subnet} | {model_params:,} Params | loss {loss}")
                return loss, 0
            else:
                return 10.0, 1
        
        lb = [0 for _ in range(searcher.degrees_of_freedom)]
        ub = \
            [len(searcher.search_space["embed_dim"]) - 1] + \
            [len(searcher.search_space["ffn_ratio"]) - 1 for _ in range(searcher.max_depth)] + \
            [len(searcher.search_space["heads_num"]) - 1 for _ in range(searcher.max_depth)] + \
            [len(searcher.search_space["layer_num"]) - 1]
        problem = ea.Problem(
            name='subnet search', M=1, maxormins=[1], Dim=searcher.degrees_of_freedom,
            varTypes=[1 for _ in range(searcher.degrees_of_freedom)], lb=lb, ub=ub,
            evalVars=evalVars
        )
        algorithm = ea.soea_SEGA_templet(
            problem, ea.Population(Encoding='RI', NIND=args.num_subnet_batch), 
            MAXGEN=args.num_iteration, logTras=args.save_interval
        )
        Params = clever_format(args.max_params, format='%.0f')
        ans_file = args.result_file.replace(
            "_last.pt", 
            f"_{Params}_NIND{args.num_subnet_batch}_MAXGEN{args.num_iteration}"
        )
        ans = ea.optimize(
            algorithm, seed=1, verbose=True, drawing=1, outputMsg=True, drawLog=False, saveFlag=True, 
            dirName=os.path.join(args.result_path, ans_file)
        )
        logger.info(ans)
        best_subnet = searcher.encode2subnet(ans['Vars'][0])
        model.set_sample_config(best_subnet)
        model_params = model.calc_sampled_param_num()
        best_subnet_des = f"{best_subnet}".replace("'", '"')
        logger.info(f"Best subnet | {best_subnet_des} | {model_params:,} Params | loss {ans['ObjV'][0]}")
        logger.info("finished subnet search via evolutionary algorithm.")

def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--search-space-yaml", 
        required=True, 
        type=str, 
        help="yaml to nas search space, i.e., supernet"
    )
    parser.add_argument(
        "--search-algorithm", 
        default="random", 
        choices=["random", "ea"], 
        help="search alogrithm to subnet"
    )
    parser.add_argument("--data", required=True, type=str, help="path to data and labels")
    parser.add_argument("--path", required=True, type=str, help="path to checkpoint")
    parser.add_argument("--result-path", required=True, type=str, help="path to result")
    parser.add_argument("--result-file", default=None, type=str, help="file to result")
    parser.add_argument("--teacher-path", required=True, type=str, help="path to teacher checkpoint")
    parser.add_argument("--max-params", default=None, type=float, help="maximum parameters as upper bound")
    parser.add_argument("--max-tokens", default=1900000, type=int)
    parser.add_argument("--num-workers", default=6, type=int)
    parser.add_argument("--num-subnet-batch", default=1, type=int)
    parser.add_argument("--num-iteration", default=1000, type=int)
    parser.add_argument("--save-interval", default=1, type=int)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)

if __name__ == "__main__":
    cli_main()
