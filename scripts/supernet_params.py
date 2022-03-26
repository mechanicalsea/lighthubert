"""
Sample a number of subnets and calculate their number of parameters.
"""
import argparse
import json
import os.path as op
from tqdm import tqdm

from fairseq.models.compress_hubert.hubert_pruner import HubertPrunerConfig, HubertPruner
from fairseq.tasks.compress_hubert import CompressHubertPretrainingConfig, CompressHubertPretrainingTask

def process(args, model):
    arch_sampler = model.supernet
    model.set_sample_config(arch_sampler.max_subnet)
    results = []
    for i in tqdm(range(args.num_sample), desc="calc params", total=args.num_sample):
        subnet = arch_sampler.sample_subnet()
        model.set_sample_config(subnet)
        param = model.calc_sampled_param_num()
        results.append({'architecture': subnet, "parameter": param})

    with open(args.output_path.replace(".json", f"-{args.num_sample}.json"), "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="data and label dir")
    parser.add_argument("--normalize", default=True, type=bool, help="extractor mode")
    parser.add_argument("--extractor-mode", default="layer_norm", choices=["layer_norm", "default"])
    parser.add_argument("--supernet-yaml", type=str, required=True, help="supernet dir")
    parser.add_argument("--num-sample", type=int, default=1000, help="number of sampled subnets")
    parser.add_argument("--output-path", type=str, default='', help="output path")
    args = parser.parse_args()
    assert (args.normalize and args.extractor_mode == "layer_norm") or (
        not args.normalize and args.extractor_mode == "default"
    ), f"--normalize {args.normalize} mismatch --extractor-mode {args.extractor_mode}"
    if args.output_path == "":
        args.output_path = "results.json"
    assert op.exists(op.dirname(args.output_path)), args.output_path
    return args

def get_model(args):
    # model config
    model_cfg = HubertPrunerConfig()
    model_cfg._name = "hubert_pruner"
    model_cfg.label_rate = 50
    model_cfg.extractor_mode = args.extractor_mode
    model_cfg.final_dim = 256
    model_cfg.encoder_layerdrop = 0.05
    model_cfg.untie_final_proj = False
    model_cfg.pruner_supernet = args.supernet_yaml

    # task config
    task_cfg = CompressHubertPretrainingConfig()
    task_cfg._name = "compress_hubert_pretraining"
    task_cfg.data = args.data_dir
    task_cfg.label_dir = task_cfg.data
    task_cfg.labels = ['km']
    task_cfg.label_rate = 50
    task_cfg.sample_rate = 16000
    task_cfg.max_sample_size = 320000
    task_cfg.min_sample_size = 32000
    task_cfg.pad_audio = False
    task_cfg.random_crop = True
    task_cfg.normalize = args.normalize
    task_cfg.freeze_extractor = False
    task_cfg.subnet_log = False

    task = CompressHubertPretrainingTask(task_cfg)
    model = HubertPruner(model_cfg, task.cfg, task.dictionaries)

    return model

def main():
    args = get_options()
    model = get_model(args)
    process(args, model)

if __name__ == "__main__":
    main()
    