import logging
import random
import numpy as np
import geatpy as ea

from pathlib import Path
from fairseq.models.compress_hubert import HubertSupernetConfig
from typing import List

class RandomHubertSupernet(HubertSupernetConfig):
    """Random search HuBERT Supernet.
    func:
        .batch_search(): sample a batch of subnet randomly
    """

    def __init__(self, yaml_path: Path, batch_size: int = 1):
        super().__init__(yaml_path)
        self.strategy = "random"
        self.batch_size = batch_size
        self.cur_batch = []
        self.his_batch = []
        random.seed(1)
    
    @property
    def search_strategy(self):
        return f"Random Search in {self.search_space}"
    
    def batch_search(self) -> List:
        self.his_batch.extend(self.cur_batch)
        self.cur_batch = []
        for _ in range(self.batch_size):
            self.cur_batch.append(self.sample_subnet())
        return self.cur_batch

class EAHubertSupernet(HubertSupernetConfig):
    def __init__(self, yaml_path: Path, batch_size: int = 1):
        super().__init__(yaml_path)
        self.strategy = "EA"
        np.random.seed(1)
        self.dict_subnet2encode = {
            "atten_dim": {ai: i for i, ai in enumerate(self.search_space["atten_dim"])},
            "embed_dim": {ei: i for i, ei in enumerate(self.search_space["embed_dim"])},
            "ffn_ratio": {f"{fi:.2f}": i for i, fi in enumerate(self.search_space["ffn_ratio"])},
            "heads_num": {hi: i for i, hi in enumerate(self.search_space["heads_num"])},
            "layer_num": {li: i for i, li in enumerate(self.search_space["layer_num"])},
        }
        self.dict_encode2subnet = {
            "atten_dim": {i: ai for i, ai in enumerate(self.search_space["atten_dim"])},
            "embed_dim": {i: ei for i, ei in enumerate(self.search_space["embed_dim"])},
            "ffn_ratio": {i: fi for i, fi in enumerate(self.search_space["ffn_ratio"])},
            "heads_num": {i: hi for i, hi in enumerate(self.search_space["heads_num"])},
            "layer_num": {i: li for i, li in enumerate(self.search_space["layer_num"])},
        }
    
    @property
    def search_strategy(self):
        return f"Evolutionary Algorithm in {self.search_space}"

    @property
    def max_depth(self):
        return max(self.search_space["layer_num"])

    @property
    def degrees_of_freedom(self):
        return len(
            ["embed_dim"] + ["ffn_ratio"] * self.max_depth + \
            ["heads_num"] * self.max_depth + ["layer_num"]
        )
    
    def subnet2encode(self, subnet: dict):
        """Count the number of subnets in the supernet.
        search_space:
            embed_dim: List[int]
            ffn_ratio: List[float]
            heads_num: List[int]
            layer_num: List[int]
        
        Returns
        -------
            ndarray[int]: in sequence of [embed_dim, ffn_ratio, heads_num, layer_num]
        """
        subnet_seq = []
        for key in ["embed_dim", "ffn_ratio", "heads_num", "layer_num"]:
            if key == "ffn_ratio" and "ffn_ratio" not in subnet:
                assert "embed_dim" in subnet and "ffn_embed" in subnet
                subnet["ffn_ratio"] = [f"{xi / subnet['embed_dim']:.2f}" for xi in subnet[key]]
            seq_i = [self.dict_subnet2encode[key][xi] for xi in subnet[key]]
            if key not in ["embed_dim", "layer_num"] and len(seq_i) < self.max_depth:
                seq_i += [0 for _ in range(len(seq_i), self.max_depth)]
            subnet_seq += seq_i
        return np.array(subnet_seq, dtype=int)

    def encode2subnet(self, subnet_seq: np.ndarray):
        """sequence of [embed_dim, ffn_ratio, heads_num, layer_num]"""
        layer_num = self.dict_encode2subnet["layer_num"][subnet_seq[-1]]
        embed_dim = self.dict_encode2subnet["embed_dim"][subnet_seq[0]]
        ffn_ratio = [
            self.dict_encode2subnet["ffn_ratio"][subnet_seq[i]] 
            for i in range(1, 1 + layer_num)
        ]
        ffn_embed = [int(xi * embed_dim) for xi in ffn_ratio]
        heads_num = [
            self.dict_encode2subnet["heads_num"][subnet_seq[i]] 
            for i in range(1 + self.max_depth, 1 + self.max_depth + layer_num)
        ]
        atten_dim = [int(xi * 64) for xi in heads_num]
        slide_wsz = ["global" for _ in range(layer_num)]
        return {
            "atten_dim": atten_dim, # List[int]
            "embed_dim": embed_dim, # int
            "ffn_embed": ffn_embed, # List[int]
            "heads_num": heads_num, # List[int]
            "layer_num": layer_num, # int
            "slide_wsz": slide_wsz, # List[int] or List[str]
        }
