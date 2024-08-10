from typing import OrderedDict
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

class OrthoLinear(torch.nn.Linear):
    def reset_parameters(self):
        torch.nn.init.orthogonal_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


class XavierLinear(torch.nn.Linear):
    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


class NNModel(nn.Module):
    def __init__(self, config):
        """Instantiates NN linear model with arguments from

        Args:
            config (args): Model Configuration parameters.
        """
        if config["actfxn"] == 'relu':
            actfxn = nn.ReLU()
        elif config["actfxn"] == 'elu':
            actfxn = nn.ELU()
        elif config["actfxn"] == 'gelu':
            actfxn = nn.GELU()
        elif config["actfxn"] == 'selu':
            actfxn = nn.SELU()
        else:
            raise ValueError(f"Unsupported activation function: {config['actfxn']}")

        if config["linear_type"] == 'ortho':
            linear_layer = OrthoLinear
        elif config["linear_type"] == 'xavier':
            linear_layer = XavierLinear
        else:
            raise ValueError(f"Unsupported linear layer type: {config['linear_type']}")

        input_size = config["input_size"]
        emb_size = config["emb_size"]
        hidden_size = emb_size // 2

        super(NNModel, self).__init__()
        self.embeds: nn.Sequential = nn.Sequential(
            nn.Linear(config["input_size"], emb_size),
            actfxn,
            linear_layer(emb_size, hidden_size),
            actfxn,
        )

        layers = []
        prev_size = hidden_size
        while prev_size > 16:
            next_size = prev_size // 2
            layers.append(nn.Sequential(
                linear_layer(prev_size, next_size), 
                actfxn
            ))
            prev_size = next_size
        
        self.linearlayers = nn.ModuleList(layers)
        self.output = nn.Linear(prev_size, config["output_size"])

        # manual linearlayer definition
        # self.linearlayers: nn.ModuleList = nn.ModuleList([
        #     nn.Sequential(OrthoLinear(config["hidden_size"], 256), nn.ReLU()),
        #     nn.Sequential(OrthoLinear(256, 128), nn.ReLU()),
        #     nn.Sequential(OrthoLinear(128,64), nn.ReLU()),
        #     nn.Sequential(OrthoLinear(64, 32), nn.ReLU()),
        #     nn.Sequential(OrthoLinear(32, 32), nn.ReLU()),
        #     # nn.Sequential(OrthoLinear(32, 16), nn.ReLU())
        # ])
        # self.output: nn.Linear = nn.Linear(32, config["output_size"])


    def forward(self, x: torch.tensor):
        """
        Args:
            x (torch.tensor): Shape[batch_size, input_size]

        Returns:
            _type_: _description_
        """
        embeds: torch.tensor = self.embeds(x)
        for i, layer in enumerate(self.linearlayers):
            embeds: torch.tensor = layer(embeds)
        output: torch.tensor = self.output(embeds)
        
        return torch.sigmoid(output) 

