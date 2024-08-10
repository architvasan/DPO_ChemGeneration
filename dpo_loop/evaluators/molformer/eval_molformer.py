from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForMaskedLM
from pathlib import Path
import pandas as pd
import sklearn.model_selection
import torch
import wandb
import pdb
from dpo_loop.evaluators.molformer.classification_layer import NNModel
from dpo_loop.evaluators.molformer.data_utils import CustomDataset_inf
import torch.nn.functional as F
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModel, AutoTokenizer
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path
from torch import nn
import intel_extension_for_pytorch as ipex


def evaluate(data, modelcheckpoint, good_index, device=0):
    LLModel = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
    device = torch.device(f"xpu:{device}")
    LLModel.to(device)
    LLModel.eval()

    test_set = CustomDataset_inf(tokenizer, data, max_input_length=512, max_target_length=512)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=16, shuffle=False)

    nnmodel = NNModel(config={"input_size": 768, "embedding_size": 512, "hidden_size": 256, "output_size": 2, "n_layers": 5}).to("xpu")
    checkpoint = torch.load(modelcheckpoint, map_location=torch.device('cpu'))
    nnmodel.load_state_dict(checkpoint)#['model_state_dict'])
    softmax_nnmodel = nn.Softmax(dim=1)
    smiles_out = []
    out_vals_0 = []
    out_vals_1 = []

    out_bin = []
    for batch in test_dataloader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        with torch.no_grad():
            outputs = LLModel(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)#labels=labels)
            encoder = outputs["hidden_states"][-1]
            # inference regression head on top of encoder output to label
            # average over second dimension of encoder output to get a single vector for each example
            encoder = encoder.mean(dim=1)
            # pass encoder output to regression head
            nn_outputs = nnmodel(encoder)
            nn_outputs = softmax_nnmodel(nn_outputs)
            nn_outputs_vals_0 = nn_outputs.cpu().detach().numpy()[:,0]
            nn_outputs_vals_1 = nn_outputs.cpu().detach().numpy()[:,1]
            out_vals_0.extend(nn_outputs_vals_0)
            out_vals_1.extend(nn_outputs_vals_1)
            out_bin.extend(np.round(nn_outputs_vals_1))
            smiles_out.extend(batch["smiles"])#.cpu().detach().numpy())
            print(out_bin)
    return smiles_out, out_vals_0, out_vals_1, out_bin
