from transformers import AutoTokenizer, AutoModelForCausalLM
import selfies as sf
import os
import time
import json
import shutil
from tqdm import tqdm
from collections import defaultdict
from neptune.types import File
import argparse
import numpy as np
import pandas as pd
import neptune
from sklearn.model_selection import train_test_split
from neptune_pytorch import NeptuneLogger
from torch.optim import AdamW

import torch
from torch import nn
import torch.nn.functional as F
import intel_extension_for_pytorch as ipex
import dpo_loop.finetune
import dpo_loop.sample
from dpo_loop.evaluators.molformer import eval_molformer
def raw_sampling(first_inp_data, 
                    smiles_col,
                    label_col,
                    pos_label,
                    neg_label,
                    scaffolds_sampling,
                    batch_sample,
                    out_dir,
                    out_pattern,
                    evaluator_checkpoint,
                    loop_number,
                    goal_ratio,
                    device=0):
    """
    First: dpo finetune with input data
    """
    
    device_use = device
    
    print(device_use)
    os.environ["ZE_FLAT_DEVICE_HIERARCHY"]="FLAT"
    os.environ["ZE_AFFINITY_MASK"]=f'{device_use}'
    
    device = 0
    data_smi = pd.read_csv(first_inp_data)
    smiles_all = data_smi
    print(smiles_all)
    """
    Second: start loop:
        a. sample using pretrained model
        b. evaluate sample using user defined function --> 0s or 1s for each smiles. Using molformer classifier model for now 
            """
    sampled_smiles = dpo_loop.sample.sample(f'placeholder', scaffolds_sampling, smiles_col, batch_sample, device)
    smiles_out, out_vals_0, out_vals_1, out_bin = eval_molformer.evaluate(sampled_smiles, evaluator_checkpoint, pos_label, device)
    smiles_df = pd.DataFrame({smiles_col: smiles_out, label_col: out_bin, "prob0": out_vals_0, "prob1": out_vals_1})
    smiles_df.to_csv(f'{out_dir}/{out_pattern}.nofinetune.csv', index=False)
    
    return 






if False:
    ##########################################################################
    #------------------------------------------------------------------------
    ##########################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        required=True,
        type=str,
    )
    
    parser.add_argument(
        "--output_model",
        required=True,
        type=str,
    )
    
    parser.add_argument(
        "--smiles_col",
        type=str,
        default=0,
        help="device int",
    )
    
    parser.add_argument(
        "--label_col",
        type=str,
        default=0,
        help="device int",
    )
    
    parser.add_argument(
        "--pos_label",
        type=int,
        default=0,
        help="device int",
    )
    
    parser.add_argument(
        "--neg_label",
        type=int,
        default=0,
        help="device int",
    )
    
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="device int",
    )
    
    args = parser.parse_args()
    
    dpo_loop_main(first_inp_data, 
                        smiles_col,
                        label_col,
                        pos_label,
                        neg_label,
                        scaffolds_sampling,
                        batch_sample,
                        out_dir,
                        out_pattern,
                        evaluator_checkpoint,
                        loop_number,
                        goal_ratio,
                        device=0)
    
    
    
    data_smi = pd.read_csv(args.data)
    print(len(data_smi))
    print(data_smi[data_smi['label']==1])
    #data_smi = data_smi[data_smi['label']< -9]#['smiles']
    #print
    pos_smi = [sf.encoder(smi_it) for smi_it in data_smi[data_smi['label']==1]['SMILES']]
    neg_smi = [sf.encoder(smi_it) for smi_it in data_smi[data_smi['label']==0]['SMILES']]
    print(pos_smi)
    print(len(pos_smi))
    print(len(neg_smi))
    #print(data_smi)
    #train_smi = data_smi.sample(frac=0.8,random_state=200)#.tolist()
    #test_smi=data_smi.drop(train_smi.index).tolist()
    #train_smi = train_smi.tolist()
    
    tokenizer = AutoTokenizer.from_pretrained("ncfrey/ChemGPT-1.2B")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    policy = AutoModelForCausalLM.from_pretrained("ncfrey/ChemGPT-1.2B")
    ref = AutoModelForCausalLM.from_pretrained("ncfrey/ChemGPT-1.2B")
    max_length = 32
    
    DPO_Instruct = DPOInstruct(policy, ref, tokenizer, 0, 16, 50, 5e-7, max_length, 0.5) 
    losst, rewardt, loss_epoch, reward_epoch = DPO_Instruct.train(pos_smi, neg_smi, args.output_model)
    
    print(loss_epoch)
    print(reward_epoch)


if False:
    # Make train, val, and test datasets
    train_dataset = CustomDataset(tokenizer, train_smi,  max_length, max_length, device="xpu")
    val_dataset = CustomDataset(tokenizer, test_smi,  max_length, max_length, device="xpu")
    #print(train_dataset)
    #train_dataset, val_dataset = get_datasets(policy.tokenizer)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True) #get_dataloader(train_dataset, "train")
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False) #get_dataloader(train_dataset, "train")
    
    from torch.optim import AdamW
    
    optimizer = AdamW(policy.parameters(), lr=5e-5)
    
    from transformers import get_scheduler
    
    num_epochs = 3
    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    
    device = torch.device("xpu") #if torch.cuda.is_available() else torch.device("cpu")
    policy.to(device)
    
    from tqdm.auto import tqdm
    
    progress_bar = tqdm(range(num_training_steps))
    
    policy.train()
    for epoch in range(num_epochs):
        for batch in train_loader:
            #batch = {k: v.to(device) for k, v in batch.items()}
            #print(batch)
            outputs = policy(**batch)
            #print(outputs.logits)
            print(torch.nn.functional.log_softmax(outputs.logits, dim=-1))
            loss = outputs.loss
            loss.backward()
    
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
    
    
    #trainer = DPO(
    #    reference, policy, config, logger=npt_logger, run=npt_logger, device=args.device
    #)
    #trainer.save_configs(config["model_path"])
    #trainer.train(train_loader, val_loader)
