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


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, tokenizer, data,  max_input_length, max_target_length, device="xpu"):
        self.tokenizer = tokenizer
        self.data = data
        self.device = device
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = str(self.data[idx])
        print(data)
        labels = str(self.data[idx])

        # tokenize data
        inputs = self.tokenizer(data, max_length=self.max_input_length, truncation=True, padding="max_length", return_tensors="pt")
        labels = self.tokenizer(labels, max_length=self.max_target_length, truncation=True, padding="max_length", return_tensors="pt")

        return {
            "input_ids": inputs["input_ids"].flatten().to(self.device),
            "attention_mask": inputs["attention_mask"].flatten().to(self.device),
            "labels": labels["input_ids"].flatten().to(self.device),
        }


class DPOInstruct:

    """ DPO in instruct mode """

    def __init__(self, model, ref_model, tokenizer, device, batch_size, epochs, learning_rate, context_length, beta):
        """ Initialize:
            model
            reference model
            tokenizer
            beta
            context length
            learning rate
            optimizer
            batch size
            epochs
            """
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.beta = beta
        self.context_length = context_length
        self.device = "xpu"#device
        self.learning_rate = learning_rate
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)

        self.batch_size = batch_size
        self.epochs = epochs 

    def batch_logits_labels(self, fmodel, preferred_tokenizable_batch, unpreferred_tokenizable_batch):
        """ Function to get logits and labels for a batch of
             preferred and unpreferred sequences 
            """

        fmodel.to("xpu")
        # Tokenize the batch of preferred and unpreferred sequences
        preferred_tokens = self.tokenizer(preferred_tokenizable_batch, 
                                max_length=self.context_length, 
                                truncation=True,
                                padding='max_length', 
                                return_tensors='pt')
        unpreferred_tokens = self.tokenizer(unpreferred_tokenizable_batch, 
                                max_length=self.context_length, 
                                truncation=True,
                                padding='max_length', 
                                return_tensors='pt')

        # Forward pass with the model to return preferred and unpreferred batch of logits 
        preferred_logits = fmodel(preferred_tokens["input_ids"].to("xpu"),
                            attention_mask=preferred_tokens["attention_mask"].to("xpu"),
                            ).logits
        # print(f'preferred logits dtype: {preferred_logits.dtype}')

        unpreferred_logits = fmodel(unpreferred_tokens["input_ids"].to("xpu"),
                            attention_mask=unpreferred_tokens["attention_mask"].to("xpu"),
                            ).logits
        # print(f'unpreferred logits dtype: {unpreferred_logits.dtype}')

        # Labels given by input ids
        preferred_labels = preferred_tokens['input_ids'][:].to(self.device)
        unpreferred_labels = unpreferred_tokens['input_ids'][:].to(self.device)
        # print(f'preferred labels dtype: {preferred_labels.dtype}')
        # print(f'unpreferred labels dtype: {unpreferred_labels.dtype}')

        return preferred_logits, unpreferred_logits, preferred_labels, unpreferred_labels

    def logprobs_logits(self, logits, labels):
        """ Function to compute log probabilities using logits and labels """

        # Log softmax of logits
        logps = F.log_softmax(logits, dim=-1)

        # Gather along the last dimension with indices given by the label ids
        logps_label = torch.gather(logps, dim=2, index=labels.unsqueeze(2)).squeeze(-1)

        return logps_label

    def batch_seq_logprobs(self, logits, labels):
        """ Function to compute a batch of sequence log probabilities """

        logits = logits[:, :-1, :] # skip last logit
        logits_logsoftmax = logits.log_softmax(-1) # compute log softmax of logits

        labels = labels[:, 1:].clone() # clone labels

        # Loss mask to avoid padded tokens while computing loss
        loss_mask = labels != self.tokenizer.pad_token_id
        # print(f'Labels shape: {labels.shape}')
        # print(f'loss_mask shape: {loss_mask.shape}')
        # print(f'loss_mask dtype: {loss_mask.dtype}')

        # Gather logps and squeeze last dimension
        logprobs = torch.gather(logits_logsoftmax, dim=2, index=labels.unsqueeze(2)).squeeze(2)
        # print(f'seq_logprobs shape: {logprobs.shape}')

        # Weighted sum over logprobs using loss mask
        seq_logprobs = (logprobs * loss_mask).sum(-1)

        return seq_logprobs

    def loss_function_dpo(self,
                preferred_logprobs, 
                unpreferred_logprobs, 
                ref_preferred_logprobs, 
                ref_unpreferred_logprobs
                ):

        """ Function to compute DPO loss using preferred and 
            unpreferred log probabilities 
            """

        # Get ratios of preferred log probabilities from model and ref model
        preferred_logprob_ratio = preferred_logprobs - ref_preferred_logprobs

        # Get ratios of unpreferred log probabilities from model and ref model
        unpreferred_logprob_ratio = unpreferred_logprobs - ref_unpreferred_logprobs

        # Difference of logprobs ratios scaled by beta
        scaled_diff_logprob_ratios = self.beta * (preferred_logprob_ratio - unpreferred_logprob_ratio)

        # Losses computed as negative logsigmoid of scaled difference
        losses = -F.logsigmoid(scaled_diff_logprob_ratios)

        # preferred dpo rewards
        preferred_dpo_rewards = (self.beta * preferred_logprob_ratio).detach()

        # unpreferred dpo rewards
        unpreferred_dpo_rewards = (self.beta * unpreferred_logprob_ratio).detach()

        return losses, preferred_dpo_rewards, unpreferred_dpo_rewards

    def compute_dpo_loss(self, preferred_batch, unpreferred_batch):

        """ Function to compute DPO loss with preferred
            and unpreferred sequence batch
            """

        # Get preferred and unpreferred logits and labels
        pref_logits, unpref_logits, pref_labels, unpref_labels = self.batch_logits_labels(
                                                                    self.model,
                                                                    preferred_batch, 
                                                                    unpreferred_batch
                                                                    )

        # Compute preferred logprobs
        preferred_logps = self.batch_seq_logprobs(pref_logits, pref_labels)

        # Compute unpreferred logprobs
        unpreferred_logps = self.batch_seq_logprobs(unpref_logits, unpref_labels)

        # Get reference preferred and unpreferred logits and labels
        with torch.no_grad():
            ref_pref_logits, ref_unpref_logits, pref_labels, unpref_labels = self.batch_logits_labels(
                                                                                self.ref_model,
                                                                                preferred_batch, 
                                                                                unpreferred_batch
                                                                                )

            # Compute ref preferred logprobs
            ref_preferred_logps = self.batch_seq_logprobs(ref_pref_logits, pref_labels)

            # Compute ref unpreferred logprobs
            ref_unpreferred_logps = self.batch_seq_logprobs(ref_unpref_logits, unpref_labels)

        # Compute losses
        losses, pref_dpo_rewards, unpref_dpo_rewards = self.loss_function_dpo(
                                                            preferred_logps, 
                                                            unpreferred_logps, 
                                                            ref_preferred_logps, 
                                                            ref_unpreferred_logps
                                                            )

        # Implicit DPO rewards
        implicit_dpo_rewards = (pref_dpo_rewards > unpref_dpo_rewards).float()
        rewards = implicit_dpo_rewards.cpu().mean()

        # Compute mean loss
        loss = losses.mean()
        # print(f'Loss dtype: {loss.dtype}')

        return loss, rewards

    def train(self, tokenized_preferred, tokenized_unpreferred, output_model):

        from tqdm import tqdm
        """ Function to train the model using DPO loss """

        bs = self.batch_size
        epochs = self.epochs
        steps = min(len(tokenized_preferred), len(tokenized_unpreferred))//bs # steps per epoch
        print(f'num steps per epoch: {steps}')
        print(f'num epochs: {epochs}')

        # Model set to train
        self.model.train()

        # Lists to append mean loss and rewards per step
        losst = []
        rewardt = []

        # Lists to append loss and rewards per epoch
        loss_epoch = []
        reward_epoch = []

        # Training loop
        for epoch in range(epochs):
            for step in tqdm(range(steps)):

                # Get preferred and unpreferred tokenized batches
                pref_tokenized_batch = tokenized_preferred[step*bs:(step+1)*bs]
                unpref_tokenized_batch = tokenized_unpreferred[step*bs:(step+1)*bs]
                #print(len(pref_tokenized_batch), len(unpref_tokenized_batch))

                # DPO loss per batch
                loss, reward = self.compute_dpo_loss(pref_tokenized_batch, unpref_tokenized_batch)
                #print(f'step: {step}, loss:{loss.detach().tolist()}')
                #print(f'reward: {reward}')

                # Append loss and rewards per batch
                losst.append(loss.detach().tolist())
                rewardt.append(reward)

                # Loss backprop
                loss.backward()

                # Step optimizer
                self.optimizer.step()

                # # Learning rate scheduler step
                # lr_scheduler.step()

                # zero optimizer gradients
                self.optimizer.zero_grad()

            # Append mean losses and rewards per epoch
            loss_epoch_it = np.array(losst).mean()
            reward_epoch_it = np.array(rewardt).mean()
            print(loss_epoch_it)
            print(reward_epoch_it)
            loss_epoch.append(loss_epoch_it)
            reward_epoch.append(reward_epoch_it)

            if reward_epoch_it == np.max(np.array(reward_epoch)):
                torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': loss_epoch_it,
                        }, f"{output_model}")

        return losst, rewardt, loss_epoch, reward_epoch

def finetune(data_smi, checkpoint_fil, output_model, smiles_col, label_col, pos_label, neg_label, device=0):
    pos_smi = [sf.encoder(smi_it) for smi_it in data_smi[data_smi[label_col]==pos_label][smiles_col]]
    neg_smi = [sf.encoder(smi_it) for smi_it in data_smi[data_smi[label_col]==neg_label][smiles_col]]
    
    tokenizer = AutoTokenizer.from_pretrained("ncfrey/ChemGPT-1.2B")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    policy = AutoModelForCausalLM.from_pretrained("ncfrey/ChemGPT-1.2B")
    ref = AutoModelForCausalLM.from_pretrained("ncfrey/ChemGPT-1.2B")
    max_length = 32

    try:
        checkpoint = torch.load(checkpoint_fil)
        policy.load_state_dict(checkpoint['model_state_dict'])

    except:
        pass

    DPO_Instruct = DPOInstruct(policy, ref, tokenizer, 0, 16, 5, 5e-7, max_length, 0.5)
    losst, rewardt, loss_epoch, reward_epoch = DPO_Instruct.train(pos_smi, neg_smi, output_model)
    return loss_epoch, reward_epoch




####################################################################################
#### ------------------ OLD ------------------------------------------------------##
####################################################################################



if False:
    def dpo_loss(
        policy_pos_logprobs,
        policy_neg_logprobs,
        ref_pos_logprobs,
        ref_neg_logprobs,
        beta=0.5,
    ):
        """
        Computes DPO loss under human preference model given preferred/rejected
        log probabilities. See https://arxiv.org/pdf/2305.18290.pdf.
    
        Arguments:
            policy_pos_logprobs: log prob of preferred seqs under current model (batch_size,)
            policy_neg_logprobs: log prob of rejected seqs under current model (batch_size,)
            ref_pos_logprobs: log prob of preferred seqs under reference model (batch_size,)
            ref_neg_logprobs: log prob of rejected seqs under reference model (batch_size,)
            beta: temperature parameter (float scalar, default 0.5)
    
        Returns:
            loss, accepted, and rejected reward tensors (batch_size,)
        """
        policy_logratios = policy_pos_logprobs - policy_neg_logprobs
        ref_logratios = ref_pos_logprobs - ref_neg_logprobs
    
        loss = -F.logsigmoid(beta * (policy_logratios - ref_logratios))
        pos_rewards = beta * (policy_pos_logprobs - ref_pos_logprobs).detach()
        neg_rewards = beta * (policy_neg_logprobs - ref_neg_logprobs).detach()
    
        return loss, pos_rewards, neg_rewards
    
    
    class DPO:
        """Direct preference optimization for gpt/rnn models."""
    
        def __init__(
            self,
            reference,
            policy,
            config,
            device="xpu:0",
            logger=None,
            run=None,
        ):
            """
            Initializes DPO trainer. Sensible defaults are provided for training hyperparams,
            minus the 'model_path', which must be specified in the initial config. This is
            where checkpoints will be stored.
    
            Arguments:
                reference: pretrained reference model
                policy: pretrained policy model
                config: dictionary of training hyperparameters
                device: device to use for training (str, default "cuda:0")
                logger: logger to use for training (type NeptuneLogger, default None)
                run: run to log training progress (type neptune.run.Run, default None)
            """
            self.config = config
            self.dump_path = config["model_path"]
    
            self.beta = self.default("beta", 0.5)
            self.lr = self.default("lr", {"stop": 5e-7, "steps": 150})
            self.grad_norm_clip = self.default("grad_norm_clip", 10.0)
            self.gradient_accumulation = self.default("gradient_accumulation", 2)
            self.min_log_interval = self.default("min_log_interval", 2.0)
    
            self.eval_every = self.default("eval_every", 20000)
            self.max_epochs = self.default("max_epochs", 5)
    
            self.policy = policy
            self.reference = reference
    
            self.reference = reference.half()
            self.reference.eval()
    
            print(device)
            self.device = torch.device("xpu:0")#device)
            print(self.device)
            self.policy.to(self.device)
    
            self.reference.to(self.device)
    
            self.logger = logger
            self.run = run
            if run and not logger or logger and not run:
                raise ValueError("Must provide both logger and run, or neither")
            if self.logger:
                self.config["neptune"] = self.run.get_url()
                cfg = {k: (v if v is not None else "null") for k, v in self.config.items()}
                self.run[self.logger.base_namespace]["config"] = cfg
    
        def default(self, key, value):
            """
            Returns value if key is in config, otherwise sets key to value and returns value.
    
            Arguments:
                key: key to look up in config (str)
                value: default value to set if key is not in config (any)
            Returns:
                value, or self.config[key] if key is in self.config
            """
            if key in self.config:
                return self.config[key]
            else:
                self.config[key] = value
                return value
    
        def save_configs(self, dir, dpo_name="config-dpo.json", base_name="config.json"):
            """
            Saves base model and DPO training configs to directory.
    
            Arguments:
                dir: directory to save configs to (str)
                dpo_name: name of DPO config file (str, default "config-dpo.json")
                base_name: name of base config file (str, default "config.json")
            """
            os.makedirs(dir, exist_ok=True)
            with open(os.path.join(dir, base_name), "w") as f:
                json.dump(self.policy.config, f, indent=4)
            with open(os.path.join(dir, dpo_name), "w") as f:
                json.dump(self.config, f, indent=4)
    
        def loss(self, batch, prefix="train"):
            """
            Computes DPO loss and other metrics for a batch of data.
    
            Arguments:
                batch: batch of data (type dict, keys 'positive', 'negative',
                    'positive_length', 'negative_length')
                prefix: prefix to use for logging (str, default "train")
    
            Returns:
                loss, dict of metrics including acc, loss, margin, pos/neg rewards,
                pos/neg log probabilities
            """
            pos = batch["positive"].to(self.device)
            neg = batch["negative"].to(self.device)
            pos_len = batch["positive_length"].to(self.device)
            neg_len = batch["negative_length"].to(self.device)
    
            seqs = torch.cat([pos, neg])
            lengths = torch.cat([pos_len, neg_len])
    
            policy_logprobs = self.policy.logprobs(seqs, lengths)
            policy_pos, policy_neg = torch.split(policy_logprobs, pos.shape[0])
    
            with torch.no_grad():
                ref_logprobs = self.reference.logprobs(seqs, lengths)
                ref_pos, ref_neg = torch.split(ref_logprobs, pos.shape[0])
    
            loss, pos_reward, neg_reward = dpo_loss(
                policy_pos, policy_neg, ref_pos, ref_neg, beta=self.beta
            )
    
            acc = (pos_reward > neg_reward).float().mean()
            margin = (pos_reward - neg_reward).mean()
            pos_reward, neg_reward = pos_reward.mean(), neg_reward.mean()
            pos_logprobs, neg_logprobs = (
                policy_pos.detach().mean(),
                policy_neg.detach().mean(),
            )
            loss_val = loss.detach().mean()
    
            return loss.mean(), {
                f"{prefix}/acc": acc,
                f"{prefix}/loss": loss_val,
                f"{prefix}/margin": margin,
                f"{prefix}/pos_reward": pos_reward,
                f"{prefix}/neg_reward": neg_reward,
                f"{prefix}/pos_logprobs": pos_logprobs,
                f"{prefix}/neg_logprobs": neg_logprobs,
            }
    
        def log(self, metrics, examples):
            """
            Logs metrics to logger and run. Does nothing if no logger specified.
    
            Arguments:
                metrics: dictionary of metrics to log
                examples: number of examples seen so far (int)
            """
            if self.logger:
                for key, value in metrics.items():
                    if not isinstance(value, File):
                        value = value.item()
                    self.run[self.logger.base_namespace][key].append(
                        value=value, step=examples
                    )
    
        def save_checkpoint(self, metrics=None, examples=None):
            """
            Saves policy only to checkpoint path.
    
            Arguments:
                metrics: dictionary of metrics to save (None)
                examples: number of examples seen so far (None)
            """
            checkpoint = {
                "state_dict": self.policy.state_dict(),
                "metrics": metrics or {},
                "examples": examples or 0,
            }
            model_name = f"model-{examples}.ckpt" if examples is not None else "model.ckpt"
            torch.save(checkpoint, os.path.join(self.dump_path, model_name))
            if model_name != "model.ckpt":
                curr = os.path.join(self.dump_path, "model.ckpt")
                try:
                    os.remove(curr)
                except FileNotFoundError:
                    pass
                shutil.copyfile(os.path.join(self.dump_path, model_name), curr)
    
        def train(self, train, val=None):
            """
            Trains policy on training data, evaluating on validation data.
            Uses RMSprop with linear warmup/constant LR schedule, as per original paper.
            Support for model checkpointing, logging to Neptune, and gradient accumulation.
            This is preferred over a Pytorch Lightning implementation because
            logging/checkpointing is more convenient, and this is probably a little faster.
    
            Arguments:
                train: training dataloader (type torch.utils.data.DataLoader)
                val: validation dataloader (type torch.utils.data.DataLoader, default None)
            """
    
            def print_info(metrics, key):
                print(
                    f"({key}) "
                    f"epoch: {epoch} | "
                    f"examples: {examples} | "
                    f"step: {step} | "
                    f"acc: {metrics['{}/acc'.format(key)]:.3f} | "
                    f"pos_reward: {metrics['{}/pos_reward'.format(key)]:.3f} | "
                    f"neg_reward: {metrics['{}/neg_reward'.format(key)]:.3f} | "
                    f"elapsed: {time.time() - start_time:.2f}s"
                )
    
            optimizer = torch.optim.RMSprop(
                self.policy.parameters(),
                lr=self.lr["stop"],
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda step: min(1.0, (step + 1) / (self.lr["steps"] + 1))
            )
    
            self.policy.train()
            start_time = time.time()
            last_log = None
            examples = 0
            step = 0
            since_last_val = 0
    
            for epoch in range(1, self.max_epochs + 1):
                train_iter = iter(train)
                while True:
                    try:
                        batch = next(train_iter)
                    except StopIteration:
                        break
    
                    # Evaluation step
                    if val is not None and (
                        since_last_val >= self.eval_every or examples == 0
                    ):
                        self.policy.eval()
                        eval_metrics = defaultdict(list)
    
                        with torch.no_grad():
                            for val_batch in tqdm(val, desc=f"eval @ {examples} examples"):
                                _, batch_metrics = self.loss(val_batch, prefix="eval")
                                for k, v in batch_metrics.items():
                                    eval_metrics[k].append(v)
    
                        eval_metrics = {k: sum(v) / len(v) for k, v in eval_metrics.items()}
                        self.log(eval_metrics, examples=examples)
                        print_info(eval_metrics, "eval")
    
                        if examples > 0:
                            self.save_checkpoint(eval_metrics, examples)
    
                        since_last_val = 0
                        self.policy.train()
    
                    # Compute loss and metrics
                    metrics = defaultdict(list)
                    batch_size = len(batch["positive"])
                    chunk_size = batch_size // self.gradient_accumulation
    
                    for idx in range(self.gradient_accumulation):
                        end = (idx + 1) * chunk_size
                        if batch_size - end < chunk_size:
                            end = batch_size
                        slicer = slice(idx * chunk_size, end)
    
                        microbatch = {k: v[slicer] for k, v in batch.items()}
                        loss, metrics_micro = self.loss(microbatch)
                        (loss / self.gradient_accumulation).backward()
    
                        for k, v in metrics_micro.items():
                            metrics[k].append(v)
    
                    # Update parameters and scheduler
                    norm = torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(), self.grad_norm_clip
                    )
                    metrics["train/norm"] = [norm]
    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
    
                    # Log train metrics periodically
                    if last_log is None or time.time() - last_log > self.min_log_interval:
                        last_log = time.time()
                        metrics = {k: sum(v) / len(v) for k, v in metrics.items()}
    
                        self.log(metrics, examples=examples)
                        print_info(metrics, "train")
    
                    step += 1
                    examples += len(batch["positive"])
                    since_last_val += len(batch["positive"])
    
            print("done training, saving final checkpoint...")
            self.save_checkpoint(examples="final")



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
