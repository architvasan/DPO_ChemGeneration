# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForMaskedLM
from pathlib import Path
import pandas as pd
import sklearn.model_selection
import torch
import wandb
import pdb
#from classification_layer import NNModel
#from data_utils import CustomDataset
import torch.nn.functional as F
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModel, AutoTokenizer
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path
from tqdm import tqdm
from torch import nn

from sklearn.metrics import roc_auc_score

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

LLModel = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True)

total_params = sum(
	param.numel() for param in LLModel.parameters()
)

print(total_params)
