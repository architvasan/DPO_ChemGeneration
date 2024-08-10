# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForMaskedLM
from pathlib import Path
import pandas as pd
import sklearn.model_selection
import torch
import wandb
import pdb
from classification_layer import NNModel
from data_utils import CustomDataset
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

def calc_auc(grnd_truth, predictions):
    # Calculate AUC for each class
    auc_scores = []
    grnd_truth = np.array(grnd_truth)
    predictions = np.array(predictions)
    for i in range(grnd_truth.shape[1]):
        auc_score = roc_auc_score(grnd_truth[:, i], predictions[:, i])
        auc_scores.append(auc_score)
    
    # Average AUC scores
    auc_macro = np.mean(auc_scores)

    return auc_macro
    


parser = ArgumentParser()#add_help=False)
parser.add_argument(
    "-d", "--dataset", type=Path, required=True, help="Input data for training/validation"
)

parser.add_argument(
    "-s", "--smilescol", type=str, required=True, help="Column for SMILES"
)

parser.add_argument(
    "-l", "--labelcol", type=str, required=True, help="Column for labels"
)

parser.add_argument(
    "-t", "--testprop", type=float, required=True, help="Proportion of data used for training"
)

parser.add_argument(
    "-E", "--epochs", type=int, required=True, help="Number of epochs"
)


args = parser.parse_args()


wandb.init()

#tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
#LLModel = AutoModelForMaskedLM.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

LLModel = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)

LLModel.to("cuda")
for name, layer in LLModel.named_children():
    print(name, layer)


nnmodel = NNModel(config={"input_size": 768, "embedding_size": 512, "hidden_size": 256, "output_size": 2, "n_layers": 5}).to("cuda")

wandb.watch(nnmodel, log_freq=100)


# import data (local import, change path to your data)
data = pd.read_csv(args.dataset)
#print(data)
#data_pwd = '/nfs/lambda_stor_01/data/avasan/PharmacoData/data/admet_open_data/admet_labelled_data/Solubility'
#data_path: Path = Path(f"{data_pwd}/data.csv")
#data = pd.read_csv(data_path)
#data = data[data[args.labelcol]!=0]
#data['label_scaled'] = (data[args.labelcol] - data[args.labelcol].min()) / (data[args.labelcol].max() - data[args.labelcol].min())     

# split data
X_train, X_test = sklearn.model_selection.train_test_split(data[args.smilescol], test_size=args.testprop, random_state=42)
Y_train, Y_test = sklearn.model_selection.train_test_split(data[args.labelcol], test_size=args.testprop, random_state=42)

# convert pandas dataframe to list for tokenization
X_train = X_train.tolist()
X_test = X_test.tolist()

# convert pandas dataframe to tensor
Y_train = torch.tensor(Y_train.tolist())
Y_test = torch.tensor(Y_test.tolist())

Y_hot_train = torch.zeros(Y_train.size(0), Y_train.max() + 1)
Y_hot_train.scatter_(1, Y_train.unsqueeze(1), 1)

Y_hot_test = torch.zeros(Y_test.size(0), Y_train.max() + 1)
Y_hot_test.scatter_(1, Y_test.unsqueeze(1), 1)

#print(Y_hot_train)
#print(Y_hot_test)

# create CustomDataset object
training_set = CustomDataset(tokenizer, X_train, Y_hot_train, max_input_length=512, max_target_length=512)
test_set = CustomDataset(tokenizer, X_test, Y_hot_test, max_input_length=512, max_target_length=512)

# create dataloader
train_dataloader = torch.utils.data.DataLoader(training_set, batch_size=16, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=16, shuffle=False)

# Initialize optimizer
optimizer = torch.optim.Adam(nnmodel.parameters(), lr=1e-5)

# Timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Training Loop
# Output of the model is a dictionary with keys "loss" and "logits"
def train_one_epoch(epoch_index, criterion):
    LLModel.eval()
    running_loss = 0.0
    total_loss = 0
    num_of_examples: int = 0
    for batch in tqdm(train_dataloader):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        y_regression_values = batch["y_regression_values"]

        with torch.no_grad():
            outputs = LLModel(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)#labels=labels, 
            #print(outputs)
            #loss = outputs["loss"]
            #logits = outputs["logits"]
            encoder = outputs["hidden_states"][-1]

        # train regression head on top of encoder output to label
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # average over second dimension of encoder output to get a single vector for each example
        encoder = encoder.mean(dim=1)

        # pass encoder output to regression head
        nn_outputs = nnmodel(encoder)
        # calculate loss from outputs and ground_truth_y_values
        #nn_loss = F.mse_loss(nn_outputs.flatten(), y_regression_values)
        nn_loss = criterion(nn_outputs, y_regression_values)
        total_loss += nn_loss.item()
        nn_loss.backward()
        optimizer.step()
        running_loss += nn_loss.item()
        if num_of_examples % 10 == 0:
            last_loss = running_loss / 10 # loss per X examples
            print('num_of_examples {} loss: {} %_data_trained : {}'.format(num_of_examples + 1, last_loss, num_of_examples / len(X_train) * 10))
            wandb.log({"num_of_examples": num_of_examples, "train_loss": last_loss})
            running_loss = 0.
        num_of_examples += len(batch["input_ids"])
    return running_loss

def inference_test_set(epoch_index, criterion):
    LLModel.eval()
    running_tloss = 0.0
    total_tloss = 0
    num_of_examples: int = 0
    # dictionary of all ground_truth and predictions
    outputs_dict = {"ground_truth": [], "predictions": []}
    for batch in test_dataloader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        y_regression_values = batch["y_regression_values"]

        with torch.no_grad():
            outputs = LLModel(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)#labels=labels)
            #print(outputs)
            #loss = outputs["loss"]
            #logits = outputs["logits"]
            encoder = outputs["hidden_states"][-1]

            # inference regression head on top of encoder output to label
            # average over second dimension of encoder output to get a single vector for each example
            encoder = encoder.mean(dim=1)

            # pass encoder output to regression head
            nn_outputs = nnmodel(encoder)
            #nn_loss = F.mse_loss(nn_outputs.flatten(), y_regression_values)
            nn_loss = criterion(nn_outputs, y_regression_values)
            # add to dictionary
            outputs_dict["ground_truth"].extend(y_regression_values.cpu().numpy())
            outputs_dict["predictions"].extend(nn_outputs.cpu().detach().numpy())

            total_tloss += nn_loss.item()
            #nn_loss.backward()
            #optimizer.step()
            running_tloss += nn_loss.item()
            if num_of_examples % 10 == 0:
                last_tloss = running_tloss / 100 # loss per X examples
                print('  num_of_examples {} test_loss: {}'.format(num_of_examples + 1, last_tloss))
                wandb.log({"num_of_test_examples": num_of_examples, "test_loss": last_tloss})
                running_tloss = 0.
                # Track best performance, and save the model's state
                num_of_examples += len(batch["input_ids"])
    return outputs_dict

def generate_parity_plot(ground_truth, predictions):
    plt.scatter(ground_truth, predictions, s=0.2)
    # draw line of best fit
    m, b = np.polyfit(ground_truth, predictions, 1)
    plt.plot(ground_truth, [m* g + b for g in ground_truth])#m*ground_truth + b)
    # add labels of correlation coefficient
    # correlation coefficient
    r = np.corrcoef(ground_truth, predictions)[0, 1]
    # pearson's r squared
    r2 = sklearn.metrics.r2_score(ground_truth, predictions)
    plt.legend(["Data", "y = {:.2f}x + {:.2f}; r={}; r2={}".format(m, b, r, r2)], loc="upper left")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.title("Ground Truth vs Predictions")
    plt.savefig("parity_plot.png")


# Training loop
epoch_number = 0
best_auc = 0
early_stop = 20
stop_crit = 0
training_df = {"train_loss":[], "val-auc": []}
import sys
old_stdout = sys.stdout
criterion = nn.CrossEntropyLoss()

log_file = open("logfile.log","w")

for epoch in (range(args.epochs)):
    sys.stdout = log_file
    trainloss = train_one_epoch(epoch, criterion)
    training_df['train_loss'].append(trainloss)
    outputs_dict = inference_test_set(epoch, criterion)
    epoch_number += 1
    ep_auc = calc_auc(outputs_dict["ground_truth"], outputs_dict["predictions"]) #sklearn.metrics.r2_score(outputs_dict["ground_truth"], outputs_dict["predictions"])
    training_df['val-auc'].append(ep_auc)
    wandb.log({"epoch": epoch, "val-auc": ep_auc})

    print(f'Test auc: {ep_auc}')

    if ep_auc>best_auc:
        best_auc = ep_auc
        #best_vloss = last_tloss
        #model_path = 'model_{}'.format(timestamp)
        #model_scripted = torch.jit.script(nnmodel)
        #model_scripted.save(f'model_{timestamp}.pt')
        torch.save(nnmodel.state_dict(), f'model_weights.pt')
        #del(model_scripted)
        #if epoch>0.75*args.epochs:
        #    # Generate Parity Plot
        #    generate_parity_plot(outputs_dict["ground_truth"], outputs_dict["predictions"])
        y_true_test = np.argmax(outputs_dict["ground_truth"], axis=1)
        y_pred_test = np.argmax(outputs_dict["predictions"], axis=1)
        cm = confusion_matrix(y_true_test, y_pred_test)
        print("Confusion Matrix:\n", cm)
    else:
       stop_crit+=1
    if stop_crit>early_stop:
        break

    sys.stdout = old_stdout

log_file.close()

training_pddf = pd.DataFrame(training_df)
training_pddf.to_csv('training_df.csv')
# Generate Parity Plot
#generate_parity_plot(outputs_dict["ground_truth"], outputs_dict["predictions"])
