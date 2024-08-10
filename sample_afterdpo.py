import argparse
import pandas as pd
from pref_opt_for_mols.models import GPT, CharRNN
import torch
import json
import os
import intel_extension_for_pytorch as ipex

parser = argparse.ArgumentParser()
parser.add_argument(
    "--arch",
    type=str,
    default="rnn",
    help="which model class to use, either 'gpt' or 'rnn'",
)
parser.add_argument(
    "--model_path",
    required=True,
    type=str,
)

parser.add_argument(
    "--inp_smi",
    required=False,
    type=str,
)

parser.add_argument(
    "--num_batches",
    default=1,
    type=int,
)
parser.add_argument(
    "--batch_size",
    default=128,
    type=int,
)
parser.add_argument(
    "--device",
    default=0,
    type=int,
)
parser.add_argument(
    "--out",
    required=True,
    type=str,
)
args = parser.parse_args()

if __name__ == "__main__":
    # Load model directly

    if args.arch not in {"molgen", "chemgpt"}:
        with open(os.path.join(args.model_path, "config.json")) as f:
            config = json.load(f)

    if args.arch == "molgen":
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import selfies as sf
        tokenizer = AutoTokenizer.from_pretrained("zjunlp/MolGen-large")
        model = AutoModelForSeq2SeqLM.from_pretrained("zjunlp/MolGen-large")
        #sf_input = tokenizer("[C][=C][C][=C][C][=C][Ring1][=Branch1]", return_tensors="pt")
    
    if args.arch == "chemgpt":
        # Load model directly
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import selfies as sf
        tokenizer = AutoTokenizer.from_pretrained("ncfrey/ChemGPT-1.2B")
        model = AutoModelForCausalLM.from_pretrained("ncfrey/ChemGPT-1.2B")
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.eval()
        #sf_input = tokenizer("[C][=C][C][=C]", return_tensors="pt")

    elif args.arch == "gpt":
        # when we load we make sure we first map the models to cpu,
        # and then transfer over to the desired device
        model = GPT.load_from_checkpoint(
            config,
            os.path.join(args.model_path, "model.ckpt"),
            device=args.device,
            disable_dropout=True,
        )

    elif args.arch == "rnn":
        model = CharRNN.load_from_checkpoint(
            config,
            os.path.join(args.model_path, "model.ckpt"),
            device=args.device,
            disable_dropout=True,
        )
    else:
        raise ValueError(f"Unrecognized model {args.arch}")

    device = torch.device(f"xpu:{args.device}")
    model.to(device)

    sampled_smiles = []
    if args.arch in {"molgen", "chemgpt"}:
        smi_df = pd.read_csv(args.inp_smi)['smiles']
        for smi in smi_df[:args.num_batches]:#range(args.num_batches):
            try:
                print(sf.encoder(smi))
                sf_input = tokenizer(sf.encoder(smi), return_tensors="pt")
                
                generated = model.generate(input_ids=sf_input["input_ids"].to(device),
                                            attention_mask=sf_input["attention_mask"].to(device),
                                            do_sample=False,
                                            max_length=32,
                                            min_length=5,
                                            num_return_sequences =args.batch_size,
                                            num_beams = args.batch_size,)
                                            #temperature = 1)#n_batch=args.batch_size)
                 
                sf_output = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True).replace(" ","") for g in generated]
                curr_smiles = [sf.decoder(sf_it) for sf_it in sf_output]

                sampled_smiles.extend(curr_smiles)

                print(
                    f"{len(sampled_smiles)}/{args.batch_size*args.num_batches} SMILES sampled"
                )
       
            except:
                continue
    else:
        for j in range(args.num_batches):
            curr_smiles = model.sample(n_batch=args.batch_size)
            sampled_smiles.extend(curr_smiles)
            print(
                f"{len(sampled_smiles)}/{args.batch_size*args.num_batches} SMILES sampled"
            )
    df = pd.DataFrame({"smiles": sampled_smiles})
    df = df.drop_duplicates(subset=["smiles"])
    df.to_csv(args.out)#, index=False)
