import selfies as sf
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import intel_extension_for_pytorch as ipex
import pandas as pd

def sample(checkpoint_fil, promptsmi_file, smi_col, batch_size, device=0):
    tokenizer = AutoTokenizer.from_pretrained("ncfrey/ChemGPT-1.2B")
    model = AutoModelForCausalLM.from_pretrained("ncfrey/ChemGPT-1.2B")
    try:
        checkpoint = torch.load(checkpoint_fil)
        model.load_state_dict(checkpoint['model_state_dict'])
    except:
        pass
    model.eval()
    device = torch.device(f"xpu:{device}")
    model.to(device)

    smi_df = pd.read_csv(promptsmi_file)[smi_col]
    sampled_smiles = []
    for smi in smi_df:#range(args.num_batches):
        try:
            sf_input = tokenizer(sf.encoder(smi), return_tensors="pt")
            generated = model.generate(input_ids=sf_input["input_ids"].to(device),
                                        attention_mask=sf_input["attention_mask"].to(device),
                                        do_sample=False,
                                        max_length=32,
                                        min_length=5,
                                        num_return_sequences = batch_size,
                                        num_beams = batch_size,)

            sf_output = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True).replace(" ","") for g in generated]
            curr_smiles = [sf.decoder(sf_it) for sf_it in sf_output]
            sampled_smiles.extend(curr_smiles)
        except:
            continue

    return sampled_smiles
