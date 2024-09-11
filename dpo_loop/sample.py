import selfies as sf
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import intel_extension_for_pytorch as ipex
import pandas as pd
from tqdm import tqdm

def sample(checkpoint_fil, promptsmi_file, smi_col, output, batch_size, device=0):
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
    completed = 0
    writing_smiles = []
    for it, smi in tqdm(enumerate(smi_df)):#range(args.num_batches):
        try:
            sf_input = tokenizer(sf.encoder(smi), return_tensors="pt", truncation=True, max_length=25)
            generated = model.generate(input_ids=sf_input["input_ids"].to(device),
                                    attention_mask=sf_input["attention_mask"].to(device),
                                    do_sample=False,
                                    max_length=32,
                                    min_length=5,
                                    num_return_sequences = batch_size,
                                    num_beams = batch_size,)


            sf_output = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True).replace(" ","") for g in generated]
            curr_smiles = [sf.decoder(sf_it) for sf_it in sf_output]
            print(f"{output}/{it}.smi")
            if completed%100==0 and completed !=0:
                print("Writing step")
                dict_curr_smiles = pd.DataFrame({"SMILES": writing_smiles, "it": it})
                
                dict_curr_smiles.to_csv(f"{output}/{it}.smi", index=False)
                writing_smiles = []

            sampled_smiles.extend(curr_smiles)
            writing_smiles.extend(curr_smiles)
            completed+=1

            for rand_it in tqdm(range(2)):
                sf_rand_its = torch.randperm(sf_input['input_ids'].size(1))
                #sf_rand_its = torch.randperm(sf_input.size(1))
                sf_input_rand = sf_input['input_ids'][:,sf_rand_its]#[:20]
                sf_input_rand_att = sf_input['attention_mask'][:,sf_rand_its]#[:20]
                try:
                    sf_input_rand = sf_input_rand[:25]
                    sf_input_rand_att = sf_input_rand_att[:25]
                except:
                    pass
                #sf_input_rand = sf_input[:,sf_rand_its][:20]
                generated = model.generate(input_ids=sf_input_rand.to(device),
                                        attention_mask=sf_input_rand_att.to(device),
                                        do_sample=False,
                                        max_length=32,
                                        min_length=5,
                                        num_return_sequences = batch_size,
                                        num_beams = batch_size,)

                sf_output = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True).replace(" ","") for g in generated]
                curr_smiles = [sf.decoder(sf_it) for sf_it in sf_output]
                print(f"{output}/{it}.smi")
                if completed%100==0 and completed !=0:
                    print("Writing step")
                    dict_curr_smiles = pd.DataFrame({"SMILES": writing_smiles, "it": it})
                    
                    dict_curr_smiles.to_csv(f"{output}/{it}.smi", index=False)
                    writing_smiles = []
                                                                                                                                                  
                sampled_smiles.extend(curr_smiles)
                writing_smiles.extend(curr_smiles)
                completed+=1

        except:
            print(f"{it}: error with generation")
            continue

    return sampled_smiles
