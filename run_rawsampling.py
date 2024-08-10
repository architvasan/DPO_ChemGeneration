from dpo_loop.raw_sampling_eval import *
import json
##########################################################################
#------------------------------------------------------------------------
##########################################################################
def ParamsJson(json_file):
    with open(json_file) as f:
       params = json.load(f)
    return params


parser = argparse.ArgumentParser()
parser.add_argument(
    "-c",
    "--config",
    required=True,
    type=str,
)

args = parser.parse_args()

config = ParamsJson(args.config)

raw_sampling(config["data"], 
              config["smiles_col"],
              config["label_col"],
              config["pos_label"],
              config["neg_label"],
              config["scaffolds_sampling"],
              config["batch_sample"],
              config["out_dir"],
              config["out_pattern"],
              config["evaluator_checkpoint"],
              config["loop_number"],
              config["goal_ratio"],
              config["device"])

