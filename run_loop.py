from dpo_loop.dpo_loop import *
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

try:
    evaluator_type = config["evaluator_type"]
except:
    evaluator_type = 'molformer'

try:
    evaluator_config = config["evaluator_config"]
except:
    evaluator_config = 'none'

dpo_loop_main(config["data"], 
              config["smiles_col"],
              config["label_col"],
              config["pos_label"],
              config["neg_label"],
              config["scaffolds_sampling"],
              config["batch_sample"],
              config["out_dir"],
              config["out_pattern"],
              config["evaluator_checkpoint"],
              config["finetune_eps"],
              config["loop_number"],
              config["goal_ratio"],
              evaluator_type,
              evaluator_config,
              config["device"])

