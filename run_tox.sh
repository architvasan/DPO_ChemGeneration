module load frameworks/2024.1
conda activate /lus/gila/projects/candle_aesp_CNDA/avasan/envs/rl_dpo 
python run_loop.py -c configs/config.hepa.json > logs/hepa.log 2> logs/hepa.err &
python run_loop.py -c configs/config.resp.json > logs/resp.log 2> logs/resp.err &
python run_loop.py -c configs/config.cardio5_inhib.json > logs/cardio5.log 2> logs/cardio5.err &
python run_loop.py -c configs/config.cardio10_inhib.json > logs/cardio10.log 2> logs/cardio10.err &
python run_loop.py -c configs/config.CYP3A4_inhib.json > logs/CYP3A4_inhib.log 2> logs/CYP3A4_inhib.err &
python run_loop.py -c configs/config.CYP2C9_inhib.json > logs/CYP2C9_inhib.log 2> logs/CYP2C9_inhib.err &
python run_loop.py -c configs/config.CYP2C19_inhib.json > logs/CYP2C19_inhib.log 2> logs/CYP2C19_inhib.err &
python run_loop.py -c configs/config.CYP21A2_inhib.json > logs/CYP21A2_inhib.log 2> logs/CYP21A2_inhib.err

