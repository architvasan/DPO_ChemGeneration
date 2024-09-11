module load frameworks/2024.1
conda activate /lus/gila/projects/candle_aesp_CNDA/avasan/envs/rl_dpo 
python run_loop.py -c configs/config.cardio5_inhib.json > logs/cardio5.log 2> logs/cardio5.err &
python run_loop.py -c configs/config.cardio10_inhib.json > logs/cardio10.log 2> logs/cardio10.err &

