module load frameworks/2024.1
conda activate /lus/gila/projects/candle_aesp_CNDA/avasan/envs/rl_dpo 
python run_loop.py -c configs/config.BBBP.json > logs/BBBP.log 2> logs/BBBP.err &
python run_loop.py -c configs/config.CYP450_2C19_substrate.json > logs/2C19.log 2> logs/2C19.err &
python run_loop.py -c configs/config.CYP450_3A4_substrate.json > logs/3A4.log 2> logs/3A4.err &    
python run_loop.py -c configs/config.P-glycoprotein_substrate.json > logs/PGP-sub.log 2> logs/PGP-sub.err &
python run_loop.py -c configs/config.Caco-2_permeability.json > logs/Caco.log 2> logs/Caco.err & 
python run_loop.py -c configs/config.CYP450_2C9_substrate.json > logs/2C9.log 2> logs/2C9.err & 
python run_loop.py -c configs/config.CYP450_1A2_substrate.json > logs/1A2.log 2> logs/1A2.err &
python run_loop.py -c configs/config.CYP450_2D6_substrate.json > logs/2D6.log 2> logs/2D6.err &
python run_loop.py -c configs/config.P-glycoprotein_inhibitor.json > logs/PGP-inhib.log 2> logs/PGP-inhib.err
