module load frameworks/2024.1
conda activate /lus/gila/projects/candle_aesp_CNDA/avasan/envs/rl_dpo 
python run_rawsampling.py -c configs/config.BBBP.json > logs/BBBP_raw.log 2> logs/BBBP_raw.err &
python run_rawsampling.py -c configs/config.CYP450_2C19_substrate.json > logs/2C19_raw.log 2> logs/2C19_raw.err &
python run_rawsampling.py -c configs/config.CYP450_3A4_substrate.json > logs/3A4_raw.log 2> logs/3A4_raw.err &    
python run_rawsampling.py -c configs/config.P-glycoprotein_substrate.json > logs/PGP-sub_raw.log 2> logs/PGP-sub_raw.err &
python run_rawsampling.py -c configs/config.Caco-2_permeability.json > logs/Caco_raw.log 2> logs/Caco_raw.err & 
python run_rawsampling.py -c configs/config.CYP450_2C9_substrate.json > logs/2C9_raw.log 2> logs/2C9_raw.err & 
python run_rawsampling.py -c configs/config.CYP450_1A2_substrate.json > logs/1A2_raw.log 2> logs/1A2_raw.err &
python run_rawsampling.py -c configs/config.CYP450_2D6_substrate.json > logs/2D6_raw.log 2> logs/2D6_raw.err &
python run_rawsampling.py -c configs/config.P-glycoprotein_inhibitor.json > logs/PGP-inhib_raw.log 2> logs/PGP-inhib_raw.err
