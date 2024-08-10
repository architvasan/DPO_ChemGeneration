#my_array=("HIA" "PPB")
my_array=($(ls -d */))
export CUDA_VISIBLE_DEVICES=0
# Iterate over the array elements

for task in "${my_array[@]}"; do
    if [ "$task" == "Scripts" ]
        then
            continue
    fi
    
    echo "$task start"
    cd $task
    cp ../Scripts/*py . 
    python run_script.py -d /lambda_stor/data/avasan/PharmacoData/data/admet_open_data/admet_labelled_data/${task}/data.csv -s SMILES -l label -t 0.2 -E 100 > run.log 2> run.err
    cd ..
    echo "$task done"
done
