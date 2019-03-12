#!/bin/bash
module load python_gpu/3.6.1 
# 4 cores, .25x4=1 GPU, 4x4096Mb RAM, 6 hrs waiting time
#bsub -n 4 -R "rusage[gpu=0.25]" -R "rusage[mem=4096]" -W 6:00 python /cluster/home/acharyad/NLU18/train_rnn.py
#bsub -R gpu -W 3:00 python /cluster/home/schneech/NLU18/train_rnn.py
bsub -R "rusage[ngpus_excl_p=1]" -R "rusage[mem=4096]" -W 4:00 python /cluster/home/ltran/AML18/Task1/main.py