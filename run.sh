#!/bin/sh

#SBATCH -J  extract         # Job name
#SBATCH -o  ./out/extract.%j.out   # Name of stdout output file (%j expands to %jobId)
#SBATCH -p 2080ti           # queue  name  or  partiton name
#SBATCH -t 72:00:00               # Run time (hh:mm:ss) - 1.5 hours

#### Select  GPU
#SBATCH   --gres=gpu:1

## 노드 지정하지않기
#SBATCH   --nodes=1

cd  $SLURM_SUBMIT_DIR

echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "CUDA_HOME=$CUDA_HOME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_VERSION=$CUDA_VERSION"

## path  Erase because of the crash
module  purge
#module  load  postech

echo "Start"
echo "conda PATH "

echo "source  $HOME/anaconda3/etc/profile.d/conda.sh"
source  $HOME/anaconda3/etc/profile.d/conda.sh

echo "conda activate QA "
conda activate QA

export PYTHONPATH=.


python  $HOME/bert-extract-qa/main.py --dataset_name mrqa --batch_size 4

echo " conda deactivate QA "

conda deactivate QA

squeue  --job  $SLURM_JOBID

echo  "##### END #####"
