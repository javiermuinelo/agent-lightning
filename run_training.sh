#!/bin/bash -l
#SBATCH --partition=dev_accelerated
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1        # use 1 task-per-node for clarity
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --job-name=ag-w-rl
#SBATCH --output=logs/slurm_llm_%j.out
#SBATCH --error=logs/slurm_llm_%j.err

module add compiler/gnu/12 mpi/openmpi devel/cuda
source /home/hk-project-p0022560/tum_aho7196/miniconda3/bin/activate agent-lightning

export PYTHONUNBUFFERED=1
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=true
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=1 # required by agentlightning (otherwise conflict between dependencies)

CHECKPOINT_DIR=/home/hk-project-p0022560/tum_aho7196/agent-lightning/examples/workflow_agent/checkpoints
LOG_DIR=/home/hk-project-p0022560/tum_aho7196/agent-lightning/examples/workflow_agent/logs
mkdir -p $CHECKPOINT_DIR $LOG_DIR

# Get the list of nodes
# Getting the node names
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
head=$(echo $nodes | awk '{print $1}')
worker=$(echo $nodes | awk '{print $2}')


echo "Head: $head"
echo "Worker: $worker"

head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head" hostname --ip-address)

port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"
echo "------------------------------------------------"

# Start Ray head on the head node
srun -N1 -n1 -w $head bash -c "
  unset ROCR_VISIBLE_DEVICES
  unset HIP_VISIBLE_DEVICES
  export CUDA_VISIBLE_DEVICES=0,1,2,3
  export VLLM_USE_V1=1
  ray start --head --node-ip-address=$head --port=6379 \
    --dashboard-host=0.0.0.0 --dashboard-port=8265 --num-cpus=32 --num-gpus=4 --block
" &

sleep 10  # give head time to boot


# Start worker node (normal srun)
srun -N1 -n1 -w $worker bash -c "
  unset ROCR_VISIBLE_DEVICES
  unset HIP_VISIBLE_DEVICES
  export CUDA_VISIBLE_DEVICES=0,1,2,3
  export VLLM_USE_V1=1
  echo 'Starting Ray worker on $worker...'
  ray start --address=$ip_head --num-cpus=32 --num-gpus=4 
"

sleep 10

# ------------------ TRAINING PHASE ------------------

echo "------------------------------------------------"

echo "Starting client workflow on client node..."
srun --overlap --nodes=1 --ntasks=1  -w $worker \
    bash -c "VERL_API_BASE='http://${head_node_ip}:9999/' python -u examples/workflow_agent/workflow_agent.py" &


sleep 20

echo "Starting server training on head node..."
srun --overlap --nodes=1 --ntasks=1 -w $head \
    /home/hk-project-p0022560/tum_aho7196/agent-lightning/examples/workflow_agent/train.sh &

sleep 10
wait

