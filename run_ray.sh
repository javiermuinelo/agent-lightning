#!/bin/bash -l
#SBATCH --partition=dev_accelerated
#SBATCH --nodes=2
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1        # use 1 task-per-node for clarity
#SBATCH --cpus-per-task=32
#SBATCH --time=00:20:00
#SBATCH --job-name=ag-w-rl
#SBATCH --output=logs/slurm_llm_%j.out
#SBATCH --error=logs/slurm_llm_%j.err

module add compiler/gnu/12 mpi/openmpi devel/cuda
source /home/hk-project-p0022560/tum_aho7196/miniconda3/bin/activate agent-lightning

export PYTHONUNBUFFERED=1
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=true

CHECKPOINT_DIR=/home/hk-project-p0022560/tum_aho7196/agent-lightning/examples/workflow_agent/checkpoints
LOG_DIR=/home/hk-project-p0022560/tum_aho7196/agent-lightning/examples/workflow_agent/logs
mkdir -p $CHECKPOINT_DIR $LOG_DIR

# Getting the node names
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
head=$(echo $nodes | awk '{print $1}')
worker=$(echo $nodes | awk '{print $2}')

echo "Head: $head"
echo "Worker: $worker"

head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head" hostname --ip-address)


echo "Head node: $head"
port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Head node: $head"
echo "Worker node: $worker"
echo "SLURM_GPUS_PER_NODE: $SLURM_GPUS_PER_NODE"

echo "------------------------------------------------"

# Start Ray head in a detached background subshell that runs `--block`
srun -N1 -n1 -w $head bash -c "
  ray start --head --node-ip-address=$head --port=6379 \
    --dashboard-host=0.0.0.0 --dashboard-port=8265 --num-cpus=32 --num-gpus=1 --block
" &

sleep 15  # give head time to boot

# Start worker node (normal srun)
srun -N1 -n1 -w $worker bash -c "
  echo 'Starting Ray worker on $worker...'
  ray start --address=$ip_head --num-cpus=32 --num-gpus=1 
"

# Verify Ray cluster from head node
echo "------------------------------------------------"
echo "Running Ray verification from head node..."
srun --overlap --nodes=1 --ntasks=1 -w $head --output=/dev/stdout --error=/dev/stderr python - <<'PY'
import ray, socket, time
time.sleep(3)
ray.init(address="auto")
print("Connected to Ray cluster successfully!")
print(ray.nodes())
@ray.remote
def f(x): return (x, socket.gethostname())
print("Sample:", ray.get([f.remote(i) for i in range(4)]))
PY
echo "------------------------------------------------"

echo "Cluster started successfully. Keeping job alive..."
sleep infinity
