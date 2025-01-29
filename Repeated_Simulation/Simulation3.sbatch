#!/bin/bash
#SBATCH --job-name sim3_1      # Set a name for your job. This is especially useful if you have multiple jobs queued.
#SBATCH --partition medium         # Slurm partition to use
#SBATCH --time 6-11:10             # Wall time limit in D-HH:MM
#SBATCH --nodes=1
#SBATCH --exclusive
#1 task per node is supposed to ensure that each ray worker runtime will be on a separate node:
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --mem-per-cpu=5000           # Memory limit for each tasks (in MB)
#SBATCH -o out/sim3.out        # File to which STDOUT will be written
#SBATCH -e out/sim3.err        # File to which STDERR will be written
#SBATCH --mail-type=ALL            # Type of email notification- NONE,BEGIN,END,FAIL,ALL
#SBATCH --mail-user=yikun@uw.edu  # Email to which notifications will be sent

module load Python
source ../yikun_venv/bin/activate


# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --partition=${SLURM_JOB_PARTITION} --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
head_node_ip=${ADDR[1]}
else
head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
srun --partition=${SLURM_JOB_PARTITION} --nodes=1 --ntasks=1 -w "$head_node" \
ray start --head --node-ip-address="$head_node_ip" --port=$port \
--num-cpus "${SLURM_CPUS_PER_TASK}" --block &

sleep 3

worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
node_i=${nodes_array[$i]}
echo "Starting WORKER $i at $node_i"
srun --partition=${SLURM_JOB_PARTITION}--nodes=1 --ntasks=1 -w "$node_i" \
ray start --address "$ip_head" \
--num-cpus "${SLURM_CPUS_PER_TASK}" --block &
sleep 5
done


python ./Simulation3.py "$SLURM_CPUS_PER_TASK"

