#!/bin/bash
#PBS -N scale_folding
#PBS -l select=1024
#PBS -l walltime=02:00:00
#PBS -q prod
#PBS -A candle_aesp_cnda
#PBS -l filesystems=home:flare
#PBS -o logs/
#PBS -e logs/
#PBS -m abe
#PBS -M avasan@anl.gov

WORKDIR=/flare/FoundEpidem/avasan/IDEAL/INCITE_Scaling
cd ${WORKDIR}

# To ensure GPU affinity mask matches the physical order of the GPUs on the node
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export CPU_AFFINITY="verbose,list:0-7,104-111:8-15,112-119:16-23,120-127:24-31,128-135:32-39,136-143:40-47,144-151:48-55,152-159:56-63,160-167:64-71,168-175:72-79,176-183:80-87,184-191:88-95,192-199"
export CCL_LOG_LEVEL="DEBUG"
export NUMEXPR_MAX_THREADS=208
export ITEX_LIMIT_MEMORY_SIZE_IN_MB=8192
export ITEX_ENABLE_NEXTPLUGGABLE_DEVICE=0
# Set environment variable to print MPICH's process mapping to cores:
export HYDRA_TOPO_DEBUG=1

export RANKS_PER_NODE=12

mpiexec -np $(cat $PBS_NODEFILE | wc -l) -ppn 1 --pmi=pmix hostname > ./helpers/hostnamelist.dat
NUMNODES_TOT="$(cat ./helpers/hostnamelist.dat | wc -l)"
echo $NUMNODES_TOT
module load frameworks
source /lus/flare/projects/FoundEpidem/avasan/envs/peptide_des_venv/bin/activate
mpiexec -np $NUMNODES_TOT -ppn 1 cp -r chai_weights_downloaded /dev/shm/ 
#node_list=(1 2)
node_list=(1024)
export CHAI_DOWNLOADS_DIR=/dev/shm/chai_weights_downloaded
mkdir ./Folding/aa_1000
for n in "${node_list[@]}"; do
    echo "Value: $n"
    TOTAL_NUMBER_OF_RANKS=$((n * RANKS_PER_NODE))
    mpirun -n $TOTAL_NUMBER_OF_RANKS -ppn 12 \
        ./helpers/interposer.sh \
        ./helpers/set_ze_mask.sh \
        python -m incite_bench.fold \
        -s input_data/whsc1_seq.txt \
        -o /dev/shm/out_test_${n} \
        -t 2 \
        -N ${n} \
        -tf Folding/aa_1000/folding_times_new \
        > logs/folds_1k_${n}.log 2> logs/folds_1k_${n}.err
done
