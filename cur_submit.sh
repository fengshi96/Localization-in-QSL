#!/bin/bash
Lx=12
Ly=5

Hmin=0.00
Hmax=0.18

chi=300

for h in $(seq $Hmin 0.01 $Hmax)
do
	mkdir -p h_$h
	cd h_$h
	
	rawjob=$(cat <<EOF
#!/bin/bash

# hardware requirements
#SBATCH --time=23:00:00                    # enter a maximum runtime for the job. (format: DD-HH:MM:SS, or just HH:MM:SS)
#SBATCH --cpus-per-task=8                  # use multi-threading with 4 cpu threads (= 2 physical cores * hyperthreading)
#SBATCH --mem=24G                           # request this amount of memory (total per node)

#SBATCH --partition=cpu                    # optional, cpu is default. needed for gpu/classes.
# #SBATCH --qos=debug                        # Submit debug job for quick test
# #SBATCH --nodelist=curie42

# some further useful options, uncomment as needed/desired
#SBATCH --job-name cur_h$h                  # descriptive name shown in queue and used for output files
#SBATCH --output %x.%j.out                 # this is where the (text) output goes. %x=Job name, %j=Jobd id, %N=node.
#SBATCH --mail-type=FAIL                  # uncomment to ask for email notification.
#SBATCH --mail-user=shi.feng@tum.de      # email to send to. Defaults to your personal ga12abc@mytum.de address
#SBATCH --get-user-env


set -e  # abort the whole script if one command fails

export OMP_NUM_THREADS=\$SLURM_CPUS_ON_NODE  
echo "OMP_NUM_THREADS=\$OMP_NUM_THREADS"
export MKL_NUM_THREADS=\$SLURM_CPUS_ON_NODE  
echo "MKL_NUM_THREADS=\$MKL_NUM_THREADS"
export PYTHONPATH="/space/go76jez/miniconda3/bin/python"


echo "starting job on \$(hostname) at \$(date) with \$SLURM_CPUS_ON_NODE cores"
time python ../current_dynamics.py $(pwd)/ground_states/GS_L${Lx}${Ly}Cstylechi${chi}_K-1.00Fx${h}Fy${h}Fz${h}W0.00EpFalse.h5 &> curDyn.out


EOF
)

	echo "$rawjob" &> jobCurDyn.bat
 	sbatch jobCurDyn.bat
	echo "Current Dynamics for h=$h is submitted"
	cd ..
done
