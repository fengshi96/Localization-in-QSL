#!/bin/bash
Hmin=0.00
Hmax=0.18

for hin in $(seq $Hmin 0.01 $Hmax)
do
	mkdir -p hin_$hin
	cd hin_$hin
	
	mydir=$(pwd)
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
#SBATCH --job-name hin_$hin                   # descriptive name shown in queue and used for output files
#SBATCH --output %x.%j.out                 # this is where the (text) output goes. %x=Job name, %j=Jobd id, %N=node.
#SBATCH --mail-type=FAIL                  # uncomment to ask for email notification.
#SBATCH --mail-user=shi.feng@tum.de      # email to send to. Defaults to your personal ga12abc@mytum.de address
#SBATCH --get-user-env


set -e  # abort the whole script if one command fails

export OMP_NUM_THREADS=\$SLURM_CPUS_ON_NODE  # number of CPUs per node, total for all the tasks below.
echo "OMP_NUM_THREADS=\$OMP_NUM_THREADS"
export PYTHONPATH="/space/go76jez/miniconda3/bin/python"


echo "starting job on \$(hostname) at \$(date) with \$SLURM_CPUS_ON_NODE cores"
time
python ../gs.py $hin -$hin 0 &> gs.out
time

EOF
)

	echo "$rawjob" &> jobGS.bat
 	sbatch jobGS.bat
	echo "hin=$hin is submitted"
	cd ..
done
