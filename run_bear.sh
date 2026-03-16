#!/bin/bash

#SBATCH --account=baberc-human-agent-teaming
#SBATCH --qos=bbdefault

# SBATCH --time 0-00:45:00  # days-hours:minutes:seconds
# SBATCH --nodes 1
# SBATCH --ntasks 16

module purge
module load bluebear
module load bear-apps/2018b
module load Python/3.6.6-foss-2018b

python -c "print('hello world')"