#!/usr/bin/env bash
# Site-neutral validation template.  Add account/partition/QoS directives in
# a copied template and pass it with --slurm-template when the cluster needs
# them.  Slurm exports the activated Vitriflow environment by default.
#SBATCH --job-name={{JOB_NAME}}
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -Eeuo pipefail
cd -- {{TASK_DIR}}
{{EXECUTE_CMD}}
