#!/bin/bash
#SBATCH --job-name=fulle_config
#SBATCH --output=output_%j.txt
#SBATCH --error=errors_%j.txt
#SBATCH --mail-user=simionricc@dei.unipd.it
#SBATCH --mail-type=END
#SBATCH --time=36:00:00               # Elevato a 36 ore per la Config C
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --partition=allgroups
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx:1

set -e
cd $SLURM_SUBMIT_DIR

# Attivazione conda se necessario o uso container
CONFIG=$1
if [ -z "$CONFIG" ]; then
    echo "Errore: specificare la configurazione (A, B o C)."
    exit 1
fi

echo "Esecuzione FullE Ensemble via Singularity - Configurazione: $CONFIG"

SIF_PATH="./ensemble.sif"

# Cartella temporanea ISOLATA per evitare race condition
TEMP_DIR="./global_logits_temp_${CONFIG}"
mkdir -p "$TEMP_DIR"

# Lancio passando il parametro --logits_dir
srun singularity exec --nv \
    --bind $SLURM_SUBMIT_DIR:/workspace \
    --bind /home/simionricc/datasets:/home/simionricc/datasets \
    $SIF_PATH \
    bash -c "cd /workspace && python scripts/train_fulle_pipeline.py --config $CONFIG --logits_dir $TEMP_DIR"

# Spostamento logit nella cartella finale
TARGET_DIR="/home/simionricc/global_logits"
mkdir -p "$TARGET_DIR"

mv $TEMP_DIR/* $TARGET_DIR/
rm -rf $TEMP_DIR

echo "Job FullE $CONFIG terminato con successo!"
