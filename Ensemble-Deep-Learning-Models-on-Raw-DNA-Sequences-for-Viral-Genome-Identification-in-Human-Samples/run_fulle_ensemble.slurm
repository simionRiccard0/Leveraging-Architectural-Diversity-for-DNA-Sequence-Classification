#!/bin/bash
#SBATCH --job-name=fulle_ensemble
#SBATCH --output=output_%j.txt
#SBATCH --error=errors_%j.txt
#SBATCH --mail-user=simionricc@dei.unipd.it
#SBATCH --mail-type=END
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --partition=allgroups
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx:1
set -e
cd $SLURM_SUBMIT_DIR
CONFIG=$1
if [ -z "$CONFIG" ]; then
    echo "Errore: specificare la configurazione (A, B o C). Esempio: sbatch run_fulle_ensemble.slurm A"
    exit 1
fi
echo "Esecuzione FullE ENSEMBLE COMPLETO (10 reti) via Singularity - Configurazione: $CONFIG"
# Percorso dell'immagine Singularity caricata
SIF_PATH="./ensemble.sif"
# Crea la cartella temporanea locale per il container
mkdir -p ./global_logits_temp
# Lancio dell'ambiente isolato Singularity con l'orchestrator completo (10 modelli x Config)
# NOTA: --viraminer_folder esplicito per chiarezza, anche se coincide con il
# default dello script python (il path reale confermato è
# <repo_root>/dataset/fullset_train.csv, NON scripts/dataset/).
srun singularity exec --nv \
    --bind $SLURM_SUBMIT_DIR:/workspace \
    --bind /home/simionricc/datasets:/home/simionricc/datasets \
    $SIF_PATH \
    bash -c "cd /workspace && python scripts/train_fulle_pipeline.py --config $CONFIG --viraminer_folder dataset --data_folder /home/simionricc/datasets"
# Fuori dal container: copia i file TXT direttamente nella directory globale piatta
echo "Spostamento logit estratti nella directory globale..."
TARGET_DIR="/home/simionricc/global_logits"
mkdir -p $TARGET_DIR
# Muove i file .txt e pulisce la directory temporanea
mv ./global_logits_temp/* $TARGET_DIR/
rm -rf ./global_logits_temp
echo "Job FullE Ensemble Completo ($CONFIG) terminato correttamente. File TXT pronti e spazio pulito."
