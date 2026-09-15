#!/bin/bash
# EXP-110 seed factorial: one run = train 30 epochs on the arm's training set with the
# given seed, score every epoch on the original 19LA dev, pick the five dev-tied
# checkpoints, score them on 21LA. One job, one exit code (repository rule 14); every
# step is individually resumable, so resubmitting the identical command continues.
# Usage: run_seed.sh <arm1|arm2> <seed>
set -euo pipefail
trap ':' USR1
ARM=$1; SEED=$2
NAME="${ARM}_s${SEED}"
RUN=/home/rv/exp-artifacts/icassp2027/EXP-110/$NAME
CODE=/home/rv/icassp-runs/EXP-401-areaII-stage0/code
PROTO=/home/rv/icassp-runs/EXP-401-areaII-stage0/data/
EXP110=/home/rv/projects/academic/icassp2027/experiments/EXP-110-a2-channel-training
MAN=/home/rv/exp-artifacts/icassp2027/EXP-001/manifests/asv21la.csv
case "$ARM" in
  arm1) DB=/home/rv/data/corpora/anti-spoofing/ASVspoof2019/LA/ ;;          # EXP-401 recipe, original 19LA
  arm2) DB=/home/rv/exp-artifacts/icassp2027/EXP-110/arm2_db/ ;;            # channel-matched train, original dev
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
TAG="exp110${ARM}s${SEED}"
MODELS="$CODE/models/model_LA_weighted_CCE_30_14_1e-06_$TAG"

export EXP110_RUN_DIR="$RUN"
export EXP110_MODELS_DIR="$MODELS"
export EXP110_CODE_DIR="$CODE"
mkdir -p "$RUN"
cd "$CODE"

if [ ! -f "$MODELS/epoch_29.pth" ]; then
  uv run --project /home/rv/projects/voxtech/deepfake-model-assessment \
    --with librosa --with tensorboardX --with pandas python "$EXP110/train_cooperative.py" \
    --database_path "$DB" --protocols_path "$PROTO" \
    --num_epochs 30 --batch_size 14 --lr 0.000001 --seed "$SEED" --comment "$TAG"
fi
test -f "$MODELS/epoch_29.pth" || { echo "training incomplete: epoch_29.pth missing"; exit 2; }

uv run --project /home/rv/projects/voxtech/deepfake-model-assessment \
  --with librosa --with tensorboardX --with pandas python "$EXP110/score_epochs_arm.py" \
  --models_dir "$MODELS"

TIED=$(uv run --project "$EXP110" python "$EXP110/pick_tied.py" --results "$RUN/results_stage0.json")
echo "$NAME tied epochs: $TIED"
for e in $TIED; do
  uv run --project /home/rv/projects/voxtech/deepfake-model-assessment python "$EXP110/score_ckpt.py" \
    --checkpoint "$MODELS/epoch_$e.pth" --manifest "$MAN" --out "$RUN/scores/epoch_$e" --batch-size 24
done
echo "$NAME scoring complete"
