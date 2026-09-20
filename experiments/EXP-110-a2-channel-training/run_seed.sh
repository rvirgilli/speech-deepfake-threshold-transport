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
# Paths are environment overrides so the recipe is readable and runnable outside the
# authors' machine. The defaults are the layout the reported runs used.
ARTIFACTS=${A2_ARTIFACTS:-$HOME/exp-artifacts/icassp2027}
RECIPE=${A2_RECIPE:-$HOME/icassp-runs/EXP-401-areaII-stage0}
VENV=${A2_VENV:-$HOME/projects/voxtech/deepfake-model-assessment}
RUN=$ARTIFACTS/EXP-110/$NAME
CODE=$RECIPE/code
PROTO=$RECIPE/data/
EXP110=$(cd "$(dirname "$0")" && pwd)
MAN=$ARTIFACTS/EXP-001/manifests/asv21la.csv
case "$ARM" in
  arm1) DB=${A2_DATA:-$HOME/data/corpora/anti-spoofing}/ASVspoof2019/LA/ ;;          # EXP-401 recipe, original 19LA
  arm2) DB=$ARTIFACTS/EXP-110/arm2_db/ ;;            # channel-matched train, original dev
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
  uv run --project $VENV \
    --with librosa --with tensorboardX --with pandas python "$EXP110/train_cooperative.py" \
    --database_path "$DB" --protocols_path "$PROTO" \
    --num_epochs 30 --batch_size 14 --lr 0.000001 --seed "$SEED" --comment "$TAG"
fi
test -f "$MODELS/epoch_29.pth" || { echo "training incomplete: epoch_29.pth missing"; exit 2; }

uv run --project $VENV \
  --with librosa --with tensorboardX --with pandas python "$EXP110/score_epochs_arm.py" \
  --models_dir "$MODELS"

TIED=$(uv run --project "$EXP110" python "$EXP110/pick_tied.py" --results "$RUN/results_stage0.json")
echo "$NAME tied epochs: $TIED"
for e in $TIED; do
  uv run --project $VENV python "$EXP110/score_ckpt.py" \
    --checkpoint "$MODELS/epoch_$e.pth" --manifest "$MAN" --out "$RUN/scores/epoch_$e" --batch-size 24
done
echo "$NAME scoring complete"
