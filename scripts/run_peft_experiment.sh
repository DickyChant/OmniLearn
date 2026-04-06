#!/usr/bin/env bash
# PEFT experiment: fine-tune pretrained JetClass model on Top dataset
# with different head/body training combinations, then evaluate all.
#
# Usage: bash run_peft_experiment.sh [FOLDER]
#   FOLDER: path containing TOP/ subdirectory with {train,val,test}_ttbar.h5
#           and checkpoints/ with pretrained JetClass weights.

set -euo pipefail

FOLDER="${1:-.}"
COMMON="--dataset top --layer_scale --local --fine_tune --folder $FOLDER"
TRAIN_COMMON="--warm_epoch 3 --epoch 10 --stop_epoch 3 --batch 256 --wd 0.1 --lr 3e-5"
EVAL_COMMON="--batch 500 --num_timesteps 100"

echo "============================================"
echo "PEFT Experiment on Top Dataset"
echo "Folder: $FOLDER"
echo "============================================"

# --- Training phase ---
# Condition 1: pretrained_only — no training needed, evaluate JetClass checkpoint directly.

# Condition 2: classifier head only (body frozen)
echo "[2/7] Training: classifier head only (freeze_body)"
python train.py $COMMON $TRAIN_COMMON --freeze_body --mode classifier

# Condition 3: generator head only (body frozen)
echo "[3/7] Training: generator head only (freeze_body)"
python train.py $COMMON $TRAIN_COMMON --freeze_body --mode generator

# Condition 4: both heads (body frozen)
echo "[4/7] Training: both heads (freeze_body)"
python train.py $COMMON $TRAIN_COMMON --freeze_body --mode all

# Condition 5: classifier head + LoRA body
echo "[5/7] Training: classifier head + LoRA body"
python train.py $COMMON $TRAIN_COMMON --freeze_body --lora_rank 16 --mode classifier

# Condition 6: generator head + LoRA body
echo "[6/7] Training: generator head + LoRA body"
python train.py $COMMON $TRAIN_COMMON --freeze_body --lora_rank 16 --mode generator

# Condition 7: both heads + LoRA body
echo "[7/8] Training: both heads + LoRA body"
python train.py $COMMON $TRAIN_COMMON --freeze_body --lora_rank 16 --mode all

# Condition 8: LoRA body only (heads frozen, body adapts via LoRA)
echo "[8/8] Training: LoRA body only (heads frozen)"
python train.py $COMMON $TRAIN_COMMON --freeze_body --freeze_heads --lora_rank 16 --mode all

echo ""
echo "============================================"
echo "Training complete. Starting evaluation..."
echo "============================================"

# --- Evaluation phase ---
# Each evaluation runs BOTH classifier head AND generative classification,
# regardless of which head was trained. --mode controls checkpoint name only.

# Condition 1: pretrained only (zero-shot)
echo "[Eval 1/7] Pretrained only (zero-shot)"
python evaluate_generative_classifier.py $COMMON $EVAL_COMMON --mode all --pretrained_only

# Condition 2: classifier head only
echo "[Eval 2/7] Classifier head only (trained cls, untrained gen)"
python evaluate_generative_classifier.py $COMMON $EVAL_COMMON --freeze_body --mode classifier

# Condition 3: generator head only
echo "[Eval 3/7] Generator head only (untrained cls, trained gen)"
python evaluate_generative_classifier.py $COMMON $EVAL_COMMON --freeze_body --mode generator

# Condition 4: both heads
echo "[Eval 4/7] Both heads"
python evaluate_generative_classifier.py $COMMON $EVAL_COMMON --freeze_body --mode all

# Condition 5: classifier + LoRA
echo "[Eval 5/7] Classifier + LoRA (trained cls+body, untrained gen)"
python evaluate_generative_classifier.py $COMMON $EVAL_COMMON --freeze_body --lora_rank 16 --mode classifier

# Condition 6: generator + LoRA
echo "[Eval 6/7] Generator + LoRA (untrained cls, trained gen+body)"
python evaluate_generative_classifier.py $COMMON $EVAL_COMMON --freeze_body --lora_rank 16 --mode generator

# Condition 7: both + LoRA
echo "[Eval 7/8] Both + LoRA"
python evaluate_generative_classifier.py $COMMON $EVAL_COMMON --freeze_body --lora_rank 16 --mode all

# Condition 8: LoRA body only (pretrained heads, adapted body)
echo "[Eval 8/8] LoRA body only (frozen heads)"
python evaluate_generative_classifier.py $COMMON $EVAL_COMMON --freeze_body --freeze_heads --lora_rank 16 --mode all

echo ""
echo "============================================"
echo "All experiments complete!"
echo "============================================"
