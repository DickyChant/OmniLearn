#!/usr/bin/env bash
# Finish PEFT experiment: train condition 8 (LoRA body only) sub-variants,
# then evaluate ALL 10 conditions with both classifier and generative methods.
set -euo pipefail

FOLDER="${1:-.}"
COMMON="--dataset top --layer_scale --local --fine_tune --folder $FOLDER"
TRAIN_COMMON="--warm_epoch 3 --epoch 10 --stop_epoch 3 --batch 256 --wd 0.1 --lr 3e-5"
EVAL_COMMON="--batch 500 --num_timesteps 100 --nevts 50000"

echo "============================================"
echo "PEFT Experiment — Finish (train 8a-c + eval all)"
echo "============================================"

# --- Train condition 8: LoRA body only (frozen heads) ---
for MODE in classifier generator all; do
    echo "[Train 8] LoRA body only, frozen heads, mode=$MODE"
    python train.py $COMMON $TRAIN_COMMON --freeze_body --freeze_heads --lora_rank 16 --mode $MODE
done

echo ""
echo "============================================"
echo "Training complete. Starting evaluation..."
echo "============================================"

# --- Evaluate ALL 10 conditions ---
# Each eval runs BOTH classifier head AND generative classification.

# Condition 1: pretrained only (zero-shot)
echo "[Eval 1] Pretrained only (zero-shot)"
python evaluate_generative_classifier.py $COMMON $EVAL_COMMON --mode all --pretrained_only

# Conditions 2-4: frozen body, heads only
for MODE in classifier generator all; do
    echo "[Eval 2-4] Frozen body, mode=$MODE"
    python evaluate_generative_classifier.py $COMMON $EVAL_COMMON --freeze_body --mode $MODE
done

# Conditions 5-7: frozen body + LoRA + heads
for MODE in classifier generator all; do
    echo "[Eval 5-7] LoRA body + heads, mode=$MODE"
    python evaluate_generative_classifier.py $COMMON $EVAL_COMMON --freeze_body --lora_rank 16 --mode $MODE
done

# Conditions 8a-8c: frozen body + frozen heads + LoRA
for MODE in classifier generator all; do
    echo "[Eval 8] LoRA body only (frozen heads), mode=$MODE"
    python evaluate_generative_classifier.py $COMMON $EVAL_COMMON --freeze_body --freeze_heads --lora_rank 16 --mode $MODE
done

echo ""
echo "============================================"
echo "All experiments complete!"
echo "============================================"
