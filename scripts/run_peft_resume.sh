#!/usr/bin/env bash
# Resume PEFT experiment from condition 3 (condition 2 already done)
set -euo pipefail

FOLDER="${1:-.}"
COMMON="--dataset top --layer_scale --local --fine_tune --folder $FOLDER"
TRAIN_COMMON="--warm_epoch 3 --epoch 10 --stop_epoch 3 --batch 256 --wd 0.1 --lr 3e-5"
EVAL_COMMON="--batch 500 --num_timesteps 100"

echo "============================================"
echo "PEFT Experiment (resumed from condition 3)"
echo "============================================"

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
echo "[7/7] Training: both heads + LoRA body"
python train.py $COMMON $TRAIN_COMMON --freeze_body --lora_rank 16 --mode all

echo ""
echo "============================================"
echo "Training complete. Starting evaluation..."
echo "============================================"

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
echo "[Eval 7/7] Both + LoRA"
python evaluate_generative_classifier.py $COMMON $EVAL_COMMON --freeze_body --lora_rank 16 --mode all

echo ""
echo "============================================"
echo "All experiments complete!"
echo "============================================"
