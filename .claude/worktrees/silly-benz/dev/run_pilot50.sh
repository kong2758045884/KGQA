#!/bin/bash
set -e

source /home/kkg/miniconda3/etc/profile.d/conda.sh
conda activate kgqa
eval "$(grep '^export' /home/kkg/.bashrc)"

cd /home/kkg/code/SubgraphRAG/.claude/worktrees/silly-benz/reason

echo "=========================================="
echo "  Running Pilot 50 — Round 3 Improved Code"
echo "=========================================="
echo "Start time: $(date)"

python main.py -d cwq --prompt_mode scored_100 --llm_mode sys_icl_dc \
  -m qwen-plus --frequency_penalty 0.16 --thres 0.0 \
  --pilot 50 --no_wandb

echo ""
echo "Inference done. Start time: $(date)"
echo ""

PRED_FILE=./results/KGQA/cwq/SubgraphRAG/qwen-plus/scored_100-sys_icl_dc-0.16-thres_0.0-test-predictions.jsonl

echo "=========================================="
echo "  Evaluating — all modes + breakdown"
echo "=========================================="

python eval_standalone.py --pred_file "$PRED_FILE" --eval_mode all --breakdown

echo ""
echo "=========================================="
echo "  Pilot 50 Complete"
echo "=========================================="
echo "End time: $(date)"
