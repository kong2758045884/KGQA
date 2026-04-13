#!/bin/bash
set -e

source /home/kkg/miniconda3/etc/profile.d/conda.sh
conda activate kgqa
eval "$(grep '^export' /home/kkg/.bashrc)"

DIR=/home/kkg/code/SubgraphRAG/.claude/worktrees/silly-benz/reason/results/KGQA/cwq/SubgraphRAG/qwen-plus

# Clean up previous pilot files
rm -f "${DIR}/scored_100-sys_icl_dc-0.16-thres_0.0-test-predictions-resume.jsonl" 2>/dev/null
rm -f "${DIR}/scored_100-sys_icl_dc-0.16-thres_0.0-test-predictions.jsonl" 2>/dev/null
rm -f "${DIR}/scored_100-sys_icl_dc-0.16-thres_0.0-test-predictions_eval_results_all.json" 2>/dev/null

cd /home/kkg/code/SubgraphRAG/.claude/worktrees/silly-benz/reason

echo "=========================================="
echo "  Pilot 250 — Round 2 Code (7f5614f)"
echo "=========================================="
echo "Start time: $(date)"

python main.py -d cwq --prompt_mode scored_100 --llm_mode sys_icl_dc \
  -m qwen-plus --frequency_penalty 0.16 --thres 0.0 \
  --pilot 250 --no_wandb

echo ""
echo "Inference done at: $(date)"

PRED_FILE=./results/KGQA/cwq/SubgraphRAG/qwen-plus/scored_100-sys_icl_dc-0.16-thres_0.0-test-predictions.jsonl

echo "=========================================="
echo "  Evaluating — all modes + breakdown"
echo "=========================================="

python eval_standalone.py --pred_file "$PRED_FILE" --eval_mode all --breakdown

echo ""
echo "=========================================="
echo "  Pilot 250 Complete"
echo "=========================================="
echo "End time: $(date)"
