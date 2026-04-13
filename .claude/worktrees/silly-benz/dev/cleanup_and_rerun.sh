#!/bin/bash
set -e

DIR=/home/kkg/code/SubgraphRAG/.claude/worktrees/silly-benz/reason/results/KGQA/cwq/SubgraphRAG/qwen-plus

# Backup round 3a (failed) results
mv "${DIR}/scored_100-sys_icl_dc-0.16-thres_0.0-test-predictions.jsonl" "${DIR}/round3a-failed-predictions.jsonl" 2>/dev/null || true
mv "${DIR}/scored_100-sys_icl_dc-0.16-thres_0.0-test-predictions_eval_results_all.json" "${DIR}/round3a-failed-eval.json" 2>/dev/null || true
rm -f "${DIR}/scored_100-sys_icl_dc-0.16-thres_0.0-test-predictions-resume.jsonl" 2>/dev/null || true

echo "Cleaned up. Files in dir:"
ls -la "${DIR}/"

echo ""
echo "=========================================="
echo "  Running Pilot 50 — Round 3b (DC restored)"
echo "=========================================="

source /home/kkg/miniconda3/etc/profile.d/conda.sh
conda activate kgqa
eval "$(grep '^export' /home/kkg/.bashrc)"

cd /home/kkg/code/SubgraphRAG/.claude/worktrees/silly-benz/reason

echo "Start time: $(date)"

python main.py -d cwq --prompt_mode scored_100 --llm_mode sys_icl_dc \
  -m qwen-plus --frequency_penalty 0.16 --thres 0.0 \
  --pilot 50 --no_wandb

echo ""
echo "Inference done at: $(date)"

PRED_FILE=./results/KGQA/cwq/SubgraphRAG/qwen-plus/scored_100-sys_icl_dc-0.16-thres_0.0-test-predictions.jsonl

echo "=========================================="
echo "  Evaluating — all modes + breakdown"
echo "=========================================="

python eval_standalone.py --pred_file "$PRED_FILE" --eval_mode all --breakdown

echo ""
echo "=========================================="
echo "  Pilot 50 Round 3b Complete"
echo "=========================================="
echo "End time: $(date)"
