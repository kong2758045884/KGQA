#!/bin/bash
set -e

source /home/kkg/miniconda3/etc/profile.d/conda.sh
conda activate kgqa
eval "$(grep '^export' /home/kkg/.bashrc)"

cd /home/kkg/code/SubgraphRAG/reason

BASELINE_PRED=./results/KGQA/cwq/SubgraphRAG/qwen-plus/scored_100-sys_icl_dc-0.16-thres_0.0-test-predictions.jsonl

echo "=========================================="
echo "  Re-evaluating FULL BASELINE with all eval modes"
echo "=========================================="
echo "File: $BASELINE_PRED"
echo "Samples: $(wc -l < "$BASELINE_PRED")"
echo ""

# Use the IMPROVED eval_standalone.py from the worktree
EVAL_SCRIPT=/home/kkg/code/SubgraphRAG/.claude/worktrees/silly-benz/reason/eval_standalone.py

python "$EVAL_SCRIPT" --pred_file "$BASELINE_PRED" --eval_mode all --breakdown
