#!/bin/bash
eval "$(grep '^export' /home/kkg/.bashrc)"
source /home/kkg/miniconda3/etc/profile.d/conda.sh
conda activate kgqa

EVAL=/home/kkg/code/SubgraphRAG/reason/eval_standalone.py
PRED=/home/kkg/code/SubgraphRAG/reason/results/KGQA/cwq/SubgraphRAG/qwen-plus/scored_100-sys_icl_dc-0.16-thres_0.0-test-predictions.jsonl

echo "========================================="
echo "  IMPROVED CODE - pilot 50 results"
echo "========================================="
python "$EVAL" --pred_file "$PRED" --eval_mode all --breakdown --error_analysis
