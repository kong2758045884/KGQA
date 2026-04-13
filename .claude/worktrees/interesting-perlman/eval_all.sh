#!/bin/bash
eval "$(grep '^export' /home/kkg/.bashrc)"
source /home/kkg/miniconda3/etc/profile.d/conda.sh
conda activate kgqa

EVAL=/home/kkg/code/SubgraphRAG/reason/eval_standalone.py
DIR=/home/kkg/code/SubgraphRAG/results/KGQA/cwq/SubgraphRAG/qwen-plus

echo "========================================="
echo "  BASELINE pilot50-backup"
echo "========================================="
python "$EVAL" --pred_file "$DIR/scored_100-sys_icl_dc-0.16-thres_0.0-test-predictions-pilot50-backup.jsonl" --eval_mode all --breakdown

echo ""
echo "========================================="
echo "  BASELINE (current file, same as above)"
echo "========================================="
python "$EVAL" --pred_file "$DIR/baseline-pilot50-backup.jsonl" --eval_mode all --breakdown

echo ""
echo "========================================="
echo "  REORG pilot50"
echo "========================================="
python "$EVAL" --pred_file "$DIR/scored_100_reorg-sys_icl_dc-0.16-thres_0.0-test-predictions.jsonl" --eval_mode all --breakdown
