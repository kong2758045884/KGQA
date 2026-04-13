#!/bin/bash
set -e

WT=/home/kkg/code/SubgraphRAG/.claude/worktrees/silly-benz
MAIN=/home/kkg/code/SubgraphRAG

echo "Setting up worktree symlinks..."

# scored_triples in reason/
ln -sf "${MAIN}/scored_triples/cwq_240907_unidir_test.pth" "${WT}/reason/scored_triples/cwq_240907_unidir_test.pth"
ln -sf "${MAIN}/scored_triples/webqsp_240912_unidir_test.pth" "${WT}/reason/scored_triples/webqsp_240912_unidir_test.pth"

# scored_triples in root
ln -sf "${MAIN}/scored_triples/cwq_240907_unidir_test.pth" "${WT}/scored_triples/cwq_240907_unidir_test.pth"
ln -sf "${MAIN}/scored_triples/webqsp_240912_unidir_test.pth" "${WT}/scored_triples/webqsp_240912_unidir_test.pth"

# RoG results — main.py runs from reason/, so relative path ./results/ means reason/results/
mkdir -p "${WT}/reason/results/KGQA/cwq/SubgraphRAG/qwen-plus"
mkdir -p "${WT}/reason/results/KGQA/cwq/RoG/test"
ln -sfn "${MAIN}/results/KGQA/cwq/RoG/test/results_gen_rule_path_RoG-cwq_RoG_test_predictions_3_False_jsonl" \
        "${WT}/reason/results/KGQA/cwq/RoG/test/results_gen_rule_path_RoG-cwq_RoG_test_predictions_3_False_jsonl"

# Also link root results (for eval)
mkdir -p "${WT}/results/KGQA/cwq/RoG/test"
ln -sfn "${MAIN}/results/KGQA/cwq/RoG/test/results_gen_rule_path_RoG-cwq_RoG_test_predictions_3_False_jsonl" \
        "${WT}/results/KGQA/cwq/RoG/test/results_gen_rule_path_RoG-cwq_RoG_test_predictions_3_False_jsonl"

echo "=== Verify ==="
ls -la "${WT}/reason/scored_triples/"
ls -la "${WT}/reason/results/KGQA/cwq/RoG/test/"
echo "Done."
