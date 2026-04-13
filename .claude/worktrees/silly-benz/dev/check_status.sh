#!/bin/bash
echo "=== Baseline pilot 50 eval results ==="
cat /home/kkg/code/SubgraphRAG/results/KGQA/cwq/SubgraphRAG/qwen-plus/baseline-pilot50-backup_eval_results_all.json 2>/dev/null || echo "NOT FOUND"

echo ""
echo "=== Round 2 pilot 50 eval results ==="
cat /home/kkg/code/SubgraphRAG/results/KGQA/cwq/SubgraphRAG/qwen-plus/scored_100-sys_icl_dc-0.16-thres_0.0-test-predictions-pilot50-backup_eval_results_all.json 2>/dev/null || echo "NOT FOUND"

echo ""
echo "=== Full baseline eval results (saved JSON) ==="
cat /home/kkg/code/SubgraphRAG/results/KGQA/cwq/SubgraphRAG/qwen-plus/scored_100-sys_icl_dc-0.16-thres_0.0-test-predictions_eval_results.json 2>/dev/null || echo "NOT FOUND"
find /home/kkg/code/SubgraphRAG -name "*eval_result*" -path "*qwen*" 2>/dev/null

echo ""
echo "=== Round 3 pilot progress ==="
RESUME=/home/kkg/code/SubgraphRAG/.claude/worktrees/silly-benz/reason/results/KGQA/cwq/SubgraphRAG/qwen-plus/scored_100-sys_icl_dc-0.16-thres_0.0-test-predictions-resume.jsonl
if [ -f "$RESUME" ]; then
    wc -l "$RESUME"
else
    echo "Resume file not found"
fi
