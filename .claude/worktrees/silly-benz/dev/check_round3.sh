#!/bin/bash
DIR=/home/kkg/code/SubgraphRAG/.claude/worktrees/silly-benz/reason/results/KGQA/cwq/SubgraphRAG/qwen-plus
echo "=== Files in output dir ==="
ls -la "$DIR/" 2>/dev/null || echo "Dir not found"
echo ""
echo "=== Check resume file ==="
RESUME="$DIR/scored_100-sys_icl_dc-0.16-thres_0.0-test-predictions-resume.jsonl"
if [ -f "$RESUME" ]; then
    echo "Resume: $(wc -l < "$RESUME") lines"
fi
echo "=== Check final file ==="
FINAL="$DIR/scored_100-sys_icl_dc-0.16-thres_0.0-test-predictions.jsonl"
if [ -f "$FINAL" ]; then
    echo "Final: $(wc -l < "$FINAL") lines"
fi
