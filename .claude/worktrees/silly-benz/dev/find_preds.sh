#!/bin/bash
echo "=== All prediction JSONL files with line counts ==="
find /home/kkg/code/SubgraphRAG -name "*prediction*" -name "*.jsonl" 2>/dev/null | while read f; do
    lines=$(wc -l < "$f")
    echo "$lines $f"
done | sort -rn

echo ""
echo "=== Also check for backup files ==="
find /home/kkg/code/SubgraphRAG -name "*backup*" -o -name "*bak*" -o -name "*.bak" 2>/dev/null | while read f; do
    lines=$(wc -l < "$f" 2>/dev/null || echo "?")
    echo "$lines $f"
done

echo ""
echo "=== Check for full baseline (3531 lines) ==="
find /home/kkg/code/SubgraphRAG -name "*.jsonl" 2>/dev/null | while read f; do
    lines=$(wc -l < "$f")
    if [ "$lines" -gt 3000 ]; then
        echo "$lines $f"
    fi
done
