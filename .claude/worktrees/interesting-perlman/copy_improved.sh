#!/bin/bash
set -e
SRC=/home/kkg/code/SubgraphRAG/.claude/worktrees/interesting-perlman/reason
DST=/home/kkg/code/SubgraphRAG/reason

# Backup originals (only if .bak doesn't already exist)
for f in prompts.py llm_utils.py eval_standalone.py main.py; do
    [ ! -f "$DST/$f.bak" ] && cp "$DST/$f" "$DST/$f.bak"
done
[ ! -f "$DST/preprocess/prepare_prompts.py.bak" ] && cp "$DST/preprocess/prepare_prompts.py" "$DST/preprocess/prepare_prompts.py.bak"

# Copy improved files
cp "$SRC/prompts.py" "$DST/prompts.py"
cp "$SRC/llm_utils.py" "$DST/llm_utils.py"
cp "$SRC/eval_standalone.py" "$DST/eval_standalone.py"
cp "$SRC/preprocess/prepare_prompts.py" "$DST/preprocess/prepare_prompts.py"
cp "$SRC/main.py" "$DST/main.py"

echo "All improved files copied successfully."
grep -c "dc_fallback_prompt" "$DST/prompts.py" "$DST/llm_utils.py" "$DST/main.py"
grep -c "postprocess_output" "$DST/llm_utils.py"
grep -c "eval_mode" "$DST/eval_standalone.py"
