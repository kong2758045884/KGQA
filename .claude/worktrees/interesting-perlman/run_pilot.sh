#!/bin/bash
# Source env vars directly (bashrc guard blocks non-interactive)
eval "$(grep '^export' /home/kkg/.bashrc)"
source /home/kkg/miniconda3/etc/profile.d/conda.sh
conda activate kgqa

cd /home/kkg/code/SubgraphRAG/reason

echo "Python: $(which python)"
echo "API key set: $([ -n "$DASHSCOPE_API_KEY" ] && echo YES || echo NO)"
echo "BASE_URL: $OPENAI_BASE_URL"

python main.py \
  -d cwq \
  --prompt_mode scored_100 \
  --llm_mode sys_icl_dc \
  -m qwen-plus \
  --split test \
  --frequency_penalty 0.16 \
  --thres 0.0 \
  --pilot 50 \
  --no_wandb
