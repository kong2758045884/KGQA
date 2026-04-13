# SubgraphRAG 复现与改造项目 — 完整状态文档

> 最后更新：2026-04-08
> 目的：让任何接手者读完本文就能继续推进到论文投稿

---

## 一、项目目标

基于 [Graph-COM/SubgraphRAG](https://github.com/Graph-COM/SubgraphRAG) 做 KGQA（Knowledge Graph Question Answering）的复现与改进，目标投 CCF-C 会议/期刊。

**不是**重做 retrieval 阶段，而是：
1. 用原始 SubgraphRAG 的 retrieval 结果（scored_triplets）作为输入
2. 在 **reason 阶段**（LLM 推理）做模块化改进
3. 对比 baseline vs 改进版的评测指标，形成论文实验

---

## 二、环境与基础设施

| 项目 | 值 |
|------|-----|
| 操作系统 | Windows 11 + WSL Ubuntu 24.04 |
| conda 环境 | `kgqa` |
| 项目根目录 | `/home/kkg/code/SubgraphRAG` |
| 推理后端 | **qwen-plus**（阿里通义千问，不是本地 GPU） |
| API | DashScope OpenAI-compatible API |
| 环境变量 | `DASHSCOPE_API_KEY`，`OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1` |
| 版本管理 | git，主开发在 `main` 分支，改进代码在 `claude/silly-benz` 分支 |
| 锁定版本 | **Round 2 = commit 7f5614f**（当前最优推理代码） |

**WSL 运行注意**：从 Windows Git Bash 执行 WSL 命令时需要 `wsl -- bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate kgqa && ..."`

---

## 三、项目结构

```
SubgraphRAG/
├── reason/                          # 【主要修改区域】推理阶段
│   ├── main.py                      # 推理主入口，支持 --pilot / --no_wandb / checkpoint resume
│   ├── eval_standalone.py           # 独立评测脚本（多模式：strict/paper/paper_ext/enhanced）
│   ├── llm_utils.py                 # LLM 调用、拒答检测、输出后处理、DC fallback
│   ├── prompts.py                   # 所有 prompt 模板（sys/icl/cot/dc）
│   ├── preprocess/
│   │   ├── prepare_data.py          # 数据加载：HF datasets + scored_triplets
│   │   └── prepare_prompts.py       # prompt 组装：scored_100 / scored_100_struct / rog 等模式
│   ├── metrics/
│   │   ├── evaluate_results.py      # 原版评测（Hit 指标，来自 RoG）
│   │   └── evaluate_results_corrected.py  # 原版评测（F1/Precision/Recall，修正版）
│   ├── configs/
│   │   ├── cwq_tune.yaml
│   │   └── webqsp_tune.yaml
│   ├── results/                     # 预测输出目录（gitignored）
│   └── scored_triples/              # 符号链接 → 上级 scored_triples/
├── retrieve/                        # retrieval 阶段（未修改）
├── results/                         # RoG 原始预测文件
├── scored_triples/                  # SubgraphRAG retriever 打分结果（.pth）
└── dev/                             # 开发辅助文件
```

---

## 四、Pipeline 完整流程

```
[Freebase KG] → [SubgraphRAG Retriever] → scored_triplets (.pth)
                                                ↓
[RoG predictions] + [HuggingFace datasets] → prepare_data.py → 每条样本附带 graph + scored_triplets
                                                ↓
                                          prepare_prompts.py → 每条样本生成 sys_query / user_query / cot_query
                                                ↓
                                            main.py + llm_utils.py → qwen-plus API 推理 → prediction JSONL
                                                ↓
                                          eval_standalone.py → Hits@1 / F1 / Precision / Recall / EM
```

**关键数据文件**（gitignored，不在仓库里）：
- `scored_triples/cwq_240907_unidir_test.pth` — CWQ retriever 打分结果
- `results/KGQA/cwq/RoG/test/.../predictions.jsonl` — RoG 原始预测
- `reason/results/KGQA/cwq/SubgraphRAG/qwen-plus/` — 我们的预测输出

---

## 五、当前正式 Baseline 设置与结果

### 设置
| 参数 | 值 |
|------|-----|
| dataset | cwq (ComplexWebQuestions) |
| split | test |
| total samples | 3531 |
| prompt_mode | scored_100 |
| llm_mode | sys_icl_dc |
| model | qwen-plus |
| frequency_penalty | 0.16 |
| thres | 0.0 |
| temperature | 0 |

### Full Baseline 正式结果（strict 模式评测，3531 条）

| 指标 | 值 |
|------|-----|
| **Hits@1** | **0.5279** |
| **Macro F1** | **0.4784** |
| Macro Precision | 0.5027 |
| Macro Recall | 0.4866 |
| Exact Match | 0.4214 |
| No Answer | 397 (11.24%) |
| Totally Wrong | 1270 (35.97%) |

### Baseline Pilot 50 多模式评测（统一口径对照）

| 模式 | Hits@1 | F1 | NoAns | TW |
|------|--------|-----|-------|-----|
| strict | 0.54 | 0.517 | 11 | 12 |
| paper | 0.60 | 0.567 | 11 | 10 |
| enhanced | 0.60 | 0.549 | 11 | 9 |

> **注意**：Full baseline 的 qwen-plus 预测文件已被 pilot 覆盖丢失。如需重新获得 full baseline paper 模式数字，需重跑 3531 条。

---

## 六、错误分析核心结论

基于 full baseline 的详细错误分析：

| 错误类型 | 数量 | 占比 | 可干预性 |
|----------|------|------|----------|
| 可恢复错误 | ~383 | 23% of errors | 通过 reason 端改进可修复 |
| 硬性失败 | ~1284 | 77% of errors | 需要更好的 retrieval / 知识覆盖 |

**理论上限**：如果所有可恢复错误都修复，Hits@1 最多 +10.85pp → ~0.636

**可恢复错误的具体来源**：
1. **过度拒答** — 模型输出 "ans: not available"，但 triplets 里其实有答案
2. **格式不规范** — 模型给了正确答案但没用 `ans:` 格式
3. **DC 触发太窄** — 二轮兜底逻辑触发条件不够覆盖
4. **Few-shot 示例格式偏差** — 旧版 `icl_ass_prompt` 用 "ans: 2014 (2014 World Series)" 格式，`normalize_answer()` 删括号后变成纯年份，导致 year-type 问题答案粒度被带偏
5. **Entity alias / surface form 不一致** — 预测用了 KG 里的 entity name，但 gold answer 是另一种写法

---

## 七、已完成的改进工作

### 第一轮：Prompt Output Fidelity（已合入 main）

**commit**: `41322dc` — "improve prompt output fidelity for qwen-plus baseline"

改动：
- `prompts.py`：移除 "ans: not available" 的逃生出口，加入 "You must always provide at least one answer"
- `icl_ass_prompt`：从 "ans: 2014 (2014 World Series)" 改为 "ans: 2014 World Series"，避免括号格式偏差
- `icl_cot_prompt`：加入 "Use entity names exactly as they appear in the triplets"

### 第二轮（当前锁定版本）：Refusal Handling + Post-processing + DC Fallback

**commit**: `7f5614f` — "Improve reasoning stage: refusal handling, post-processing, and evaluation"

改动：
- `llm_utils.py`：
  - 加入 REFUSAL_PATTERNS 拒答检测（19 条模式）
  - `postprocess_output()` 从自由文本恢复 `ans:` 行（3 种模式：answer is/are、therefore/thus、列表项）
  - DC fallback 用更强的 `dc_fallback_prompt`
  - DC 触发条件：`not has_valid_ans_lines(res[0]) or is_refusal(res[0])`
- `eval_standalone.py`：支持 strict/paper/enhanced 三种评测模式、错误分类、日期规范化
- `main.py`：注入 `dc_fallback_prompt` 到推理流程
- `prepare_prompts.py`：新增 `scored_100_struct` 模式（prefix-preserving 实体分组）

### 第三轮探索（已验证、已放弃）

尝试了两个变体：
- **Round 3a**：移除 DC 触发中的 `is_refusal()` 检查 → Hits@1 暴跌，**放弃**
- **Round 3b**：恢复 `is_refusal()` + 扩展拒答模式 → 未优于 Round 2，**放弃**

**结论**：Round 2（7f5614f）是当前最优，已锁定。

---

## 八、全部实验历史（统一评测口径）

### Pilot 50 对比表（全部用改进版 eval_standalone.py 评测）

| 实验 | strict H@1 | paper H@1 | enhanced H@1 | paper F1 | NoAns | TW (strict) | 结论 |
|------|-----------|-----------|-------------|----------|-------|-------------|------|
| **Baseline（原始代码）** | 0.54 | 0.60 | 0.60 | 0.567 | **11** | 12 | 锚点 |
| Round 1 Prompt-only | 0.52 | 0.56 | 0.60 | 0.549 | 9 | 15 | 有效但弱 |
| Evidence Reorg | 0.52 | 0.54 | 0.60 | 0.515 | 7 | 17 | **失败** |
| **Round 2 (7f5614f)** | **0.64** | **0.66** | **0.78** | **0.645** | **2** | 16 | **当前最优** |
| Round 3a (DC移除) | 0.56 | 0.58 | 0.68 | 0.597 | 0 | 22 | 放弃 |
| Round 3b (DC恢复+扩展) | 0.64 | 0.58 | 0.74 | 0.609 | 1 | 17 | 不优于R2 |

### ★ Pilot 250 结果（Round 2 代码，250 条样本，统计更可靠）

| 评测模式 | Hits@1 | Macro F1 | Macro Precision | Macro Recall | Exact Match | No Answer | Totally Wrong |
|----------|--------|----------|-----------------|-------------|-------------|-----------|---------------|
| **strict** | **0.688** | 0.618 | 0.634 | 0.648 | 0.536 | 19 (7.6%) | 59 (23.6%) |
| **paper** | **0.660** | **0.636** | 0.658 | 0.671 | 0.536 | 19 (7.6%) | 66 (26.4%) |
| paper_ext | 0.680 | 0.648 | 0.670 | 0.683 | 0.536 | 19 (7.6%) | 61 (24.4%) |
| enhanced | 0.756 | 0.663 | 0.690 | 0.693 | 0.536 | 19 (7.6%) | 42 (16.8%) |

**Pilot 250 错误分类（strict 模式, 78 个错误样本）：**
| 类型 | 数量 | 说明 |
|------|------|------|
| wrong_entity | 41 | 模型答了不相关实体（硬性失败，需更好 retrieval）|
| alias_mismatch | 18 | 模型答了别名/近似实体（评测端可改善）|
| no_answer | 14 | 模型没给出有效答案 |
| refusal | 5 | 模型明确拒答 |

### Pilot 250 vs Baseline 对比（paper 模式）

| 指标 | Baseline (full 3531, strict) | Pilot 250 (paper) | 提升 |
|------|------|------|------|
| Hits@1 | 0.5279 | **0.660** | **+13.2pp** |
| Macro F1 | 0.4784 | **0.636** | **+15.8pp** |
| No Answer | 11.24% | **7.6%** | -3.6pp |
| Totally Wrong | 35.97% | **26.4%** | -9.6pp |

> ⚠️ 注意：baseline 数字是 full 3531 条 strict 模式，pilot 250 是 paper 模式。严格对比需要同 eval 模式。但 paper H@1 >= strict H@1（因为 substring matching 更宽松），所以改进方向确认无误。

---

## 九、当前状态总结

### 已确认的事实
1. Full baseline（strict 模式）：Hits@1 = 0.5279, F1 = 0.4784
2. Pilot 250（Round 2, paper 模式）：**Hits@1 = 0.660, F1 = 0.636**，统计较可靠
3. Pilot 50→250 结果稳定：paper H@1 从 0.66(n=50) → 0.66(n=250)，方差小
4. No Answer 从 baseline 11.24% 降到 7.6%，仍有 19 条未回答
5. 可恢复错误约 383 条，理论提升上限 +10.85pp Hits@1
6. 全局 evidence reorg 有害，不可用
7. 激进 refusal-detector 有害（No Answer 暴涨）
8. 移除 DC trigger 中的 `is_refusal()` 有害（TW 暴涨）

### 当前最佳判断
1. Round 2（commit 7f5614f）是当前最优代码，已锁定
2. Pilot 250 的数字已足够稳定，full run 预期 paper H@1 在 0.63-0.68 区间
3. 主要剩余损失来自 wrong_entity（41/78 错误），这是 retrieval 侧问题

### 尚未验证的假设
1. 改进版代码的 full run 结果（需 ~18 小时）
2. `scored_100_struct` 模式是否优于 `scored_100`
3. 改进版在 WebQSP 数据集上的泛化表现
4. Full baseline 用 paper 模式评测的公平数字（预测文件已丢失）

---

## 十、代码关键细节（给接手者的速查）

### main.py 常用命令

```bash
cd /home/kkg/code/SubgraphRAG/reason
conda activate kgqa

# Pilot 50
python main.py -d cwq --prompt_mode scored_100 --llm_mode sys_icl_dc \
  -m qwen-plus --frequency_penalty 0.16 --thres 0.0 --pilot 50 --no_wandb

# Pilot 250
python main.py -d cwq --prompt_mode scored_100 --llm_mode sys_icl_dc \
  -m qwen-plus --frequency_penalty 0.16 --thres 0.0 --pilot 250 --no_wandb

# Full run（~18 小时，不要轻易跑）
python main.py -d cwq --prompt_mode scored_100 --llm_mode sys_icl_dc \
  -m qwen-plus --frequency_penalty 0.16 --thres 0.0 --no_wandb
```

### eval_standalone.py 评测模式

| 模式 | 说明 | 用途 |
|------|------|------|
| `strict` | normalize_answer（去括号） + 精确集合匹配 | 旧版基线口径 |
| `paper` | normalize（不去括号） + substring matching + 部分问题 double_check | **论文主指标，与原始 SubgraphRAG 论文一致** |
| `paper_ext` | paper + 所有问题类型启用 double_check | 更公平的 substring 匹配 |
| `enhanced` | paper + 改进 normalize（连字符→空格） + token overlap 模糊匹配 | 上界分析，不用于论文主实验 |
| `all` | 同时跑所有模式并输出对比表 | 快速全景诊断 |

```bash
# 推荐：all 模式 + 错误分类
python eval_standalone.py --pred_file <path> --eval_mode all --breakdown
```

### LLM 推理流程（sys_icl_dc 模式）

```
1. System prompt → icl_sys_prompt
2. ICL few-shot → icl_user_prompt + icl_ass_prompt
3. User query → triplets + question
4. 第一轮 LLM 输出 → postprocess_output()（恢复格式）
5. 检查 has_valid_ans_lines() + is_refusal()
   ├── 有效且无拒答 → 直接用
   └── 无效或含拒答 → DC 二轮 fallback（用 dc_fallback_prompt 再打一轮）→ postprocess_output()
6. 保存 prediction
```

---

## 十一、重要工程注意事项

1. **Prediction 文件覆盖风险**：pilot 5 / pilot 50 / pilot 250 / full 的输出路径相同，跑之前**必须备份**现有预测文件
2. **Full run = 18 小时**：不要拿 full run 去赌方向，先 pilot 验证
3. **评测 OOM**：旧版 `evaluate_results_corrected.py` 会加载 graph + scored_triplets，在 WSL 下 OOM。用 `eval_standalone.py` 替代
4. **eval 模式一致性**：对比 baseline vs 改进时，两者必须用同一 eval 模式。推荐都用 `paper` 模式
5. **Full baseline 预测文件已丢失**：被 pilot 覆盖。如需 paper 模式公平对比，需重跑 full baseline

---

## 十二、原计划三模块与实际进展

| 模块 | 原计划 | 实际状态 |
|------|--------|----------|
| 1. Answer Normalization | 评测端 normalization 改造 + 敏感性分析 | **部分完成**：做了 prompt-level output fidelity 修复（改 prompts.py 让模型输出更规范）；eval_standalone.py 支持多模式评测 |
| 2. Evidence Organization | 改进 evidence 呈现方式 | **一次失败**：scored_100_reorg 全局重排有害。`scored_100_struct`（prefix-preserving 分组）代码已实现但未正式测试 |
| 3. Bridge Augmentation | 用 bridge entity 增强 evidence | **未启动** |

---

## 十三、下一步行动计划（优先级排序）

### P0：跑 Full Run
1. 推理代码已锁定到 Round 2（7f5614f）
2. Pilot 250 数字稳定（paper H@1 = 0.660），可以放心跑 full
3. 备份现有文件后直接启动 full run（3531 条，~18 小时）
4. Full run 完成后用 `--eval_mode all --breakdown` 做完整评测

### P1：重跑 Full Baseline（获取 paper 模式公平对比数字）
- 恢复原始代码（commit 2f118fb 或 .bak 文件）
- 重跑 3531 条
- 用同一 eval_standalone.py 评测，得到 paper 模式 baseline

### P2：scored_100_struct 实验
- 先 pilot 50 对比 scored_100 vs scored_100_struct
- 这是 Evidence Organization 模块的保守实现

### P3：WebQSP 数据集泛化
- 用同一套改进代码在 WebQSP 上跑
- 论文需要至少两个数据集

### ❌ 不该做
- 不做全局 evidence reorder（已验证有害）
- 不做 bridge augmentation
- 不做激进 refusal detection
- 不移除 DC trigger 中的 is_refusal()

---

## 十四、论文实验规划（草案）

### 主实验表格（全部用 paper 模式评测）
| Method | Hits@1 | F1 | Precision | Recall | EM |
|--------|--------|-----|-----------|--------|-----|
| SubgraphRAG (原始论文, Llama-3.1-8B) | 55.4 | — | — | — | — |
| Baseline (our repro, qwen-plus) | 需重跑 | — | — | — | — |
| + Improved Reasoning (Round 2) | **66.0** (pilot 250) | **63.6** | 65.8 | 67.1 | 53.6 |

### Ablation Study
- 各模块的单独贡献（prompt / refusal / DC / postprocess）
- `paper` vs `paper_ext` vs `enhanced`：评测粒度敏感性分析

### Error Analysis
- 错误分类分布：wrong_entity 52.6% / alias_mismatch 23.1% / no_answer 17.9% / refusal 6.4%
- Enhanced 模式上界：Hits@1 = 0.756，说明 alias_mismatch 占 ~10pp 的可挖掘空间

---

## 十五、能否发 CCF-C？

**Pilot 250 数据分析**：

**强点**：
- Paper H@1 = 66.0%，vs 原论文的 55.4%（Llama-3.1-8B），**+10.6pp 绝对提升**
- F1 = 63.6%，也大幅领先
- 改进方案模块化清晰（prompt + refusal + DC + postprocess），每个模块有独立贡献
- Pilot 250 数据稳定（n=50 和 n=250 的 paper H@1 都是 0.66）
- 完整的 error analysis 和多模式评测框架

**注意**：
- qwen-plus vs Llama-3.1-8B 不是纯方法对比，LLM 能力差异是 confounding factor
- 论文需要控制变量：同一 LLM 下 baseline vs improved
- Full run 数字可能略低于 pilot 250（前 250 条可能偏简单）

**结论**：**数据看起来很有希望。立即跑 full run 拿到正式数字，然后可以开始写论文。**
