"""
eval_standalone.py — 独立评测脚本

功能：读取推理结果 JSONL，计算 Hits@1 / F1 / Precision / Recall / Exact Match
优势：不加载 graph、scored_triplets 等大体积数据，避免 WSL 下 OOM

用法：
    python eval_standalone.py --pred_file <path_to_predictions.jsonl>
    python eval_standalone.py --pred_file <path> --verbose        # 打印每条的详细匹配
    python eval_standalone.py --pred_file <path> --error_analysis  # 输出错误 case 到文件
"""

import os
import re
import json
import string
import argparse
from collections import defaultdict


# ============================================================
#  答案规范化（这就是你的 Module 1: Answer Normalization 的基础）
# ============================================================

def normalize_answer(text):
    """对单个答案字符串做规范化，用于匹配判断"""
    if text is None:
        return ""
    text = str(text)
    # 小写
    text = text.lower()
    # 去除括号内的补充说明，如 "2014 (2014 World Series)" → "2014"
    text = re.sub(r'\(.*?\)', '', text)
    # 去除冠词
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    # 去除标点
    text = text.translate(str.maketrans('', '', string.punctuation))
    # 合并多余空格
    text = ' '.join(text.split())
    return text.strip()


# ============================================================
#  从 LLM 输出中解析答案
# ============================================================

def parse_prediction(prediction_text):
    """
    从 LLM 输出中提取 ans: 开头的答案列表
    返回: list[str]（已规范化）
    """
    if not prediction_text:
        return []

    answers = []
    for line in prediction_text.split('\n'):
        line = line.strip()
        # 匹配 "ans:" 或 "ans :" 开头（不区分大小写）
        match = re.match(r'^ans\s*:\s*(.+)', line, re.IGNORECASE)
        if match:
            ans = match.group(1).strip()
            if ans and ans.lower() not in ('not available', 'no information available', 'n/a', 'none'):
                answers.append(ans)

    return answers


# ============================================================
#  从 gold answer 字段提取答案（兼容多种格式）
# ============================================================

def parse_gold_answers(answer_field):
    """
    兼容 CWQ / WebQSP 的多种 gold answer 格式：
    - list[str]:             ["United States", "Canada"]
    - list[dict]:            [{"answer_id": "m.09c7w0", "entity_name": "United States"}, ...]
    - list[dict] (alt key):  [{"answer": "United States"}, ...]
    - str:                   "United States"
    返回: list[str]（原始文本，未规范化）
    """
    if answer_field is None:
        return []

    if isinstance(answer_field, str):
        # 有时 answer 是 JSON 字符串
        try:
            answer_field = json.loads(answer_field)
        except (json.JSONDecodeError, TypeError):
            return [answer_field]

    if isinstance(answer_field, list):
        results = []
        for item in answer_field:
            if isinstance(item, str):
                results.append(item)
            elif isinstance(item, dict):
                # 按优先级尝试不同的 key
                for key in ['entity_name', 'answer', 'label', 'name']:
                    if key in item and item[key]:
                        results.append(str(item[key]))
                        break
                else:
                    # 如果都没有，取 answer_id
                    if 'answer_id' in item:
                        results.append(str(item['answer_id']))
        return results

    return [str(answer_field)]


# ============================================================
#  指标计算
# ============================================================

def compute_hits_at_1(pred_set, gold_set):
    """预测答案中是否有至少一个命中 gold"""
    if not gold_set:
        return 1.0 if not pred_set else 0.0
    return 1.0 if len(pred_set & gold_set) > 0 else 0.0


def compute_f1(pred_set, gold_set):
    """集合级别的 F1"""
    if not pred_set and not gold_set:
        return 1.0, 1.0, 1.0   # P, R, F1

    if not pred_set:
        return 0.0, 0.0, 0.0

    if not gold_set:
        return 0.0, 0.0, 0.0

    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gold_set) if gold_set else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def compute_exact_match(pred_set, gold_set):
    """预测集合是否完全等于 gold 集合"""
    return 1.0 if pred_set == gold_set else 0.0


# ============================================================
#  主评测逻辑
# ============================================================

def evaluate(pred_file, verbose=False, error_analysis=False):
    """
    读取 prediction JSONL，计算所有指标
    """
    # 读取数据
    with open(pred_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(data)} predictions from: {pred_file}")
    print("=" * 60)

    # 累计指标
    total = len(data)
    sum_hit1 = 0.0
    sum_f1 = 0.0
    sum_prec = 0.0
    sum_recall = 0.0
    sum_em = 0.0
    no_ans_count = 0        # 模型未给出任何答案
    totally_wrong = 0       # 模型给了答案但全错

    error_cases = []        # 错误 case 收集

    for idx, item in enumerate(data):
        # 获取 prediction 和 gold
        prediction_text = item.get('prediction', '')
        gold_raw = item.get('answer', item.get('a_entity', []))

        # 解析答案
        pred_answers = parse_prediction(prediction_text)
        gold_answers = parse_gold_answers(gold_raw)

        # 规范化
        pred_norm = set(normalize_answer(a) for a in pred_answers if normalize_answer(a))
        gold_norm = set(normalize_answer(a) for a in gold_answers if normalize_answer(a))

        # 计算指标
        hit1 = compute_hits_at_1(pred_norm, gold_norm)
        prec, recall, f1 = compute_f1(pred_norm, gold_norm)
        em = compute_exact_match(pred_norm, gold_norm)

        sum_hit1 += hit1
        sum_f1 += f1
        sum_prec += prec
        sum_recall += recall
        sum_em += em

        if not pred_norm:
            no_ans_count += 1
        elif hit1 == 0:
            totally_wrong += 1

        # verbose 模式
        if verbose and hit1 == 0:
            print(f"\n--- [WRONG] #{idx}  id={item.get('id', '?')} ---")
            print(f"  Question:   {item.get('question', '?')}")
            print(f"  Gold:       {gold_answers}")
            print(f"  Predicted:  {pred_answers}")
            print(f"  Gold(norm): {gold_norm}")
            print(f"  Pred(norm): {pred_norm}")

        # 错误分析收集
        if error_analysis and hit1 == 0:
            error_cases.append({
                'id': item.get('id', ''),
                'question': item.get('question', ''),
                'gold_answers': gold_answers,
                'pred_answers': pred_answers,
                'gold_norm': list(gold_norm),
                'pred_norm': list(pred_norm),
                'prediction_raw': prediction_text[:500],   # 截断避免太长
            })

    # 汇总
    results = {
        'total': total,
        'hits@1': sum_hit1 / total if total > 0 else 0,
        'macro_f1': sum_f1 / total if total > 0 else 0,
        'macro_precision': sum_prec / total if total > 0 else 0,
        'macro_recall': sum_recall / total if total > 0 else 0,
        'exact_match': sum_em / total if total > 0 else 0,
        'no_answer_count': no_ans_count,
        'no_answer_ratio': no_ans_count / total if total > 0 else 0,
        'totally_wrong': totally_wrong,
        'totally_wrong_ratio': totally_wrong / total if total > 0 else 0,
    }

    # 打印结果
    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Total samples:      {results['total']}")
    print(f"  ---")
    print(f"  Hits@1:             {results['hits@1']:.4f}  ({int(sum_hit1)}/{total})")
    print(f"  Macro F1:           {results['macro_f1']:.4f}")
    print(f"  Macro Precision:    {results['macro_precision']:.4f}")
    print(f"  Macro Recall:       {results['macro_recall']:.4f}")
    print(f"  Exact Match:        {results['exact_match']:.4f}")
    print(f"  ---")
    print(f"  No Answer:          {results['no_answer_count']}  ({results['no_answer_ratio']:.2%})")
    print(f"  Totally Wrong:      {results['totally_wrong']}  ({results['totally_wrong_ratio']:.2%})")
    print("=" * 60)

    # 保存错误分析
    if error_analysis and error_cases:
        error_file = pred_file.replace('.jsonl', '_errors.jsonl')
        with open(error_file, 'w', encoding='utf-8') as f:
            for case in error_cases:
                f.write(json.dumps(case, ensure_ascii=False) + '\n')
        print(f"\nError analysis saved to: {error_file}")
        print(f"Total error cases: {len(error_cases)}")

    # 保存结果摘要
    result_file = pred_file.replace('.jsonl', '_eval_results.json')
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {result_file}")

    return results


# ============================================================
#  入口
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Standalone KGQA Evaluation')
    parser.add_argument('--pred_file', type=str, required=True,
                        help='Path to prediction JSONL file')
    parser.add_argument('--verbose', action='store_true',
                        help='Print details for each wrong case')
    parser.add_argument('--error_analysis', action='store_true',
                        help='Save error cases to a separate file for analysis')

    args = parser.parse_args()

    if not os.path.exists(args.pred_file):
        print(f"Error: File not found: {args.pred_file}")
        exit(1)

    evaluate(args.pred_file, verbose=args.verbose, error_analysis=args.error_analysis)
