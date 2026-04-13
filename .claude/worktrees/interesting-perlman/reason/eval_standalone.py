"""
eval_standalone.py — Standalone evaluation with multiple modes

Modes:
  strict   — Exact set matching after normalization (original behavior)
  paper    — Paper-compatible: substring matching + double_check + date normalization
             (matches evaluate_results.py / evaluate_results_corrected.py logic)
  enhanced — Paper mode + token-overlap fuzzy matching (upper-bound analysis)

Usage:
    python eval_standalone.py --pred_file <path> --eval_mode strict
    python eval_standalone.py --pred_file <path> --eval_mode paper
    python eval_standalone.py --pred_file <path> --eval_mode enhanced
    python eval_standalone.py --pred_file <path> --eval_mode all        # report all modes
    python eval_standalone.py --pred_file <path> --verbose               # print wrong cases
    python eval_standalone.py --pred_file <path> --error_analysis        # save error cases
    python eval_standalone.py --pred_file <path> --breakdown             # error categorization
"""

import os
import re
import json
import string
import argparse
from collections import defaultdict


# ============================================================
#  Normalization functions
# ============================================================

def normalize_answer(text):
    """Strict normalization: lowercase, remove parens, articles, punctuation."""
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    return text.strip()


def normalize_paper(text):
    """Paper-compatible normalization: lowercase, remove articles/punctuation.
    Does NOT remove parenthetical content (preserves entity names)."""
    if text is None:
        return ""
    text = str(text).lower()
    exclude = set(string.punctuation)
    text = "".join(char for char in text if char not in exclude)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\b(<pad>)\b", " ", text)
    text = " ".join(text.split())
    return text.strip()


def normalize_date_answer(answer, question):
    """For date-type questions, extract year from 'YYYY-MM-DD' format."""
    q = question.lower()
    if 'when' in q or 'what year' in q:
        if '-' in answer and answer.split('-')[0].isdigit():
            return answer.split('-')[0]
    return answer


# ============================================================
#  Matching functions
# ============================================================

def match_substring(s1, s2):
    """Paper-compatible: is normalized(s2) a substring of normalized(s1)?"""
    n1 = normalize_paper(s1)
    n2 = normalize_paper(s2)
    if not n2:
        return False
    return n2 in n1


def should_double_check(question):
    """Determine if question type warrants double_check (reverse substring matching)."""
    q = question.lower()
    keywords = ['when', 'what year', 'which year', 'where', 'sport',
                "what countr", "language", 'nba finals', 'world series']
    return any(kw in q for kw in keywords)


def token_overlap_score(s1, s2):
    """Compute token overlap ratio between two strings (for fuzzy matching)."""
    t1 = set(normalize_paper(s1).split())
    t2 = set(normalize_paper(s2).split())
    if not t1 or not t2:
        return 0.0
    overlap = t1 & t2
    return len(overlap) / min(len(t1), len(t2))


# ============================================================
#  Answer parsing
# ============================================================

def parse_prediction(prediction_text):
    """Extract ans: lines. Returns list[str] (raw text after ans:, not normalized)."""
    if not prediction_text:
        return []
    answers = []
    for line in prediction_text.split('\n'):
        line = line.strip()
        m = re.match(r'^ans\s*:\s*(.+)', line, re.IGNORECASE)
        if m:
            ans = m.group(1).strip()
            if ans and ans.lower() not in ('not available', 'no information available',
                                           'n/a', 'none', 'unknown', 'unavailable',
                                           'not found'):
                answers.append(ans)
    return answers


def parse_prediction_lines(prediction_text):
    """Return full lines containing ans: (for paper-compatible substring matching)."""
    if not prediction_text:
        return []
    lines = []
    for line in prediction_text.split('\n'):
        if 'ans:' in line.lower() and 'none' not in line.lower():
            stripped = line.strip()
            if ("ans: not available" not in stripped.lower() and
                    "ans: no information available" not in stripped.lower()):
                lines.append(stripped)
    seen = set()
    result = []
    for l in lines:
        if l not in seen:
            result.append(l)
            seen.add(l)
    return result


def parse_gold_answers(answer_field):
    """Parse gold answers from various formats (list[str], list[dict], str)."""
    if answer_field is None:
        return []
    if isinstance(answer_field, str):
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
                for key in ['entity_name', 'answer', 'label', 'name']:
                    if key in item and item[key]:
                        results.append(str(item[key]))
                        break
                else:
                    if 'answer_id' in item:
                        results.append(str(item['answer_id']))
        return results
    return [str(answer_field)]


# ============================================================
#  Metric computation — STRICT mode (original behavior)
# ============================================================

def compute_hits_at_1_strict(pred_set, gold_set):
    if not gold_set:
        return 1.0 if not pred_set else 0.0
    return 1.0 if len(pred_set & gold_set) > 0 else 0.0


def compute_f1_strict(pred_set, gold_set):
    if not pred_set and not gold_set:
        return 1.0, 1.0, 1.0
    if not pred_set or not gold_set:
        return 0.0, 0.0, 0.0
    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set)
    recall = tp / len(gold_set)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def compute_exact_match(pred_set, gold_set):
    return 1.0 if pred_set == gold_set else 0.0


# ============================================================
#  Metric computation — PAPER mode (substring matching)
# ============================================================

def compute_hits_at_1_paper(pred_lines, gold_answers, question):
    """Paper-compatible Hit@1: check first prediction against all gold answers."""
    if not pred_lines:
        return 0.0
    double_check = should_double_check(question)
    first_pred = pred_lines[0]
    for gold in gold_answers:
        if match_substring(first_pred, gold):
            return 1.0
        if double_check:
            pred_text = first_pred.split('ans:')[-1].strip() if 'ans:' in first_pred else first_pred
            if match_substring(gold, pred_text):
                return 1.0
    return 0.0


def _match_pred_gold(pred_lines, gold_answers, question):
    """Shared matching logic for paper-mode precision/recall."""
    from copy import deepcopy
    pred_lines = deepcopy(pred_lines)
    pred_lines = sorted(pred_lines, key=len, reverse=True)
    double_check = should_double_check(question)
    matched = 0
    for gold in gold_answers:
        for pred in pred_lines:
            if match_substring(pred, gold):
                matched += 1
                pred_lines.remove(pred)
                break
            elif double_check:
                pred_text = pred.split('ans:')[-1].strip() if 'ans:' in pred else pred
                if match_substring(gold, pred_text) or match_substring(gold, pred):
                    matched += 1
                    pred_lines.remove(pred)
                    break
    return matched


def compute_f1_paper(pred_lines, gold_answers, question):
    """Paper-compatible F1."""
    num_pred = len(pred_lines)
    num_gold = len(gold_answers)
    if num_pred == 0 and num_gold == 0:
        return 1.0, 1.0, 1.0
    if num_pred == 0 or num_gold == 0:
        return 0.0, 0.0, 0.0
    matched = _match_pred_gold(pred_lines, gold_answers, question)
    precision = matched / num_pred
    recall = matched / num_gold
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


# ============================================================
#  Metric computation — ENHANCED mode (paper + fuzzy matching)
# ============================================================

FUZZY_THRESHOLD = 0.6


def compute_hits_at_1_enhanced(pred_answers, gold_answers, question):
    """Enhanced Hit@1: paper matching + token overlap fuzzy matching."""
    pred_lines = [f"ans: {a}" for a in pred_answers]
    hit = compute_hits_at_1_paper(pred_lines, gold_answers, question)
    if hit > 0:
        return 1.0
    for pred in pred_answers:
        for gold in gold_answers:
            if token_overlap_score(pred, gold) >= FUZZY_THRESHOLD:
                return 1.0
    return 0.0


def compute_f1_enhanced(pred_answers, gold_answers, question):
    """Enhanced F1 with fuzzy matching."""
    if not pred_answers and not gold_answers:
        return 1.0, 1.0, 1.0
    if not pred_answers or not gold_answers:
        return 0.0, 0.0, 0.0

    pred_remaining = list(pred_answers)
    gold_remaining = list(gold_answers)
    matched = 0

    # Pass 1: paper matching
    for gold in list(gold_remaining):
        for pred in list(pred_remaining):
            if match_substring(f"ans: {pred}", gold):
                matched += 1
                pred_remaining.remove(pred)
                gold_remaining.remove(gold)
                break
            elif should_double_check(question) and match_substring(gold, pred):
                matched += 1
                pred_remaining.remove(pred)
                gold_remaining.remove(gold)
                break

    # Pass 2: fuzzy matching on remaining
    for gold in list(gold_remaining):
        for pred in list(pred_remaining):
            if token_overlap_score(pred, gold) >= FUZZY_THRESHOLD:
                matched += 1
                pred_remaining.remove(pred)
                gold_remaining.remove(gold)
                break

    precision = matched / len(pred_answers)
    recall = matched / len(gold_answers)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


# ============================================================
#  Error categorization
# ============================================================

def categorize_error(pred_answers, gold_answers, prediction_text, question):
    """
    Categorize a wrong prediction:
    - 'no_answer': model gave no valid predictions
    - 'refusal': model explicitly refused
    - 'format_error': answer likely in text but not in ans: format
    - 'alias_mismatch': high token overlap but not exact match
    - 'wrong_entity': model gave unrelated entities
    """
    if not pred_answers:
        if prediction_text:
            text_lower = prediction_text.lower()
            refusal_words = ['not available', 'cannot determine', 'insufficient',
                             'unable to', 'no relevant', 'no information']
            if any(w in text_lower for w in refusal_words):
                return 'refusal'
            for gold in gold_answers:
                if normalize_paper(gold) in normalize_paper(prediction_text):
                    return 'format_error'
        return 'no_answer'

    best_overlap = 0.0
    for pred in pred_answers:
        for gold in gold_answers:
            overlap = token_overlap_score(pred, gold)
            best_overlap = max(best_overlap, overlap)

    if best_overlap >= 0.4:
        return 'alias_mismatch'
    return 'wrong_entity'


# ============================================================
#  Main evaluation
# ============================================================

def evaluate(pred_file, verbose=False, error_analysis=False,
             eval_mode='strict', breakdown=False):
    """
    Read prediction JSONL, compute metrics.
    eval_mode: 'strict' | 'paper' | 'enhanced' | 'all'
    """
    with open(pred_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(data)} predictions from: {pred_file}")
    print(f"Evaluation mode: {eval_mode}")
    print("=" * 60)

    modes_to_run = [eval_mode] if eval_mode != 'all' else ['strict', 'paper', 'enhanced']
    all_results = {}

    for mode in modes_to_run:
        total = len(data)
        sum_hit1 = 0.0
        sum_f1 = 0.0
        sum_prec = 0.0
        sum_recall = 0.0
        sum_em = 0.0
        no_ans_count = 0
        totally_wrong = 0
        error_cases = []
        error_categories = defaultdict(int)

        for idx, item in enumerate(data):
            prediction_text = item.get('prediction', '')
            gold_raw = item.get('answer', item.get('a_entity',
                       item.get('ground_truth', [])))
            question = item.get('question', '')

            pred_answers = parse_prediction(prediction_text)
            gold_answers = parse_gold_answers(gold_raw)

            # Date normalization for paper/enhanced modes
            if mode in ('paper', 'enhanced'):
                gold_answers = [normalize_date_answer(a, question) for a in gold_answers]

            if mode == 'strict':
                pred_norm = set(normalize_answer(a) for a in pred_answers if normalize_answer(a))
                gold_norm = set(normalize_answer(a) for a in gold_answers if normalize_answer(a))
                hit1 = compute_hits_at_1_strict(pred_norm, gold_norm)
                prec, recall, f1 = compute_f1_strict(pred_norm, gold_norm)
                em = compute_exact_match(pred_norm, gold_norm)

            elif mode == 'paper':
                gold_sorted = sorted(gold_answers, key=len, reverse=True)
                pred_lines = parse_prediction_lines(prediction_text)
                hit1 = compute_hits_at_1_paper(pred_lines, gold_sorted, question)
                prec, recall, f1 = compute_f1_paper(pred_lines, gold_sorted, question)
                pred_norm = set(normalize_paper(a) for a in pred_answers if normalize_paper(a))
                gold_norm = set(normalize_paper(a) for a in gold_answers if normalize_paper(a))
                em = compute_exact_match(pred_norm, gold_norm)

            elif mode == 'enhanced':
                gold_sorted = sorted(gold_answers, key=len, reverse=True)
                hit1 = compute_hits_at_1_enhanced(pred_answers, gold_sorted, question)
                prec, recall, f1 = compute_f1_enhanced(pred_answers, gold_sorted, question)
                pred_norm = set(normalize_paper(a) for a in pred_answers if normalize_paper(a))
                gold_norm = set(normalize_paper(a) for a in gold_answers if normalize_paper(a))
                em = compute_exact_match(pred_norm, gold_norm)

            sum_hit1 += hit1
            sum_f1 += f1
            sum_prec += prec
            sum_recall += recall
            sum_em += em

            if not pred_answers:
                no_ans_count += 1
            elif hit1 == 0:
                totally_wrong += 1

            if breakdown and hit1 == 0:
                cat = categorize_error(pred_answers, gold_answers, prediction_text, question)
                error_categories[cat] += 1

            if hit1 == 0:
                if verbose:
                    print(f"\n--- [WRONG] #{idx}  id={item.get('id', '?')} ---")
                    print(f"  Question:   {question}")
                    print(f"  Gold:       {gold_answers}")
                    print(f"  Predicted:  {pred_answers}")
                    if breakdown:
                        print(f"  Category:   {categorize_error(pred_answers, gold_answers, prediction_text, question)}")

                if error_analysis:
                    error_cases.append({
                        'id': item.get('id', ''),
                        'question': question,
                        'gold_answers': gold_answers,
                        'pred_answers': pred_answers,
                        'prediction_raw': prediction_text[:500],
                        'category': categorize_error(pred_answers, gold_answers, prediction_text, question) if breakdown else '',
                    })

        results = {
            'eval_mode': mode,
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
        if breakdown:
            results['error_breakdown'] = dict(error_categories)

        # Print results
        print(f"\n{'=' * 60}")
        print(f"  EVALUATION RESULTS  [{mode.upper()} mode]")
        print(f"{'=' * 60}")
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

        if breakdown and error_categories:
            print(f"  ---")
            print(f"  Error Breakdown:")
            for cat, count in sorted(error_categories.items(), key=lambda x: -x[1]):
                ratio = count / total
                print(f"    {cat:20s}: {count:5d}  ({ratio:.2%})")
        print("=" * 60)

        if error_analysis and error_cases:
            suffix = f'_errors_{mode}' if eval_mode == 'all' else '_errors'
            error_file = pred_file.replace('.jsonl', f'{suffix}.jsonl')
            with open(error_file, 'w', encoding='utf-8') as f:
                for case in error_cases:
                    f.write(json.dumps(case, ensure_ascii=False) + '\n')
            print(f"Error analysis saved to: {error_file}")

        all_results[mode] = results

    # Save results
    result_file = pred_file.replace('.jsonl', f'_eval_results_{eval_mode}.json')
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {result_file}")

    # Comparison table for 'all' mode
    if eval_mode == 'all' and len(all_results) > 1:
        print(f"\n{'=' * 70}")
        print("  COMPARISON ACROSS MODES")
        print(f"{'=' * 70}")
        print(f"  {'Metric':<20s} {'Strict':>10s} {'Paper':>10s} {'Enhanced':>10s}")
        print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
        for metric in ['hits@1', 'macro_f1', 'macro_precision', 'macro_recall', 'exact_match']:
            vals = []
            for m in ['strict', 'paper', 'enhanced']:
                if m in all_results:
                    vals.append(f"{all_results[m][metric]:.4f}")
                else:
                    vals.append("  N/A   ")
            print(f"  {metric:<20s} {vals[0]:>10s} {vals[1]:>10s} {vals[2]:>10s}")
        print(f"  {'No Answer':<20s}", end="")
        for m in ['strict', 'paper', 'enhanced']:
            if m in all_results:
                print(f" {all_results[m]['no_answer_count']:>10d}", end="")
        print()
        print(f"  {'Totally Wrong':<20s}", end="")
        for m in ['strict', 'paper', 'enhanced']:
            if m in all_results:
                print(f" {all_results[m]['totally_wrong']:>10d}", end="")
        print()
        print(f"{'=' * 70}")

    return all_results.get(modes_to_run[0], all_results)


# ============================================================
#  Entry point
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Standalone KGQA Evaluation')
    parser.add_argument('--pred_file', type=str, required=True,
                        help='Path to prediction JSONL file')
    parser.add_argument('--eval_mode', type=str, default='strict',
                        choices=['strict', 'paper', 'enhanced', 'all'],
                        help='Evaluation mode (default: strict)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print details for each wrong case')
    parser.add_argument('--error_analysis', action='store_true',
                        help='Save error cases to a separate file')
    parser.add_argument('--breakdown', action='store_true',
                        help='Categorize errors by type')

    args = parser.parse_args()

    if not os.path.exists(args.pred_file):
        print(f"Error: File not found: {args.pred_file}")
        exit(1)

    evaluate(args.pred_file, verbose=args.verbose, error_analysis=args.error_analysis,
             eval_mode=args.eval_mode, breakdown=args.breakdown)
