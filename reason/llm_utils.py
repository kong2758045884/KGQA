import os
import re
import time
import openai
from openai import OpenAI
from prompts import (
    icl_user_prompt,
    icl_ass_prompt,
    dc_fallback_prompt,
    answer_verify_sys_prompt,
    answer_verify_instruction,
)


# ============================================================
#  Refusal detection patterns
# ============================================================

REFUSAL_PATTERNS = [
    "ans: not available",
    "ans: no information available",
    "ans: n/a",
    "ans: none",
    "ans: unknown",
    "cannot determine",
    "cannot be determined",
    "no relevant information",
    "insufficient information",
    "unable to determine",
    "unable to answer",
    "not enough information",
    "no answer can be provided",
    "information is not available",
    "does not provide enough",
    "do not have enough",
    "triplets do not contain",
    "triplets do not provide",
    "no direct answer",
]


def is_refusal(text):
    """Check if LLM output is a refusal to answer."""
    text_lower = text.lower()
    return any(p in text_lower for p in REFUSAL_PATTERNS)


def has_valid_ans_lines(text):
    """Check if text contains at least one valid ans: line (not a refusal)."""
    for line in text.split('\n'):
        line = line.strip()
        m = re.match(r'^ans\s*:\s*(.+)', line, re.IGNORECASE)
        if m:
            ans = m.group(1).strip().lower()
            if ans not in ('not available', 'no information available', 'n/a',
                           'none', 'unknown', 'not found', 'unavailable'):
                return True
    return False


# ============================================================
#  Output post-processing: recover ans: lines from free text
# ============================================================

def postprocess_output(text):
    """
    If the output already has valid ans: lines, return as-is.
    Otherwise, try to extract answers from free-text patterns and
    append them as ans: lines.
    """
    if has_valid_ans_lines(text):
        return text

    recovered = []

    # Pattern 1: "The answer is/are: X" or "The answer is X"
    for m in re.finditer(
        r'(?:the\s+)?answer(?:s)?\s+(?:is|are)\s*[:\-]?\s*(.+)',
        text, re.IGNORECASE
    ):
        candidate = m.group(1).strip().rstrip('.')
        if candidate and len(candidate) < 200:
            parts = re.split(r'\s*(?:,\s*and\s*|,\s*|\s+and\s+)', candidate)
            for p in parts:
                p = p.strip().rstrip('.')
                if p and p.lower() not in ('not available', 'unknown', 'n/a'):
                    recovered.append(f"ans: {p}")

    # Pattern 2: "Therefore/Thus/So, X" at end of reasoning
    if not recovered:
        lines = text.strip().split('\n')
        for line in lines[-5:]:
            m = re.match(
                r'^(?:therefore|thus|so|hence|in conclusion)[,:]?\s*(.+)',
                line.strip(), re.IGNORECASE
            )
            if m:
                candidate = m.group(1).strip().rstrip('.')
                if candidate and len(candidate) < 200 and 'ans:' not in candidate.lower():
                    recovered.append(f"ans: {candidate}")

    # Pattern 3: Numbered/bulleted list items (only if >= 2 found)
    if not recovered:
        list_items = []
        for line in text.split('\n'):
            line = line.strip()
            m = re.match(r'^(?:\d+[.)]\s+|[-*]\s+)(.+)', line)
            if m:
                item = m.group(1).strip().rstrip('.')
                if item and len(item) < 200:
                    list_items.append(item)
        if len(list_items) >= 2:
            for item in list_items:
                recovered.append(f"ans: {item}")

    if recovered:
        return text + "\n" + "\n".join(recovered)
    return text


# ============================================================
#  LLM initialization
# ============================================================

def llm_init(
    model_name,
    tensor_parallel_size=1,
    max_seq_len_to_capture=8192,
    max_tokens=4000,
    seed=0,
    temperature=0,
    frequency_penalty=0,
):
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing API key. Please set DASHSCOPE_API_KEY or OPENAI_API_KEY."
        )

    base_url = os.getenv(
        "OPENAI_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    def llm(*, messages):
        kwargs = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if frequency_penalty is not None:
            kwargs["frequency_penalty"] = frequency_penalty
        return client.chat.completions.create(**kwargs)

    return llm


def get_outputs(outputs, model_name):
    content = outputs.choices[0].message.content
    return content if content is not None else ""


# ============================================================
#  Inference logic
# ============================================================

def llm_inf(llm, prompts, mode, model_name,
            disable_refusal_handling=False,
            disable_postprocess=False,
            disable_dc_fallback=False,
            icl_ass_prompt_override=None):
    # Ablation switches (default == current improved behavior):
    #   disable_refusal_handling: skip refusal-specific trigger for DC fallback
    #   disable_postprocess:      skip postprocess_output on LLM outputs
    #   disable_dc_fallback:      skip the entire DC fallback second-round path
    #   icl_ass_prompt_override:  if not None, replaces module-level icl_ass_prompt
    #                             (used by --use_baseline_prompts)
    res = []
    conversation = []
    outputs = ""
    ass_text = icl_ass_prompt_override if icl_ass_prompt_override is not None else icl_ass_prompt

    if "sys" in mode:
        conversation.append({"role": "system", "content": prompts["sys_query"]})

    if "icl" in mode:
        conversation.append({"role": "user", "content": icl_user_prompt})
        conversation.append({"role": "assistant", "content": ass_text})

    if "sys" in mode:
        conversation.append({"role": "user", "content": prompts["user_query"]})
        outputs = get_outputs(llm(messages=conversation), model_name)
        # Post-process to recover format issues
        if not disable_postprocess:
            outputs = postprocess_output(outputs)
        res.append(outputs)

    if "sys_cot" in mode:
        if "clear" in mode:
            conversation = []

        conversation.append({"role": "assistant", "content": outputs})
        conversation.append({"role": "user", "content": prompts["cot_query"]})
        outputs = get_outputs(llm(messages=conversation), model_name)
        if not disable_postprocess:
            outputs = postprocess_output(outputs)
        res.append(outputs)

    elif "dc" in mode:
        if not disable_dc_fallback:
            # DC fallback: trigger second round if first round has no valid answer.
            # The refusal-specific trigger is gated separately so it can be ablated
            # without breaking the empty/invalid-answer triggers.
            needs_fallback = (
                len(res) == 0
                or not has_valid_ans_lines(res[0])
                or ((not disable_refusal_handling) and is_refusal(res[0]))
            )

            if needs_fallback:
                # Use the stronger dc_fallback_prompt instead of regular cot_query
                fallback_prompt = prompts.get("dc_query", dc_fallback_prompt)
                conversation.append({"role": "assistant", "content": outputs})
                conversation.append({"role": "user", "content": fallback_prompt})
                outputs = get_outputs(llm(messages=conversation), model_name)
                if not disable_postprocess:
                    outputs = postprocess_output(outputs)
                if len(res) == 0:
                    res.append(outputs)
                else:
                    res[0] = outputs
        res.append("")
    else:
        res.append("")

    return res


def llm_inf_with_retry(llm, each_qa, llm_mode, model_name, max_retries,
                       disable_refusal_handling=False,
                       disable_postprocess=False,
                       disable_dc_fallback=False,
                       icl_ass_prompt_override=None):
    retries = 0
    while retries < max_retries:
        try:
            return llm_inf(llm, each_qa, llm_mode, model_name,
                           disable_refusal_handling=disable_refusal_handling,
                           disable_postprocess=disable_postprocess,
                           disable_dc_fallback=disable_dc_fallback,
                           icl_ass_prompt_override=icl_ass_prompt_override)
        except openai.RateLimitError:
            wait_time = (2 ** retries) * 5
            print(f"Rate limit error encountered. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            retries += 1

    raise Exception("Max retries exceeded. Please check your rate limits or try again later.")


def llm_inf_all(llm, each_qa, llm_mode, model_name, max_retries=5,
                disable_refusal_handling=False,
                disable_postprocess=False,
                disable_dc_fallback=False,
                icl_ass_prompt_override=None):
    return llm_inf_with_retry(llm, each_qa, llm_mode, model_name, max_retries,
                              disable_refusal_handling=disable_refusal_handling,
                              disable_postprocess=disable_postprocess,
                              disable_dc_fallback=disable_dc_fallback,
                              icl_ass_prompt_override=icl_ass_prompt_override)


# ============================================================
#  New answer-stage baselines (controlled by CLI flags in main.py):
#    - prompt_only          : only improved prompting kept; DC/refusal/postprocess OFF
#    - retry_once           : one plain retry on refusal/NA/empty; no DC fallback
#    - answer_then_verify   : two-stage (candidate -> lightweight verify pass)
#  These are invoked ONLY when the corresponding --enable_* flag is passed;
#  otherwise the original llm_inf_all path is used unchanged.
# ============================================================

def _with_rate_limit_retry(fn, max_retries=5):
    """Run `fn()` and retry only on openai.RateLimitError with exponential backoff."""
    retries = 0
    while retries < max_retries:
        try:
            return fn()
        except openai.RateLimitError:
            wait_time = (2 ** retries) * 5
            print(f"Rate limit error encountered. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            retries += 1
    raise Exception("Max retries exceeded. Please check your rate limits or try again later.")


def _needs_retry(output):
    """True if the output is empty, a refusal, or has no valid ans: lines."""
    if output is None:
        return True
    if not output.strip():
        return True
    if is_refusal(output):
        return True
    if not has_valid_ans_lines(output):
        return True
    return False


def _llm_inf_prompt_only_core(llm, each_qa, llm_mode, model_name,
                              icl_ass_prompt_override=None):
    """Prompt-only baseline: keep improved prompting, disable the rest.

    Behavioural enhancements that are NOT part of prompting are turned off:
      - DC fallback (second-round stronger re-ask)
      - refusal handling trigger
      - postprocess_output (free-text -> ans-line recovery)
    This isolates the contribution of the improved prompt text alone.
    """
    return llm_inf(
        llm, each_qa, llm_mode, model_name,
        disable_refusal_handling=True,
        disable_postprocess=True,
        disable_dc_fallback=True,
        icl_ass_prompt_override=icl_ass_prompt_override,
    )


def _llm_inf_retry_once_core(llm, each_qa, llm_mode, model_name,
                             icl_ass_prompt_override=None):
    """Retry-once baseline: one plain retry on refusal / NA / empty output.

    Deliberately NOT using:
      - DC fallback (different stronger prompt)
      - improved refusal handling's full chain
      - postprocess recovery
    It only retries the exact same request ONCE and returns whichever of the
    two outputs has valid ans: lines (falling back to the second attempt).
    """
    res = llm_inf(
        llm, each_qa, llm_mode, model_name,
        disable_refusal_handling=True,
        disable_postprocess=True,
        disable_dc_fallback=True,
        icl_ass_prompt_override=icl_ass_prompt_override,
    )
    first_output = res[0] if res else ""

    if _needs_retry(first_output):
        res2 = llm_inf(
            llm, each_qa, llm_mode, model_name,
            disable_refusal_handling=True,
            disable_postprocess=True,
            disable_dc_fallback=True,
            icl_ass_prompt_override=icl_ass_prompt_override,
        )
        second_output = res2[0] if res2 else ""
        chosen = second_output if has_valid_ans_lines(second_output) else (
            first_output if has_valid_ans_lines(first_output) else second_output
        )
        if res:
            res[0] = chosen
        else:
            res = [chosen]
    return res


def _llm_inf_answer_then_verify_core(llm, each_qa, llm_mode, model_name,
                                     icl_ass_prompt_override=None):
    """Answer-then-Verify baseline.

    Stage 1: generate a candidate answer with improved prompting (no DC,
             no refusal handling, no postprocess).
    Stage 2: a single lightweight verification turn that can:
               (a) keep the original answer,
               (b) correct it using only the given triplets, or
               (c) mark as "ans: not available".
    No DC fallback, no retry loop, no recursion.
    """
    res = llm_inf(
        llm, each_qa, llm_mode, model_name,
        disable_refusal_handling=True,
        disable_postprocess=True,
        disable_dc_fallback=True,
        icl_ass_prompt_override=icl_ass_prompt_override,
    )
    candidate = res[0] if res else ""

    # Build a single verifier turn: question + triplets + candidate.
    # `user_query` already contains "Triplets:\n...\n\nQuestion:\n..." (or its
    # variant for firstq/noevi modes), produced by get_prompts().
    user_query = each_qa.get("user_query", "")
    verifier_user_msg = (
        f"{user_query}\n\n"
        f"Candidate answer:\n{candidate.strip() if candidate else '(empty)'}\n\n"
        f"{answer_verify_instruction}"
    )
    messages = [
        {"role": "system", "content": answer_verify_sys_prompt},
        {"role": "user", "content": verifier_user_msg},
    ]
    verified = get_outputs(llm(messages=messages), model_name)

    if res:
        res[0] = verified
    else:
        res = [verified]
    return res


def llm_inf_all_baseline(llm, each_qa, llm_mode, model_name,
                         baseline_mode,
                         max_retries=5,
                         icl_ass_prompt_override=None):
    """Dispatch entry point for the three new answer-stage baselines.

    `baseline_mode` must be one of:
        "prompt_only", "retry_once", "answer_then_verify"
    Rate-limit retries are applied around the whole baseline call.
    """
    if baseline_mode == "prompt_only":
        core = _llm_inf_prompt_only_core
    elif baseline_mode == "retry_once":
        core = _llm_inf_retry_once_core
    elif baseline_mode == "answer_then_verify":
        core = _llm_inf_answer_then_verify_core
    else:
        raise ValueError(f"Unknown baseline_mode: {baseline_mode!r}")

    return _with_rate_limit_retry(
        lambda: core(llm, each_qa, llm_mode, model_name,
                     icl_ass_prompt_override=icl_ass_prompt_override),
        max_retries=max_retries,
    )
