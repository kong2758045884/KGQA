import os
import re
import time
import openai
from openai import OpenAI
from prompts import icl_user_prompt, icl_ass_prompt, dc_fallback_prompt


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
