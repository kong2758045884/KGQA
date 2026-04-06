import os
import time
import openai
from openai import OpenAI
from prompts import icl_user_prompt, icl_ass_prompt


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


def llm_inf(llm, prompts, mode, model_name):
    res = []
    conversation = []
    outputs = ""

    if "sys" in mode:
        conversation.append({"role": "system", "content": prompts["sys_query"]})

    if "icl" in mode:
        conversation.append({"role": "user", "content": icl_user_prompt})
        conversation.append({"role": "assistant", "content": icl_ass_prompt})

    if "sys" in mode:
        conversation.append({"role": "user", "content": prompts["user_query"]})
        outputs = get_outputs(llm(messages=conversation), model_name)
        res.append(outputs)

    if "sys_cot" in mode:
        if "clear" in mode:
            conversation = []

        conversation.append({"role": "assistant", "content": outputs})
        conversation.append({"role": "user", "content": prompts["cot_query"]})
        outputs = get_outputs(llm(messages=conversation), model_name)
        res.append(outputs)

    elif "dc" in mode:
        if (
            len(res) == 0
            or "ans:" not in res[0].lower()
            or "ans: not available" in res[0].lower()
            or "ans: no information available" in res[0].lower()
        ):
            conversation.append({"role": "user", "content": prompts["cot_query"]})
            outputs = get_outputs(llm(messages=conversation), model_name)
            if len(res) == 0:
                res.append(outputs)
            else:
                res[0] = outputs
        res.append("")
    else:
        res.append("")

    return res


def llm_inf_with_retry(llm, each_qa, llm_mode, model_name, max_retries):
    retries = 0
    while retries < max_retries:
        try:
            return llm_inf(llm, each_qa, llm_mode, model_name)
        except openai.RateLimitError:
            wait_time = (2 ** retries) * 5
            print(f"Rate limit error encountered. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            retries += 1

    raise Exception("Max retries exceeded. Please check your rate limits or try again later.")


def llm_inf_all(llm, each_qa, llm_mode, model_name, max_retries=5):
    return llm_inf_with_retry(llm, each_qa, llm_mode, model_name, max_retries)