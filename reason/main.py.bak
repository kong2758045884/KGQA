import os
import json
import wandb
import random
import argparse
from tqdm import tqdm
from pathlib import Path

from preprocess.prepare_data import get_data
from preprocess.prepare_prompts import get_prompts_for_data
from llm_utils import llm_init, llm_inf_all


def get_defined_prompts(prompt_mode, model_name, llm_mode):
    if 'gpt' in model_name or 'gpt' in prompt_mode:
        if 'gptLabel' in prompt_mode:
            from prompts import sys_prompt_gpt, cot_prompt_gpt
            return sys_prompt_gpt, cot_prompt_gpt
        else:
            from prompts import icl_sys_prompt, icl_cot_prompt
            return icl_sys_prompt, icl_cot_prompt
    elif 'noevi' in prompt_mode:
        from prompts import noevi_sys_prompt, noevi_cot_prompt
        return noevi_sys_prompt, noevi_cot_prompt
    elif 'icl' in llm_mode:
        from prompts import icl_sys_prompt, icl_cot_prompt
        return icl_sys_prompt, icl_cot_prompt
    else:
        from prompts import sys_prompt, cot_prompt
        return sys_prompt, cot_prompt


def save_checkpoint(file_handle, data):
    file_handle.write(json.dumps(data) + "\n")
    file_handle.flush()          # 每条写完立即刷盘，防止中断丢数据


def load_checkpoint(file_path):
    if os.path.exists(file_path):
        print("*" * 50)
        print(f"Resuming from {file_path}")
        with open(file_path, "r") as f:
            ckpt = [json.loads(line) for line in f]
        try:
            print(f"Last processed item: {ckpt[-1]['id']}")
        except IndexError:
            pass
        print(f"Already completed: {len(ckpt)} items")
        print("*" * 50)
        return ckpt
    return []


def main():
    parser = argparse.ArgumentParser(description="RAG for KGQA")
    parser.add_argument("-d", "--dataset_name", type=str, default="cwq", help="Dataset name")
    parser.add_argument("--prompt_mode", type=str, default="scored_100", help="Prompt mode")
    parser.add_argument("-p", "--score_dict_path", type=str)
    parser.add_argument("--llm_mode", type=str, default="sys_icl_dc", help="LLM mode")
    parser.add_argument("-m", "--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="Model name")
    parser.add_argument("--split", type=str, default="test", help="Split")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--max_seq_len_to_capture", type=int, default=8192 * 2, help="Max sequence length to capture")
    parser.add_argument("--max_tokens", type=int, default=4000, help="Max tokens")
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument("--temperature", type=float, default=0, help="Temperature")
    parser.add_argument("--frequency_penalty", type=float, default=0.16, help="Frequency penalty")
    parser.add_argument("--thres", type=float, default=0.0, help="Threshold")

    # ==================== 新增参数 ====================
    parser.add_argument("--pilot", type=int, default=0,
                        help="Pilot 模式：只跑前 N 条。设为 0 表示跑全量（默认）")
    parser.add_argument("--no_wandb", action="store_true",
                        help="禁用 wandb，本地调试时使用")
    # ==================================================

    args = parser.parse_args()
    dataset_name = args.dataset_name
    prompt_mode = args.prompt_mode
    llm_mode = args.llm_mode
    model_name = args.model_name
    split = args.split
    tensor_parallel_size = args.tensor_parallel_size
    max_seq_len_to_capture = args.max_seq_len_to_capture
    max_tokens = args.max_tokens
    seed = args.seed
    temperature = args.temperature
    frequency_penalty = args.frequency_penalty
    thres = args.thres

    pred_file_path = f"./results/KGQA/{dataset_name}/RoG/{split}/results_gen_rule_path_RoG-{dataset_name}_RoG_{split}_predictions_3_False_jsonl/predictions.jsonl"
    run_name = f"{model_name}-{prompt_mode}-{llm_mode}-{frequency_penalty}-thres_{thres}-{split}"

    # ==================== wandb 开关 ====================
    if args.no_wandb:
        run = wandb.init(mode="disabled")
    else:
        run = wandb.init(project=f"RAG-{dataset_name}", name=run_name, config=args)
    # ====================================================

    if args.score_dict_path is None:
        if dataset_name == "webqsp":
            assert split == "test"
            score_dict_path = "./scored_triples/webqsp_240912_unidir_test.pth"
        elif dataset_name == "cwq":
            assert split == "test"
            score_dict_path = "./scored_triples/cwq_240907_unidir_test.pth"
    else:
        score_dict_path = args.score_dict_path

    raw_pred_folder_path = Path(f"./results/KGQA/{dataset_name}/SubgraphRAG/{args.model_name.split('/')[-1]}")
    raw_pred_folder_path.mkdir(parents=True, exist_ok=True)
    raw_pred_file_path = raw_pred_folder_path / f"{prompt_mode}-{llm_mode}-{frequency_penalty}-thres_{thres}-{split}-predictions-resume.jsonl"

    llm = llm_init(model_name, tensor_parallel_size, max_seq_len_to_capture, max_tokens, seed, temperature, frequency_penalty)
    data = get_data(dataset_name, pred_file_path, score_dict_path, split, prompt_mode)
    sys_prompt, cot_prompt = get_defined_prompts(prompt_mode, model_name, llm_mode)

    print("Generating prompts...")
    data = get_prompts_for_data(data, prompt_mode, sys_prompt, cot_prompt, thres)

    # ==================== pilot 控制 ====================
    if args.pilot > 0:
        data = data[:args.pilot]
        print(f"[PILOT MODE] 只跑前 {args.pilot} 条")
    else:
        print(f"[FULL MODE] 跑全量 {len(data)} 条")
    # ====================================================

    print("Starting inference...")
    start_idx = len(load_checkpoint(raw_pred_file_path))

    if start_idx >= len(data):
        print(f"All {len(data)} items already completed, skipping inference.")
    else:
        with open(raw_pred_file_path, "a") as pred_file:
            for idx, each_qa in enumerate(tqdm(data[start_idx:], initial=start_idx, total=len(data))):
                res = llm_inf_all(llm, each_qa, llm_mode, model_name)

                del each_qa["graph"], each_qa["good_paths_rog"], each_qa["good_triplets_rog"], each_qa["scored_triplets"]

                each_qa["prediction"] = res[0]
                save_checkpoint(pred_file, each_qa)

    final_pred_file_path = raw_pred_file_path.with_name(
        raw_pred_file_path.stem.replace("-resume", "") + raw_pred_file_path.suffix
    )

    if raw_pred_file_path.exists():
        if final_pred_file_path.exists():
            final_pred_file_path.unlink()   # 避免 rename 冲突
        os.rename(raw_pred_file_path, final_pred_file_path)

    print("=" * 50)
    print(f"Predictions saved to: {final_pred_file_path}")
    print(f"Total samples: {len(data)}")
    print("=" * 50)
    print(f"接下来请用 eval_standalone.py 评测：")
    print(f"  python eval_standalone.py --pred_file {final_pred_file_path}")
    print("=" * 50)

    run.finish()


if __name__ == "__main__":
    main()
