import argparse
import json
import os
from itertools import islice
from typing import Iterable, List

import torch
from tqdm.auto import tqdm

import logging

# Reduce noisy debug logs from vLLM/its worker processes. Adjust level to
# `logging.WARNING` or `logging.ERROR` to be more silent.
logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)
logging.getLogger("vllm").setLevel(logging.INFO)
logging.getLogger("vllm.engine").setLevel(logging.INFO)

# Also set the vLLM-specific environment variable so worker processes
# don't emit DEBUG logs. Set to WARNING to suppress INFO/DEBUG messages.
import os
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

from vllm import LLM, SamplingParams

def read_prompts_from_file(path: str) -> List[str]:
    prompts = []
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    # support either {'text': ...} or {'prompt': ...}
                    text = obj.get("text") or obj.get("prompt") or obj.get("input")
                    if text is None:
                        # fallback to whole object repr
                        text = json.dumps(obj, ensure_ascii=False)
                except Exception:
                    text = line
                prompts.append(text)
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    prompts.append(line)

    return prompts


def batched(iterable: Iterable, n: int):
    it = iter(iterable)
    while True:
        chunk = list(islice(it, n))
        if not chunk:
            break
        yield chunk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", "-i",default="data/imdb_test.jsonl", required=False, help="Path to prompts file (text or .jsonl)")
    parser.add_argument("--hf-dataset", default="imdb", help="Hugging Face dataset id (e.g. 'imdb' or 'squad') to load and save to data/")
    parser.add_argument("--dataset-split", default="train", help="Dataset split to use when loading HF dataset")
    parser.add_argument("--dataset-column", default="text", help="Column name in dataset to use as prompt text")
    parser.add_argument("--max-samples", type=int, default=0, help="Max samples to pull from dataset (0 = all)")
    parser.add_argument("--output-file", "-o", default="outputs.jsonl", help="JSONL output file")
    parser.add_argument("--batch-size", "-b", type=int, default=16)
    # parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--data-len", type=int, default=16)
    args = parser.parse_args()

    # If a Hugging Face dataset is provided, load it and write to `data/`.
    if args.hf_dataset:
        try:
            from datasets import load_dataset
        except Exception as e:
            raise SystemExit("Please install the `datasets` library (pip install datasets)") from e

        ds = load_dataset(args.hf_dataset)
        split = args.dataset_split
        if split not in ds:
            # fall back to first split
            split = list(ds.keys())[0]

        table = ds[split]
        max_samples = args.max_samples or len(table)

        os.makedirs("data", exist_ok=True)
        safe_name = args.hf_dataset.replace("/", "_")
        save_path = os.path.join("data", f"{safe_name}_{split}.jsonl")
        with open(save_path, "w", encoding="utf-8") as sf:
            for i, item in enumerate(table):
                if i >= max_samples:
                    break
                text = item.get(args.dataset_column) or item.get("text") or item.get("prompt") or ""
                if isinstance(text, list):
                    text = " ".join(map(str, text))
                json.dump({"text": text}, sf, ensure_ascii=False)
                sf.write("\n")

        prompts = read_prompts_from_file(save_path)
    else:
        if not args.input_file:
            raise SystemExit("Provide --input-file or --hf-dataset")
        prompts = read_prompts_from_file(args.input_file)

    if not prompts:
        raise SystemExit("No prompts found in input file or dataset")
    # print(args.input_file, args.data_len)

    # exit()
    os.environ.setdefault("VLLM_USE_V1", "1")
    # print(prompts[:2])
    llm = LLM(
        model=args.model,
        trust_remote_code=args.trust_remote_code,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_cudagraph_capture_size=32,
    )

    out_f = open(args.output_file, "w", encoding="utf-8")
    global_index = 0
    max_tokens = 64
    test_sequence = [
    {
        'data_len': 32,
        'batch_size': 32,
        "max_tokens": max_tokens
    },
    {
        'data_len': 16,
        'batch_size': 16,
        "max_tokens": max_tokens
    },
    {
        'data_len': 10,
        'batch_size': 10,
        "max_tokens": max_tokens
    },
    {
        'data_len': 9,
        'batch_size': 9,
        "max_tokens": max_tokens
    },
    {
        'data_len': 8,
        'batch_size': 8,
        "max_tokens": max_tokens
    },
    {
        'data_len': 7,
        'batch_size': 7,
        "max_tokens": max_tokens
    },
    {
        'data_len': 6,
        'batch_size': 6,
        "max_tokens": max_tokens
    },
    {
        'data_len': 5,
        'batch_size': 5,
        "max_tokens": max_tokens
    },
    {
        'data_len': 4,
        'batch_size': 4,
        "max_tokens": max_tokens
    },
    {
        'data_len': 2,
        'batch_size': 2,
        "max_tokens": max_tokens
    },
    {
        'data_len': 1,
        'batch_size': 1,
        "max_tokens": max_tokens
    },
    {
        'data_len': 24,
        'batch_size': 24,
        "max_tokens": max_tokens
    },
    ]
    csv_path = f"logs"
    json_test_sequence = open(f"{csv_path}/test_{os.getpid()}", "w", encoding="utf-8")
    json.dump(test_sequence, json_test_sequence, ensure_ascii=False)
    for test in test_sequence:
        batch_size = test['batch_size']
        data_len = test['data_len']
        max_tokens = test['max_tokens']
        test_prompts = prompts[: int(data_len)] if int(data_len) > 0 else prompts
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=max_tokens)
        print(test_prompts)
        print(f"Running batch inference on {len(test_prompts)} prompts...")
        try:
            for batch in tqdm(list(batched(test_prompts, batch_size)), desc="Batches"):
                # sync before run for more accurate NVTX boundaries if user cares
                import time
                start_time = time.time()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.nvtx.range_push(f"vLLM_Batch_Generation_BS{batch_size}")
                
                outputs = llm.generate(batch, sampling_params)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.nvtx.range_pop()
                end_time = time.time()
                print(f"Batch of size {len(batch)} took {end_time - start_time:.2f} seconds")
                for input, out in zip(batch, outputs):
                    idx = global_index
                    global_index += 1
                    # try to extract text; fallback to str()
                    text = getattr(out, "text", None)
                    if text is None:
                        try:
                            text = out.completion_text  # some versions
                        except Exception:
                            text = out.outputs
                    
                    # Extract number of generated tokens
                    num_tokens = 0
                    try:
                        if hasattr(out, 'outputs') and len(out.outputs) > 0:
                            num_tokens = len(out.outputs[0].token_ids)
                    except Exception:
                        pass
                    
                    print(f"idx: {idx} | generated_tokens: {num_tokens}\ninput: {input}\noutput: {text}")
                    # record = {"index": idx, "input": input}
                    # json.dump(record, out_f, ensure_ascii=False)
                    # out_f.write("\n")
                    # record = {"index": idx, "output": text}
                    # json.dump(record, out_f, ensure_ascii=False)
                    # out_f.write("\n")
            print(f"Wrote outputs to {args.output_file}")
        finally:
            out_f.close()


if __name__ == "__main__":
    main()
