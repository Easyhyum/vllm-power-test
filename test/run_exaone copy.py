import torch
import os
from vllm import LLM, SamplingParams
# ObservabilityConfig 임포트 제거 (에러 방지)

# 1. 환경 변수 설정 (LLM 초기화 전)
os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_RPC_TIMEOUT"] = "100000"
# vLLM 내부 커널 및 레이어 NVTX 트레이싱 강제 활성화
os.environ["VLLM_ENABLE_LAYERWISE_NVTX"] = "1" 

prompts = ["Please write a simple html code to introduce Java the programming language."]
MAX_OUTPUT_TOKENS = 128
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=MAX_OUTPUT_TOKENS)

# 2. LLM 초기화 (문제가 된 observability_config 제거)
llm = LLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    trust_remote_code=True,
    gpu_memory_utilization=0.4,
    max_model_len=4096 
)

print("Start Profiling & Generation...")
torch.cuda.synchronize()

# 3. NVTX 구간 마킹
torch.cuda.nvtx.range_push("vLLM_Generation_Run")

outputs = llm.generate(prompts, sampling_params)

torch.cuda.nvtx.range_pop()

for output in outputs:
    print(output)