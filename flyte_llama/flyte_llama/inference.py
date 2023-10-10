"""
envd build -f :serving --output type=image,name=ghcr.io/unionai-oss/modelz-flyte-llama-serving:v0,push=true
"""

import os
from dataclasses import dataclass
from io import BytesIO
from typing import List

import torch  # type: ignore
from peft import (
    LoraConfig,
    get_peft_model,
)
import huggingface_hub as hh
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from mosec import Server, Worker, get_logger
from mosec.mixin import MsgpackMixin

logger = get_logger()


@dataclass
class ServingConfig:
    model_path: str
    adapter_path: str
    model_max_length: int = 1024
    padding: str = "right"
    device_map: str = "auto"
    use_4bit: bool = False


def load_pipeline(config):
    hh.login(token=os.environ["HF_AUTH_TOKEN"])

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        model_max_length=config.model_max_length,
        padding_side=config.padding,
    )
    tokenizer.pad_token = tokenizer.eos_token

    # load pre-trained model
    load_model_params = {
        # "torch_dtype": torch.float16,
        "torch_dtype": torch.float32,
        "device_map": config.device_map,
    }
    if config.use_4bit:
        load_model_params = {
            **load_model_params,
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                llm_int8_threshold=6.0,
                llm_int8_skip_modules=None,
                llm_int8_enable_fp32_cpu_offload=True,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float32,
            ),
        }

    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        **load_model_params,
    )

    # lora_config = LoraConfig.from_pretrained(config.adapter_path)
    # lora_config.inference_mode = True
    # model = get_peft_model(model, lora_config)
    model.load_adapter(config.adapter_path, adapter_name="default")
    model.set_adapter("default")

    if torch.cuda.is_available():
        device = "cuda"
    # elif torch.backends.mps.is_available():
    #     device = "mps"
    else:
        device = "cpu"

    model = model.to(device)

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map=config.device_map,
        device=device,
    )


if __name__ == "__main__":
    config = ServingConfig(
        model_path="codellama/CodeLlama-7b-hf",
        adapter_path="unionai/FlyteLlama-v0-7b-hf-flyte-repos",
        device_map=None,
    )
    print("loading pipeline")
    pipe = load_pipeline(config)
        
    print("generating...")
    results = pipe(
        ["The code below is a task that uses the Spark plugin to process pyspark dataframes"] * 4,
        max_length=1024,
        pad_token_id=pipe.tokenizer.eos_token_id,
    )

    for res in results:
        for text in res:
            print(text["generated_text"])
    