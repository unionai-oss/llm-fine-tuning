"""
envd build -f :serving --output type=image,name=ghcr.io/unionai-oss/modelz-flyte-llama-serving:v0,push=true
"""

import os
from dataclasses import dataclass

import torch  # type: ignore
import huggingface_hub as hh
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)


@dataclass
class ServingConfig:
    model_path: str
    adapter_path: str
    model_max_length: int = 1024
    max_new_tokens: int = 10
    padding: str = "right"
    device_map: str = "auto"
    use_4bit: bool = False


def load_pipeline(config):
    # hh.login(token=os.environ["HF_AUTH_TOKEN"])

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        model_max_length=config.model_max_length,
        padding_side=config.padding,
    )
    tokenizer.pad_token = tokenizer.eos_token

    # load pre-trained model
    load_model_params = {
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
        config.adapter_path,
        **load_model_params,
    )

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map=config.device_map,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--stream", action="store_true", default=False)
    args = parser.parse_args()
    
    config = ServingConfig(
        model_path="codellama/CodeLlama-7b-hf",
        adapter_path="unionai/FlyteLlama-v0-7b-hf-flyte-repos",
        device_map=None,
    )
    print("loading pipeline")
    pipe = load_pipeline(config)
        
    print("generating...")

    prompts = ["The code below shows a basic Flyte workflow"]
    print(prompts[0], end="", flush=True)

    prev_msg = prompts[0]
    if args.stream:
        tokens = pipe.tokenizer(
            prompts,
            add_special_tokens=False,
            return_tensors="pt",
        )
        inputs = tokens["input_ids"]
        for i in range(100):
            inputs = pipe.model.generate(
                inputs,
                pad_token_id=pipe.tokenizer.eos_token_id,
                max_new_tokens=config.max_new_tokens,
            )

            if inputs.shape[-1] >= config.model_max_length:
                inputs = inputs[:, -config.model_max_length:]

            msg = pipe.tokenizer.decode(inputs[0])
            print_msg = msg[len(prev_msg):]
            print(print_msg, end="", flush=True)
            prev_msg = msg

    else:
        results = pipe(
            prompts,
            max_length=40,
            pad_token_id=pipe.tokenizer.eos_token_id,
        )

        for res in results:
            for text in res:
                print(text["generated_text"])
        