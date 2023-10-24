"""
with docker:

```
docker build . -f Dockerfile.server -t ghcr.io/unionai-oss/modelz-flyte-llama-serving:v13
```

with envd:

```
envd build -f :serving --output type=image,name=ghcr.io/unionai-oss/modelz-flyte-llama-serving:v0,push=true
```
"""

import os
from dataclasses import dataclass
from typing import Optional

import torch
import huggingface_hub as hh
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from mosec import Server, Worker, SSEWorker, ValidationError, get_logger

logger = get_logger()


@dataclass
class ServingConfig:
    model_path: str
    adapter_path: str
    model_max_length: int = 1024
    n_turns: int = 100
    n_tokens_per_turn: int = 25
    padding: str = "left"
    device_map: str = "auto"
    device: Optional[str] = None
    use_float16: bool = False
    use_4bit: bool = False


def load_tokenizer_and_model(config):
    hh.login(token=os.environ["HF_AUTH_TOKEN"])
    
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        model_max_length=config.model_max_length,
        padding_side=config.padding,
    )
    tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if config.use_float16 else torch.float32

    # load pre-trained model
    load_model_params = {
        "torch_dtype": dtype,
        # "device": device,
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
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
            "load_in_4bit": True,
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


class Preprocess(Worker):
    def forward(self, params: dict):
        if params.get("prompt") is None:
            raise ValidationError("prompt is required")
        return params


class FlyteLlama(SSEWorker):

    def __init__(self):

        if torch.cuda.is_available():
            device_map = "auto"
            use_float16 = True
            use_4bit = True
        else:
            device_map = None
            use_float16 = False
            use_4bit = False

        self.config = ServingConfig(
            model_path="codellama/CodeLlama-7b-hf",
            adapter_path="unionai/FlyteLlama-v0-7b-hf-flyte-repos",
            device_map=device_map,
            use_float16=use_float16,
            use_4bit=use_4bit,
            n_turns=100,
            n_tokens_per_turn=10,
        )
        self.pipeline = load_tokenizer_and_model(self.config)
        # self.example = ["Flyte is a"]  # warmup

    def forward(self, params: dict):
        prompt = params.get("prompt")
        n_tokens = params.get("n_tokens", 1000)
        logger.info(f"generate text for {prompt}")

        tokens = self.pipeline.tokenizer(
            prompt,
            add_special_tokens=True,
            return_tensors="pt",
        )
        token_buffer = tokens["input_ids"]
        if torch.cuda.is_available():
            token_buffer = token_buffer.to("cuda")

        new_tokens = 0
        for _ in range(self.config.n_turns):
            token_buffer = self.pipeline.model.generate(
                token_buffer,
                pad_token_id=self.pipeline.tokenizer.eos_token_id,
                max_new_tokens=self.config.n_tokens_per_turn,
            )

            if token_buffer.shape[-1] >= self.config.model_max_length:
                token_buffer = token_buffer[:, -self.config.model_max_length:]

            for i in range(token_buffer.shape[0]):
                gen_text = self.pipeline.tokenizer.decode(token_buffer[i])
                self.send_stream_event(gen_text, index=i)

            new_tokens += self.config.n_tokens_per_turn
            if new_tokens >= n_tokens:
                break

        return params


if __name__ == "__main__":
    server = Server()

    def get_env(cid: int) -> dict:
        device_dict = {}
        if torch.cuda.is_available():
            device_dict["CUDA_VISIBLE_DEVICES"] = str(cid)
        return {
            "HF_AUTH_TOKEN": os.environ["HF_AUTH_TOKEN"],
            **device_dict,
        }
    
    num_devices = torch.cuda.device_count() or 1
    kwargs = dict(
        num=num_devices,
        max_batch_size=1,
        timeout=180_000,
        env=[get_env(i) for i in range(num_devices)],
    )
    server.append_worker(Preprocess, **kwargs)
    server.append_worker(FlyteLlama, **kwargs)
    server.run()
