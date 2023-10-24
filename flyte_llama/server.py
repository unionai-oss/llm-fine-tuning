"""
with docker:

```
docker build . -f Dockerfile.server -t ghcr.io/unionai-oss/modelz-flyte-llama-serving:v13
```

with envd:

```
envd build -f server.envd:serving --output type=image,name=ghcr.io/unionai-oss/modelz-flyte-llama-serving:$VERSION,push=true
```
"""

import os
from dataclasses import dataclass
from io import BytesIO
from typing import List, Optional

import torch
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
    adapter_path: Optional[str] = None
    model_max_length: int = 1024
    max_gen_length: int = 1024
    padding: str = "right"
    device_map: str = "auto"
    use_float16: bool = False
    use_4bit: bool = False


def load_pipeline(config):
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
        config.adapter_path or config.model_path,
        **load_model_params,
    )

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map=config.device_map,
    )


class FlyteLlama(MsgpackMixin, Worker):

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
        )
        self.pipe = load_pipeline(self.config)

    def forward(self, data: List[str]) -> List[memoryview]:
        logger.debug("generate images for %s", data)
        results = self.pipe(
            data,
            max_length=self.config.max_gen_length,
            pad_token_id=self.pipe.tokenizer.eos_token_id,
        )
        outputs = []
        for res in results:
            for text in res:
                dummy_file = BytesIO()
                dummy_file.write(text["generated_text"].encode())
                outputs.append(dummy_file.getbuffer())
        return outputs


if __name__ == "__main__":
    server = Server()

    num_devices = torch.cuda.device_count() or 1
    server.append_worker(
        FlyteLlama,
        num=num_devices,
        max_batch_size=4,
        max_wait_time=10,
        timeout=180_000,
        env=[{"HF_AUTH_TOKEN": os.environ["HF_AUTH_TOKEN"]}] * num_devices,
    )
    server.run()
