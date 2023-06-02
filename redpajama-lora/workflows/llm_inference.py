import os
import sys

import torch
from flytekit import Resources, task, workflow
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def generate_prompt(input, output=""):
    return f"""As an advanced chatbot, you enjoy assisting users on a community Slack platform. Write a response that appropriately answers the query. Give a detailed explanation at all times.

### Query:
{input}

### Response:
{output}"""


@task(requests=Resources(gpu="1", mem="50Gi", cpu="10"))
def generate_output(
    input: str,
    temperature: float,
    top_p: float,
    top_k: int,
    num_beams: int,
    max_new_tokens: int,
    load_8bit: bool,
    base_model: str,
    lora_weights: str,
) -> str:
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    try:
        if torch.backends.mps.is_available():
            device = "mps"
    except:  # noqa: E722
        pass

    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    prompt = generate_prompt(input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
    )

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.3,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)

    return output.split("### Response:")[1].strip()


@workflow
def inference_wf(
    input: str = "Hi! Is there a way to shorten `ttlSecondsAfterFinished`? By default, it is 3600s (1 hour) and we’d like to tear down a cluster right after a job is complete. Thanks for your help! ```$ k describe rayjobs feb5da8c2a2394fb4ac8-n0-0 -n flytesnacks-development ... Ttl Seconds After Finished: 3600```",
    temperature: float = 0.3,
    top_p: float = 0.75,
    top_k: int = 40,
    num_beams: int = 4,
    max_new_tokens: int = 128,
    load_8bit: bool = True,
    base_model: str = "togethercomputer/RedPajama-INCITE-Chat-7B-v0.1",
    lora_weights: str = "Samhita/redpajama-lora-finetuned-T4",
) -> str:
    return generate_output(
        input=input,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        load_8bit=load_8bit,
        base_model=base_model,
        lora_weights=lora_weights,
    )
