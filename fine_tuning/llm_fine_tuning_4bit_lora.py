import json
import os
import sys
from pathlib import Path
from typing import List, Optional

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

import flytekit
import torch
import torch.nn as nn
import transformers
from flytekit import Secret
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.utils import logging

from datasets import DatasetDict, load_dataset
from flytekit import Resources
from flytekitplugins.kfpytorch.task import Elastic
from kubernetes.client.models import (
    V1PodSpec,
    V1Container,
    V1Volume,
    V1EmptyDirVolumeSource,
    V1VolumeMount,
)

from fine_tuning.llm_fine_tuning import save_to_hf_hub, PublishConfig

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
    prepare_model_for_kbit_training,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


logging.set_verbosity_debug()
logger = logging.get_logger("transformers")


# Union Cloud Tenants
SECRET_GROUP = "arn:aws:secretsmanager:us-east-2:356633062068:secret:"
WANDB_API_SECRET_KEY = "wandb_api_key-n5yPqE"
HF_HUB_API_SECRET_KEY = "huggingface_hub_api_key-qwgGkT"

# Flyte Development Tenant
# SECRET_GROUP = "arn:aws:secretsmanager:us-east-2:590375264460:secret:"
# WANDB_API_SECRET_KEY = "wandb_api_key-5t1ZwJ"
# HF_HUB_API_SECRET_KEY = "huggingface_hub_api_key-86cbXP"


@dataclass_json
@dataclass
class TrainerConfig:
    base_model: str = "huggyllama/llama-13b"
    data_path: str = "yahma/alpaca-cleaned"
    instruction_key: str = "instruction"
    input_key: str = "input"
    output_key: str = "output"
    output_dir: str = "./output"
    device_map: str = "auto"
    batch_size: int = 128
    micro_batch_size: int = 4
    num_epochs: int = 3
    max_steps: int = -1
    eval_steps: int = 200
    save_steps: int = 200
    learning_rate: float = 3e-4
    cutoff_len: int = 256
    val_set_size: int = 2000
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    weight_decay: float = 0.02
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    train_on_inputs: bool = True
    add_eos_token: bool = True
    group_by_length: bool = False
    resume_from_checkpoint: Optional[str] = None
    wandb_project: str = "unionai-llm-fine-tuning"
    wandb_run_name: str = ""
    wandb_watch: str = ""  # options: false | gradients | all
    wandb_log_model: str = ""  # options: false | true
    debug_mode: bool = False
    debug_train_data_size: int = 1024



class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            if os.path.exists(pytorch_model_path):
                os.remove(pytorch_model_path)
        return control


TEMPLATE = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that "
        "provides further context. Write a response that appropriately completes "
        "the request.\n\n### Instruction:\n{instruction}\n\n### Query:\n{input}\n\n### Response:\n"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. Write a response that "
        "appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
    ),
}


def generate_prompt(
    instruction: str,
    input: Optional[str] = None,
    output: Optional[str] = None
) -> str:
    if input is not None:
        prompt = TEMPLATE["prompt_input"].format(instruction=instruction, input=input)
    else:
        prompt = TEMPLATE["prompt_no_input"].format(instruction)

    if output:
        prompt = f"{prompt}{output}"
    return prompt


class TokenizerHelper:
    def __init__(self, tokenizer, train_on_inputs, cutoff_len, add_eos_token=True):
        self.tokenizer = tokenizer
        self.train_on_inputs = train_on_inputs
        self.add_eos_token = add_eos_token
        self.cutoff_len = cutoff_len

    def tokenize(self, prompt):
        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.cutoff_len,
            # Set padding to 'max_length' instead of False for GPTNeoXTokenizerFast???
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != self.tokenizer.eos_token_id
            and len(result["input_ids"]) < self.cutoff_len
            and self.add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        return result

    def generate_and_tokenize_prompt(self, data_point):
        full_prompt = generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = self.tokenize(full_prompt)

        if not self.train_on_inputs:
            user_prompt = generate_prompt(data_point["instruction"], data_point["input"])
            tokenized_user_prompt = self.tokenize(user_prompt)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if self.add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["input_ids"][
                user_prompt_len:
            ]  # could be sped up, probably
        else:
            tokenized_full_prompt["labels"] = tokenized_full_prompt["input_ids"]

        return tokenized_full_prompt



@flytekit.task(
    retries=3,
    cache=True,
    cache_version="0.0.14",
    requests=Resources(mem="120Gi", cpu="44", gpu="8", ephemeral_storage="200Gi"),
    # task_config=Elastic(nnodes=1),
    pod_template=flytekit.PodTemplate(
        primary_container_name="unionai-llm-fine-tuning",
        pod_spec=V1PodSpec(
            containers=[
                V1Container(
                    name="unionai-llm-fine-tuning",
                    volume_mounts=[V1VolumeMount(mount_path="/dev/shm", name="dshm")]
                )
            ],
            volumes=[
                V1Volume(
                    name="dshm",
                    empty_dir=V1EmptyDirVolumeSource(medium="Memory", size_limit="60Gi")
                )
            ]
        ),
    ),
    environment={
        "WANDB_PROJECT": "unionai-llm-fine-tuning",
        "TRANSFORMERS_CACHE": "/tmp",
        "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python",
    },
    secret_requests=[
        Secret(
            group=SECRET_GROUP,
            key=WANDB_API_SECRET_KEY,
            mount_requirement=Secret.MountType.FILE,
        ),
        Secret(
            group=SECRET_GROUP,
            key=HF_HUB_API_SECRET_KEY,
            mount_requirement=Secret.MountType.FILE,
        ),
    ],
)
def train(config: TrainerConfig) -> flytekit.directory.FlyteDirectory:
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        logger.info(f"Training Alpaca-LoRA model with params:\n{config}")

    os.environ["WANDB_API_KEY"] = flytekit.current_context().secrets.get(
        SECRET_GROUP, WANDB_API_SECRET_KEY
    )

    try:
        hf_auth_token = flytekit.current_context().secrets.get(
            SECRET_GROUP,
            HF_HUB_API_SECRET_KEY,
        )
    except Exception:
        hf_auth_token = None

    wandb_run_name = os.environ.get("FLYTE_INTERNAL_EXECUTION_ID", "local")
    os.environ["WANDB_RUN_ID"] = wandb_run_name

    gradient_accumulation_steps = config.batch_size // config.micro_batch_size
    device_map = config.device_map
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    logger.info(f"WORLD_SIZE: {world_size}")
    ddp = world_size != 1
    # if ddp:
    #     device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    #     gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(config.wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )

    # Only overwrite environ if wandb param passed
    if len(config.wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = config.wandb_project
    if len(config.wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = config.wandb_watch
    if len(config.wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = config.wandb_log_model

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
        max_memory={i: '46000MB' for i in range(torch.cuda.device_count())},
        quantization_config=bnb_config,
        use_auth_token=hf_auth_token, 
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model,
        use_auth_token=hf_auth_token,
    )
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
            LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )

    data = (
        load_dataset(config.data_path)
        if not config.debug_mode
        else DatasetDict(
            {
                "train": load_dataset(
                    config.data_path,
                    split=f"train[:{config.debug_train_data_size}]"
                )
            }
        )
    )

    if config.resume_from_checkpoint:

        # Check the available weights and load them
        checkpoint_name = os.path.join(config.resume_from_checkpoint, "pytorch_model.bin")  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(config.resume_from_checkpoint, "adapter_model.bin")
            config.resume_from_checkpoint = False  # So the trainer won't try loading its state

        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    tokenizer_helper = TokenizerHelper(
        tokenizer,
        config.train_on_inputs,
        config.cutoff_len,
        config.add_eos_token,
    )
    if config.val_set_size > 0:
        train_val = data["train"].train_test_split(test_size=config.val_set_size, shuffle=True, seed=42)
        train_data = train_val["train"].shuffle().map(tokenizer_helper.generate_and_tokenize_prompt)
        val_data = train_val["test"].shuffle().map(tokenizer_helper.generate_and_tokenize_prompt)
    else:
        train_data = data["train"].shuffle().map(tokenizer_helper.generate_and_tokenize_prompt)
        val_data = None

    # if not ddp and torch.cuda.device_count() > 1:
    #     # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
    #     model.is_parallelizable = True
    #     model.model_parallel = True

    eval_steps = 20 if config.debug_mode else config.eval_steps
    save_steps = 20 if config.debug_mode else config.save_steps
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=config.micro_batch_size,
            per_device_eval_batch_size=config.micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=config.num_epochs,
            max_steps=config.max_steps,
            learning_rate=config.learning_rate,
            warmup_ratio=config.warmup_ratio,
            weight_decay=config.weight_decay,
            lr_scheduler_type=config.lr_scheduler_type,
            fp16=True,
            half_precision_backend="auto",
            logging_steps=1,
            # optim="adamw_torch",
            # optim="adamw_bnb_8bit",
            optim="paged_adamw_8bit",
            evaluation_strategy="steps" if config.val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=eval_steps if config.val_set_size > 0 else None,
            save_steps=save_steps,
            output_dir=config.output_dir,
            save_total_limit=1,
            load_best_model_at_end=True if config.val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=config.group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        # callbacks=[SavePeftModelCallback],
    )

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    logger.info("Starting training run")
    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    trainer.save_model(output_dir=config.output_dir)
    model.save_pretrained(config.output_dir)
    with (Path(config.output_dir) / "flyte_training_config.json").open("w") as f:
        json.dump(config.to_dict(), f)

    return flytekit.directory.FlyteDirectory(path=config.output_dir)


@flytekit.workflow
def fine_tune(
    config: TrainerConfig,
    publish_config: PublishConfig,
):
    model_dir = train(config=config)
    save_to_hf_hub(
        model_dir=model_dir,
        publish_config=publish_config,
    )
