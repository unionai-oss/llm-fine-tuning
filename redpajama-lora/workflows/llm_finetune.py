import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import flytekit
import torch
from dataclasses_json import dataclass_json
from datasets import DatasetDict, load_dataset
from flytekit import PodTemplate, Resources, Secret, task, workflow
from flytekit.types.directory import FlyteDirectory
from flytekitplugins.kfpytorch import Elastic
from kubernetes.client.models import (
    V1Container,
    V1EmptyDirVolumeSource,
    V1PodSpec,
    V1Volume,
    V1VolumeMount,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

SECRET_GROUP = "wandb"
SECRET_NAME = "wandb_api_key"


class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")

        if int(os.environ.get("LOCAL_RANK")) == 0:
            if os.path.exists(pytorch_model_path):
                os.remove(pytorch_model_path)
        return control


def generate_prompt(input, output=""):
    return f"""As an advanced chatbot, you enjoy assisting users on a community Slack platform. Write a response that appropriately answers the query. 

### Query:
{input}

### Response:
{output}"""


@dataclass_json
@dataclass
class TrainerConfig:
    base_model: str = "togethercomputer/RedPajama-INCITE-Chat-7B-v0.1"
    data_path: str = "Samhita/flyte-slack-data"
    output_dir: str = "./lora-redpajama"
    device_map: str = "auto"
    batch_size: int = 128
    micro_batch_size: int = 4
    num_epochs: int = 3
    learning_rate: float = 3e-4
    cutoff_len: int = 256
    val_set_size: int = 2000
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["query_key_value"])
    train_on_inputs: bool = True
    add_eos_token: bool = True
    group_by_length: bool = False
    resume_from_checkpoint: Optional[str] = None
    wandb_project: str = "redpajama-lora-finetuning"
    wandb_run_name: str = ""
    wandb_watch: str = ""  # options: false | gradients | all
    wandb_log_model: str = ""  # options: false | true
    debug_mode: bool = False
    debug_train_data_size: int = 1024


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
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = self.tokenize(full_prompt)

        if not self.train_on_inputs:
            user_prompt = generate_prompt(data_point["input"])
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


@task(
    task_config=Elastic(nnodes=1),
    requests=Resources(mem="100Gi", cpu="50", gpu="5", ephemeral_storage="100Gi"),
    pod_template=PodTemplate(
        primary_container_name="llm-fine-tuning",
        pod_spec=V1PodSpec(
            containers=[
                V1Container(
                    name="llm-fine-tuning",
                    image="ghcr.io/samhita-alla/redpajama-finetune:latest",
                    volume_mounts=[V1VolumeMount(mount_path="/dev/shm", name="dshm")],
                )
            ],
            volumes=[
                V1Volume(
                    name="dshm",
                    empty_dir=V1EmptyDirVolumeSource(
                        medium="Memory", size_limit="60Gi"
                    ),
                )
            ],
        ),
    ),
    environment={
        "TRANSFORMERS_CACHE": "/tmp",
        "CUDA_LAUNCH_BLOCKING": "1",
    },
    secret_requests=[
        Secret(
            group=SECRET_GROUP, key=SECRET_NAME, mount_requirement=Secret.MountType.FILE
        )
    ],
)
def redpajama_finetune(config: TrainerConfig) -> FlyteDirectory:
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"\n\n\nLoRA fine-tuning model with params:\n"
            f"base_model: {config.base_model}\n"
            f"data_path: {config.data_path}\n"
            f"output_dir: {config.output_dir}\n"
            f"batch_size: {config.batch_size}\n"
            f"micro_batch_size: {config.micro_batch_size}\n"
            f"num_epochs: {config.num_epochs}\n"
            f"learning_rate: {config.learning_rate}\n"
            f"cutoff_len: {config.cutoff_len}\n"
            f"val_set_size: {config.val_set_size}\n"
            f"lora_r: {config.lora_r}\n"
            f"lora_alpha: {config.lora_alpha}\n"
            f"lora_dropout: {config.lora_dropout}\n"
            f"lora_target_modules: {config.lora_target_modules}\n"
            f"train_on_inputs: {config.train_on_inputs}\n"
            f"add_eos_token: {config.add_eos_token}\n"
            f"group_by_length: {config.group_by_length}\n"
            f"resume_from_checkpoint: {config.resume_from_checkpoint or False}\n"
            f"wandb_project: {config.wandb_project}\n"
            f"wandb_run_name: {config.wandb_run_name}\n"
            f"wandb_watch: {config.wandb_watch}\n"
            f"wandb_log_model: {config.wandb_log_model}\n"
            f"debug_mode: {config.debug_mode}\n"
        )
    assert (
        config.base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    wandb_run_name = os.environ.get("FLYTE_INTERNAL_EXECUTION_ID")
    os.environ["WANDB_RUN_ID"] = wandb_run_name

    gradient_accumulation_steps = config.batch_size // config.micro_batch_size

    # world_size = int(os.environ.get("WORLD_SIZE", torch.distributed.get_world_size()))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    print(f"device map: {device_map}")

    # Check if parameter passed or if set within environ
    use_wandb = len(config.wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    os.environ["WANDB_API_KEY"] = flytekit.current_context().secrets.get(
        SECRET_GROUP, SECRET_NAME
    )
    # Only overwrite environ if wandb param passed
    if len(config.wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = config.wandb_project
    if len(config.wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = config.wandb_watch
    if len(config.wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = config.wandb_log_model

    # Model loading
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        trust_remote_code=True,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)

    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference

    # 8-bit training
    model = prepare_model_for_int8_training(model)

    # LoRA
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    data = (
        load_dataset(config.data_path)
        if not config.debug_mode
        else DatasetDict(
            {
                "train": load_dataset(
                    config.data_path, split=f"train[:{config.debug_train_data_size}]"
                )
            }
        )
    )

    if config.resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            config.resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                config.resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            config.resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )

        # The two files above have a different name depending on how they were saved,
        # but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    # Be more transparent about the % of trainable params.
    model.print_trainable_parameters()

    tokenizer_helper = TokenizerHelper(
        tokenizer,
        config.train_on_inputs,
        config.cutoff_len,
        config.add_eos_token,
    )

    if config.val_set_size > 0:
        config.val_set_size = 128 if config.debug_mode else config.val_set_size
        train_val = data["train"].train_test_split(
            test_size=config.val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"]
            .shuffle()
            .map(tokenizer_helper.generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"]
            .shuffle()
            .map(tokenizer_helper.generate_and_tokenize_prompt)
        )
    else:
        train_data = (
            data["train"].shuffle().map(tokenizer_helper.generate_and_tokenize_prompt)
        )
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism
        # when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    eval_steps = 10 if config.debug_mode else 200
    save_steps = 10 if config.debug_mode else 200
    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=TrainingArguments(
            per_device_train_batch_size=config.micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=config.num_epochs,
            learning_rate=config.learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if config.val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=eval_steps if config.val_set_size > 0 else None,
            save_steps=save_steps,
            output_dir=config.output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if config.val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=config.group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks=[SavePeftModelCallback],
    )
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)

    model.save_pretrained(config.output_dir)
    print("\n If there's a warning about missing keys above, please disregard :)")
    return FlyteDirectory(config.output_dir)


@workflow
def redpajama_finetuning_wf(config: TrainerConfig = TrainerConfig()) -> FlyteDirectory:
    return redpajama_finetune(config=config)
