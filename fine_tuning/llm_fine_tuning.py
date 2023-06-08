import copy
import json
import logging
import os
from dataclasses import asdict, dataclass, field, replace
from io import BytesIO, StringIO
from pathlib import Path
from typing import Optional, Dict, List, Sequence

import huggingface_hub as hh
import torch
import transformers
import yaml
from datasets import DatasetDict, load_dataset
from flytekit import Resources
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from kubernetes.client.models import (
    V1PodSpec,
    V1Container,
    V1Volume,
    V1EmptyDirVolumeSource,
    V1VolumeMount,
)

import flytekit
from flytekit import Secret
from flytekitplugins.kfpytorch.task import Elastic
from dataclasses_json import dataclass_json


SECRET_GROUP = "arn:aws:secretsmanager:us-east-2:356633062068:secret:"
WANDB_API_SECRET_KEY = "wandb_api_key-n5yPqE"
HF_HUB_API_SECRET_KEY = "huggingface_hub_api_key-qwgGkT"

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


@dataclass_json
@dataclass
class TrainerConfig:
    num_epochs: int = 1
    max_steps: int = -1
    learning_rate: float = 0.00002
    weight_decay: float = 0.02
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    batch_size: int = 8
    micro_batch_size: int = 1
    val_set_size: int = 0
    group_by_length: bool = False
    model_name_or_path: Optional[str] = field(default="EleutherAI/pythia-1B-deduped")
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    instruction_key: str = "instruction"
    input_key: str = "input"
    output_key: str = "output"
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    debug_mode: bool = False
    debug_train_data_size: int = 1024
    wandb_project: str = field(default="")

@dataclass_json
@dataclass
class HuggingFaceModelCard:
    language: List[str]
    license: str  # valid licenses can be found at https://hf.co/docs/hub/repositories-licenses
    tags: List[str]

@dataclass_json
@dataclass
class PublishArguments:
    repo_id: str
    readme: Optional[str] = None
    model_card: Optional[HuggingFaceModelCard] = None


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        config: TrainerConfig,
        tokenizer: transformers.PreTrainedTokenizer,
        instruction_key: str,
        input_key: str,
        output_key: str,
    ):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")

        try:
            dataset_dict = (
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
            dataset = dataset_dict["train"]
            raw_data = [*dataset]
        except:
            with open(config.data_path) as f:
                raw_data = json.load(f)

        list_data_dict = [
            {
                "instruction": data[instruction_key],
                "input": data[input_key],
                "output": data[output_key],
            }
            for data in raw_data
        ]

        logging.warning("Formatting inputs...")
        prompt_input = PROMPT_DICT["prompt_input"]
        prompt_no_input = PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example)
            if example.get("input")
            else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    config: TrainerConfig,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(
        config=config,
        tokenizer=tokenizer,
        instruction_key=config.instruction_key,
        input_key=config.input_key,
        output_key=config.output_key,
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )


finetuning_image_spec = flytekit.ImageSpec(
    base_image="pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel",
    name="unionai-llm-fine-tuning",
    python_version="3.10",
    apt_packages=["git"],
    env={
        "DS_BUILD_OPS": "1",
        "DS_BUILD_AIO": "0",
        "DS_BUILD_SPARSE_ATTN": "0",
    },
    packages=[
        "accelerate",
        # pin due to https://github.com/TimDettmers/bitsandbytes/issues/324
        "'bitsandbytes==0.37.2'",
        "datasets",
        "'deepspeed>=0.8.3,<0.9'",
        "huggingface_hub",
        "loralib",
        "numpy",
        "pyyaml",
        "rouge_score",
        "fire",
        "openai",
        "'transformers[torch,deepspeed]>=4.28.1,<5'",
        "tokenizers",
        "torch",
        "sentencepiece",
        "wandb",
        "flytekit",
        "flytekitplugins-kfpytorch",
        "'git+https://github.com/huggingface/peft.git'",
    ],
    registry="ghcr.io/unionai-oss",
)


@flytekit.task(
    task_config=Elastic(nnodes=1),
    requests=Resources(mem="120Gi", cpu="44", gpu="8", ephemeral_storage="100Gi"),
    container_image=finetuning_image_spec,
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
    },
    secret_requests=[
        Secret(
            group=SECRET_GROUP,
            key=WANDB_API_SECRET_KEY,
            mount_requirement=Secret.MountType.FILE,
        )
    ],
)
def train(
    config: TrainerConfig,
    fsdp_config: Optional[dict] = None,
    ds_config: Optional[dict] = None,
) -> flytekit.directory.FlyteDirectory:
    """Fine-tune a model on additional data."""
    os.environ["WANDB_API_KEY"] = flytekit.current_context().secrets.get(
        SECRET_GROUP,
        WANDB_API_SECRET_KEY,
    )
    os.environ["WANDB_RUN_ID"] = os.environ.get("FLYTE_INTERNAL_EXECUTION_ID")
    use_wandb = len(config.wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    if config.wandb_project:
        os.environ["WANDB_PROJECT"] = config.wandb_project

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    gradient_accumulation_steps = config.batch_size // config.micro_batch_size
    eval_steps = 10 if config.debug_mode else 200
    save_steps = 10 if config.debug_mode else 200
    training_args = TrainingArguments(
        optim="adamw_torch",
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        num_train_epochs=config.num_epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.micro_batch_size,
        per_device_eval_batch_size=config.micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        evaluation_strategy="steps" if config.val_set_size > 0 else "no",
        save_strategy="steps",
        eval_steps=eval_steps if config.val_set_size > 0 else None,
        save_steps=save_steps,
        save_total_limit=1,
        load_best_model_at_end=True if config.val_set_size > 0 else False,
        ddp_find_unused_parameters=False if ddp else None,
        group_by_length=config.group_by_length,
        report_to="wandb" if use_wandb else None,
        half_precision_backend="auto",
        logging_steps=10,
        fp16=True,
        output_dir="/tmp",
        fsdp=fsdp_config.get("fsdp", False),
        fsdp_config=fsdp_config.get("fsdp_config", {}),
        deepspeed=ds_config,
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        cache_dir=config.cache_dir,
    )

    tokenizer_kwargs = dict(
        cache_dir=config.cache_dir,
        model_max_length=config.model_max_length,
        padding_side="right",
    )

    # Try using fast version of the model's tokenizer, if available.
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            config.model_name_or_path, use_fast=True, **tokenizer_kwargs
        )
    except:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            config.model_name_or_path, use_fast=False, **tokenizer_kwargs
        )

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    if "llama" in config.model_name_or_path:
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )

    data_module = make_supervised_data_module(tokenizer=tokenizer, config=config)
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )
    print("training model")
    trainer.train()
    trainer.save_state()
    print("saving model")
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    print("saving arguments for model, data, and training")
    output_root = Path(training_args.output_dir)

    with (output_root / "config.json").open("w") as f:
        json.dump(config.to_dict(), f)

    print("done")
    return flytekit.directory.FlyteDirectory(path=training_args.output_dir)


MODEL_CARD_TEMPLATE = """
---
{model_card_content}
---

{readme_content}
""".strip()


@flytekit.task(
    requests=Resources(mem="10Gi", cpu="1"),
    secret_requests=[
        Secret(
            group=SECRET_GROUP,
            key=HF_HUB_API_SECRET_KEY,
            mount_requirement=Secret.MountType.FILE,
        )
    ]
)
def save_to_hf_hub(
    model_dir: flytekit.directory.FlyteDirectory,
    publish_args: PublishArguments,
):
    # make sure the file can be downloaded
    model_dir.download()
    root = Path(model_dir.path)
    hh.login(
        token=flytekit.current_context().secrets.get(
            SECRET_GROUP,
            HF_HUB_API_SECRET_KEY,
        )
    )
    api = hh.HfApi()
    api.create_repo(publish_args.repo_id, exist_ok=True)

    with (root / "data_args.json").open() as f:
        data_args = json.load(f)

    if publish_args.readme is not None:
        StringIO()
        model_card_dict = publish_args.model_card.to_dict()

        dataset_path = data_args.get("data_path", None)
        if dataset_path:
            model_card_dict["datasets"] = [data_args.get("data_path")]

        readme_str = MODEL_CARD_TEMPLATE.format(
            model_card_content=yaml.dump(model_card_dict),
            readme_content=publish_args.readme,
        )
        api.upload_file(
            path_or_fileobj=BytesIO(readme_str.encode()),
            path_in_repo="README.md",
            repo_id=publish_args.repo_id,
        )

    for file_name in [
        "config.json",
        "pytorch_model.bin",
        "tokenizer.json",
        "tokenizer_config.json",
    ]:
        api.upload_file(
            path_or_fileobj=root / file_name,
            path_in_repo=file_name,
            repo_id=publish_args.repo_id,
        )


@flytekit.workflow
def fine_tune(
    training_config: TrainerConfig,
    publish_args: PublishArguments,
    fsdp: Optional[List[str]] = None,
    fsdp_config: Optional[dict] = None,
    ds_config: Optional[dict] = None,
):
    model_dir = train(
        training_config=training_config,
        fsdp=fsdp,
        fsdp_config=fsdp_config,
        ds_config=ds_config,
    )
    save_to_hf_hub(
        model_dir=model_dir,
        publish_args=publish_args,
    )


def train_cli():
    parser = transformers.HfArgumentParser((TrainerConfig, ))
    config = parser.parse_args_into_dataclasses()
    train(config)


if __name__ == "__main__":
    train_cli()
