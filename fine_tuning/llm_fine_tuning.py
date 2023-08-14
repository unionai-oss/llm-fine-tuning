import copy
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Annotated, Optional, Dict, List, Sequence

import huggingface_hub as hh
import torch
import transformers
import yaml

from datasets import DatasetDict, Dataset, load_dataset
from flytekit import Resources
import pandera as pa
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
from flytekit.deck.renderer import TopFrameRenderer
from flytekitplugins.huggingface.sd_transformers import HuggingFaceDatasetRenderer
from flytekit.types.structured.structured_dataset import StructuredDataset, PARQUET
from flytekitplugins.kfpytorch.task import Elastic
from dataclasses_json import dataclass_json


class WikipediaDataset(pa.DataFrameModel):
    id: int
    url: str = pa.Field(str_startswith="https://")
    title: str = pa.Field(str_length={"max_value": 1_000})
    text: str = pa.Field(str_length={"max_value": 500_000})

    class Config:
        coerce = True



# Union Cloud Tenants
# SECRET_GROUP = "arn:aws:secretsmanager:us-east-2:356633062068:secret:"
# WANDB_API_SECRET_KEY = "wandb_api_key-n5yPqE"
# HF_HUB_API_SECRET_KEY = "huggingface_hub_api_key-qwgGkT"

# Flyte Development Tenant
SECRET_GROUP = "arn:aws:secretsmanager:us-east-2:590375264460:secret:"
WANDB_API_SECRET_KEY = "wandb_api_key-5t1ZwJ"
HF_HUB_API_SECRET_KEY = "huggingface_hub_api_key-86cbXP"

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
    base_model: Optional[str] = field(default="EleutherAI/pythia-1B-deduped")
    data_path: str = field(default="yahma/alpaca-cleaned", metadata={"help": "Path to the training data."})
    data_name: str = field(default=None, metadata={"help": "Path to the training data config name."})
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
    instruction_key: str = "instruction"
    input_key: str = "input"
    output_key: str = "output"
    device_map: str = "auto"
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
class PublishConfig:
    repo_id: str
    readme: Optional[str] = None
    model_card: Optional[HuggingFaceModelCard] = None


@dataclass_json
@dataclass
class LMEvalHarnessConfig:
    """
    Config base on:
    https://github.com/EleutherAI/lm-evaluation-harness/blob/master/main.py
    """
    model: str = field(default="hf-causal-experimental")
    model_args: str = field(default="unionai/pythia-1B-deduped-wikipedia")
    tasks: Optional[str] = field(default=None)
    provide_description: Optional[bool] = field(default=False)
    num_fewshot: int = 0
    batch_size: Optional[str] = field(default=None)
    device: Optional[str]= field(default=None)
    output_path: Optional[str]= field(default=None)
    limit: Optional[float]= field(default=None)
    data_sampling: Optional[float]= field(default=None)
    no_cache: Optional[bool] = field(default=False)
    description_dict_path: Optional[dict] = field(default=None)
    check_integrity: Optional[bool] = field(default=False)
    write_out: Optional[bool] = field(default=False)
    output_base_path: Optional[str]= field(default=None)


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


class SupervisedDataset(torch.utils.data.Dataset):
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
class DataCollatorForSupervisedDataset:
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


def make_supervised_prompt_data_module(
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


def make_causal_lm_data_module(
    dataset: Dataset, 
    tokenizer: transformers.PreTrainedTokenizer,
    config: TrainerConfig,
) -> Dict:
    def tokenize(element):
        outputs = tokenizer(
            f"<|title|>: {element['title']}\n<|content|>: {element['text']}",
            truncation=True,
            max_length=config.model_max_length,
            return_overflowing_tokens=True,
            return_length=True,
            return_tensors=None,
        )
        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            if length == config.model_max_length:
                input_batch.append(input_ids)
        return {"input_ids": input_batch}

    dataset = Dataset.from_pandas(dataset.to_pandas().head(100))
    return dict(
        train_dataset=dataset.shuffle().map(tokenize, batched=True, remove_columns=dataset.column_names),
        eval_dataset=None,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )


finetuning_pod_template = flytekit.PodTemplate(
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
)


@flytekit.task(
    requests=Resources(mem="8Gi", cpu="2", ephemeral_storage="8Gi"),
    disable_deck=False,
    cache=True,
    cache_version="0.0.0",
)
def get_data(config: TrainerConfig) -> Annotated[StructuredDataset, PARQUET]:
    dataset = load_dataset(config.data_path, config.data_name)
    pd_dataset = dataset["train"].to_pandas()

    # try:
    #     WikipediaDataset.validate(pd_dataset, lazy=True)
    # except pa.errors.SchemaErrors as exc:
    #     flytekit.Deck("pandera-errors", TopFrameRenderer(max_rows=100).to_html(exc.failure_cases))

    flytekit.Deck("dataset", HuggingFaceDatasetRenderer().to_html(dataset["train"]))
    return StructuredDataset(dataframe=dataset["train"])


@flytekit.task(
    retries=3,
    cache=True,
    cache_version="0.0.5",
    task_config=Elastic(nnodes=2, rdzv_configs={"timeout": 1200, "join_timeout": 900}),
    requests=Resources(mem="120Gi", cpu="44", gpu="8", ephemeral_storage="200Gi"),
    pod_template=finetuning_pod_template,
    environment={
        "WANDB_PROJECT": "unionai-llm-fine-tuning",
        "TRANSFORMERS_CACHE": "/tmp",
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
def train(
    config: TrainerConfig,
    clm_dataset: Optional[Annotated[StructuredDataset, PARQUET]] = None,
    ds_config: Optional[dict] = None,
) -> flytekit.directory.FlyteDirectory:
    """Fine-tune a model on additional data."""

    os.environ["WANDB_API_KEY"] = flytekit.current_context().secrets.get(
        SECRET_GROUP, WANDB_API_SECRET_KEY
    )
    print(f"SET WANDB API KEY {os.environ['WANDB_API_KEY']}")

    try:
        hf_auth_token = flytekit.current_context().secrets.get(
            SECRET_GROUP,
            HF_HUB_API_SECRET_KEY,
        )
    except Exception:
        hf_auth_token = None

    use_wandb = False
    if "WANDB_API_KEY" in os.environ:
        os.environ["WANDB_RUN_ID"] = os.environ.get("FLYTE_INTERNAL_EXECUTION_ID")
        if config.wandb_project:
            os.environ["WANDB_PROJECT"] = config.wandb_project
        use_wandb = len(config.wandb_project) > 0 or (
            "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
        )

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    gradient_accumulation_steps = config.batch_size // config.micro_batch_size
    eval_steps = 10 if config.debug_mode else 200
    save_steps = 10 if config.debug_mode else 200
    training_args = TrainingArguments(
        # optim="adamw_torch",
        fp16=True,
        fp16_full_eval=True,
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
        output_dir="/tmp",
        deepspeed=ds_config,
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.base_model,
        cache_dir=config.cache_dir,
        use_auth_token=hf_auth_token,
    )

    tokenizer_kwargs = dict(
        cache_dir=config.cache_dir,
        model_max_length=config.model_max_length,
        padding_side="right",
        pad_token=DEFAULT_PAD_TOKEN,
        use_auth_token=hf_auth_token,
    )

    # Try using fast version of the model's tokenizer, if available.
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            config.base_model, use_fast=True, **tokenizer_kwargs,
        )
    except:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            config.base_model, use_fast=False, **tokenizer_kwargs
        )

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    if "llama" in config.base_model:
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )

    if clm_dataset is not None:
        data_module = make_causal_lm_data_module(
            clm_dataset.open(Dataset).all(),
            tokenizer=tokenizer,
            config=config,
        )
    else:
        data_module = make_supervised_prompt_data_module(
            tokenizer=tokenizer,
            config=config,
        )

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
    trainer.save_model(output_dir=training_args.output_dir)

    print("saving arguments for model, data, and training")
    output_root = Path(training_args.output_dir)

    with (output_root / "flyte_training_config.json").open("w") as f:
        json.dump(config.to_dict(), f)

    print("done")
    return flytekit.directory.FlyteDirectory(path=training_args.output_dir)


@flytekit.task(
    cache=True,
    cache_version="0.0.3",
    requests=Resources(mem="120Gi", cpu="44", gpu="8", ephemeral_storage="100Gi"),
)
def quantize_model(
    config: TrainerConfig,
    model_dir: flytekit.directory.FlyteDirectory,
) -> flytekit.directory.FlyteDirectory:
    model_dir.download()

    device_map = config.device_map
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    model = transformers.AutoModelForCausalLM.from_pretrained(
        str(model_dir.path),
        cache_dir=config.cache_dir,
        load_in_8bit=True,
        device_map=device_map,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.base_model,
        use_fast=True,
        cache_dir=config.cache_dir,
        model_max_length=config.model_max_length,
        padding_side="right",
        pad_token=DEFAULT_PAD_TOKEN,
    )
    output_dir = "/tmp"
    trainer = Trainer(model=model, tokenizer=tokenizer)
    trainer.save_model(output_dir=output_dir)

    src, dst = Path(model_dir.path), Path(output_dir)
    for file_name in [
        "config.json",
        "flyte_training_config.json",
    ]:
        shutil.copy(src / file_name, dst / file_name)
    return flytekit.directory.FlyteDirectory(path=output_dir)


MODEL_CARD_TEMPLATE = """
---
{model_card_content}
---

{readme_content}
""".strip()


@flytekit.task(
    retries=3,
    cache=True,
    cache_version="0.0.4",
    requests=Resources(mem="10Gi", cpu="1", ephemeral_storage="100Gi"),
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
    publish_config: PublishConfig,
    quantized_8bit: Optional[bool] = None,
) -> str:
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
    if quantized_8bit:
        repo_id = f"{publish_config.repo_id}-8bit"
    else:
        repo_id = publish_config.repo_id

    repo_url = api.create_repo(repo_id, exist_ok=True)

    with (root / "flyte_training_config.json").open() as f:
        config = json.load(f)

    if publish_config.readme is not None:
        model_card_dict = publish_config.model_card.to_dict()

        dataset_path = config.get("data_path", None)
        if dataset_path:
            model_card_dict["datasets"] = [config.get("data_path")]

        readme_str = MODEL_CARD_TEMPLATE.format(
            model_card_content=yaml.dump(model_card_dict),
            readme_content=publish_config.readme,
        )
        api.upload_file(
            path_or_fileobj=BytesIO(readme_str.encode()),
            path_in_repo="README.md",
            repo_id=repo_id,
        )

    api.upload_folder(
        repo_id=repo_id,
        folder_path=root,
        ignore_patterns=["flyte-*", "models--*"]
    )
    return str(repo_url)


@flytekit.workflow
def fine_tune(
    config: TrainerConfig,
    publish_config: PublishConfig,
    ds_config: Optional[dict] = None,
):
    data = get_data(config=config)
    model_dir = train(
        config=config,
        clm_dataset=data,
        ds_config=ds_config,
    )
    quantized_model_dir = quantize_model(
        config=config,
        model_dir=model_dir,
    )

    repo_url = save_to_hf_hub(
        model_dir=model_dir,
        publish_config=publish_config,
    )
    quantized_repo_url = save_to_hf_hub(
        model_dir=quantized_model_dir,
        publish_config=publish_config,
        quantized_8bit=True,
    )


def train_cli():
    parser = transformers.HfArgumentParser((TrainerConfig, ))
    config = parser.parse_args_into_dataclasses()
    train(config)


if __name__ == "__main__":
    train_cli()
