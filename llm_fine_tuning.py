import copy
import json
import logging
import os
from dataclasses import dataclass, field, replace
from io import BytesIO, StringIO
from pathlib import Path
from typing import Optional, Dict, List, Sequence

import huggingface_hub as hh
import torch
import transformers
import yaml
from datasets import load_dataset
from flytekit import Resources
from torch.utils.data import Dataset
from transformers import Trainer

import flytekit
from flytekitplugins.kfpytorch.task import Elastic
from dataclasses_json import dataclass_json

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
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass_json
@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    instruction_key: str = "instruction"
    input_key: str = "input"
    output_key: str = "output"


@dataclass_json
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )

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
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        instruction_key: str,
        input_key: str,
        output_key: str,
    ):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")

        try:
            dataset_dict = load_dataset(data_path)
            dataset = dataset_dict["train"]
            raw_data = [*dataset]
        except:
            with open(data_path) as f:
                raw_data = json.load(f)

        list_data_dict = [
            {
                "instruction": x[instruction_key],
                "input": x[input_key],
                "output": x[output_key],
            }
            for x in raw_data
        ]

        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
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
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        instruction_key=data_args.instruction_key,
        input_key=data_args.input_key,
        output_key=data_args.output_key,
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


@flytekit.task(
    task_config=Elastic(nnodes=1),
    environment={
        "TRANSFORMERS_CACHE": "/tmp",
        "WANDB_API_KEY": "<wandb_api_key>",
        "WANDB_PROJECT": "unionai-llm-fine-tuning",
    },
    requests=Resources(mem="100Gi", cpu="60", gpu="8", ephemeral_storage="100Gi"),
)
def train(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
    fsdp: Optional[List[str]] = None,
    fsdp_config: Optional[str] = None,
    ds_config: Optional[str] = None,
) -> flytekit.directory.FlyteDirectory:
    os.environ["WANDB_RUN_ID"] = os.environ.get("FLYTE_INTERNAL_EXECUTION_ID")

    if fsdp_config is not None:
        with open(fsdp_config) as f:
            fsdp_config = json.load(f)

    if ds_config is not None:
        with open(ds_config) as f:
            ds_config = json.load(f)

    training_args = replace(
        training_args,
        fp16=True,
        fsdp=fsdp,
        fsdp_config=fsdp_config,
        ds_config=ds_config,
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    tokenizer_kwargs = dict(
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
    )

    # Try using fast version of the model's tokenizer, if available.
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, use_fast=True, **tokenizer_kwargs
        )
    except:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, use_fast=False, **tokenizer_kwargs
        )

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    if "llama" in model_args.model_name_or_path:
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    print("training model")
    trainer.train()
    trainer.save_state()
    print("saving model")
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    print("saving arguments for model, data, and training")
    output_root = Path(training_args.output_dir)

    for args, fn in [
        (model_args, "model_args.json"),
        (data_args, "data_args.json"),
        (training_args, "training_args.json"),
    ]:
        with (output_root / fn).open("w") as f:
            json.dump(args.to_dict(), f)

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
    environment={
        "HUGGINGFACE_TOKEN": "<huggingface_token>",
    },
)
def save_to_hf_hub(
    model_dir: flytekit.directory.FlyteDirectory,
    repo_id: str,
    model_card: HuggingFaceModelCard,
    readme: str,
):
    # make sure the file can be downloaded
    model_dir.download()
    root = Path(model_dir.path)

    hh.login(token=os.environ["HUGGINGFACE_TOKEN"])
    api = hh.HfApi()
    api.create_repo(repo_id, exist_ok=True)

    with (root / "data_args.json").open() as f:
        data_args = json.load(f)

    if readme is not None:
        StringIO()
        model_card_dict = model_card.to_dict()

        dataset_path = data_args.get("data_path", None)
        if dataset_path:
            model_card_dict["datasets"] = [data_args.get("data_path")]

        readme_str = MODEL_CARD_TEMPLATE.format(
            model_card_content=yaml.dump(model_card_dict),
            readme_content=readme,
        )
        api.upload_file(
            path_or_fileobj=BytesIO(readme_str.encode()),
            path_in_repo="README.md",
            repo_id=repo_id,
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
            repo_id=repo_id,
        )


@flytekit.workflow
def fine_tune(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
    publish_args: PublishArguments,
):
    model_dir = train(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
    )
    save_to_hf_hub(
        model_dir=model_dir,
        repo_id=publish_args.repo_id,
        model_card=publish_args.model_card,
        readme=publish_args.readme,
    )


def train_cli():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    train(model_args, data_args, training_args)


if __name__ == "__main__":
    train()
