"""Flyte LLama workflows."""

import os
from pathlib import Path
from typing import List, Optional

from flytekit import task, workflow, current_context, Resources, Secret
from flytekit.loggers import logger
from flytekit.types.directory import FlyteDirectory

import flyte_llama


# Union Tenant
SECRET_GROUP = "arn:aws:secretsmanager:us-east-2:356633062068:secret:"
WANDB_API_SECRET_KEY = "wandb_api_key-n5yPqE"
HF_HUB_API_SECRET_KEY = "huggingface_hub_api_key-qwgGkT"

# # Flyte Development Tenant
# SECRET_GROUP = "arn:aws:secretsmanager:us-east-2:590375264460:secret:"
# WANDB_API_SECRET_KEY = "wandb_api_key-5t1ZwJ"
# HF_HUB_API_SECRET_KEY = "huggingface_hub_api_key-86cbXP"


@task(
    cache=True,
    cache_version="0",
    requests=Resources(mem="8Gi", cpu="2", ephemeral_storage="8Gi"),
)
def create_dataset(additional_urls: Optional[List[str]] = None) -> FlyteDirectory:
    urls = [*flyte_llama.dataset.REPO_URLS, *(additional_urls or [])]

    ctx = current_context()
    working_dir = Path(ctx.working_directory)
    output_dir = working_dir / "dataset"
    repo_cache_dir = working_dir / "repo_cache"

    flyte_llama.dataset.create_dataset(urls, output_dir, repo_cache_dir)
    return FlyteDirectory(path=str(output_dir))


@task(
    retries=3,
    cache=True,
    cache_version="0.0.19",
    requests=Resources(mem="120Gi", cpu="44", gpu="8", ephemeral_storage="200Gi"),
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
def train(
    dataset: FlyteDirectory,
    config: flyte_llama.train.TrainerConfig,
    pretrained_adapter: Optional[FlyteDirectory] = None,
) -> FlyteDirectory:
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        logger.info(f"Training Flyte Llama with params:\n{config}")

    if pretrained_adapter is not None:
        print(f"Downloading pretrained adapter {pretrained_adapter}")
        pretrained_adapter.download()

    wandb_run_name = os.environ.get("FLYTE_INTERNAL_EXECUTION_ID", "local")
    os.environ["WANDB_RUN_ID"] = wandb_run_name

    ctx = current_context()
    os.environ["WANDB_API_KEY"] = ctx.secrets.get(SECRET_GROUP, WANDB_API_SECRET_KEY)

    dataset.download()
    config.data_dir = dataset.path

    try:
        hf_auth_token = ctx.secrets.get(SECRET_GROUP, HF_HUB_API_SECRET_KEY)
    except Exception:
        hf_auth_token = None

    flyte_llama.train.train(config, pretrained_adapter, hf_auth_token)
    return FlyteDirectory(path=str(config.output_dir))


@workflow
def train_workflow(
    config: flyte_llama.train.TrainerConfig,
    pretrained_adapter: Optional[FlyteDirectory] = None,
) -> FlyteDirectory:
    dataset = create_dataset()
    model = train(
        dataset=dataset,
        config=config,
        pretrained_adapter=pretrained_adapter,
    )
    return model
