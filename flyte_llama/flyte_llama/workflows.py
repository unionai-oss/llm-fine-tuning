"""Flyte LLama workflows."""

import asyncio
import os
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import List, Optional

from flytekit import task, workflow, map_task, dynamic, current_context, Resources, Secret, ImageSpec
from flytekit.configuration import Config, PlatformConfig
from flytekit.experimental import eager
from flytekit.loggers import logger
from flytekit.remote import FlyteRemote
from flytekit.types.directory import FlyteDirectory

import flyte_llama


# Union Tenant
SECRET_GROUP = "arn:aws:secretsmanager:us-east-2:356633062068:secret:"
WANDB_API_SECRET_KEY = "wandb_api_key-n5yPqE"
HF_HUB_API_SECRET_KEY = "huggingface_hub_api_key-qwgGkT"


image_spec = ImageSpec(
    name="flyte-llama-qlora",
    apt_packages=["git"],
    registry="ghcr.io/unionai-oss",
    requirements="requirements.txt",
    python_version="3.9",
    cuda="11.7.1",
    env={"VENV": "/opt/venv"},
)


@task(
    cache=True,
    cache_version="1",
    container_image=image_spec,
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
    cache_version="21",
    container_image=image_spec,
    requests=Resources(mem="120Gi", cpu="44", gpu="8", ephemeral_storage="100Gi"),
    environment={
        "WANDB_PROJECT": "unionai-flyte-llama",
        "TRANSFORMERS_CACHE": "/tmp",
        "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python",
        "TOKENIZERS_PARALLELISM": "true",
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
    try:
        os.environ["WANDB_API_KEY"] = ctx.secrets.get(SECRET_GROUP, WANDB_API_SECRET_KEY)
    except ValueError:
        pass

    dataset.download()
    config.data_dir = dataset.path.replace("file://", "")

    try:
        hf_auth_token = ctx.secrets.get(SECRET_GROUP, HF_HUB_API_SECRET_KEY)
    except ValueError:
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


@task(
    retries=3,
    cache=True,
    cache_version="0.0.4",
    container_image=image_spec,
    requests=Resources(mem="10Gi", cpu="1", ephemeral_storage="64Gi"),
    secret_requests=[
        Secret(
            group=SECRET_GROUP,
            key=HF_HUB_API_SECRET_KEY,
            mount_requirement=Secret.MountType.FILE,
        ),
    ],
)
def publish_model(
    model_dir: FlyteDirectory,
    config: flyte_llama.train.TrainerConfig,
) -> str:
    model_dir.download()
    model_dir = Path(model_dir.path)
    ctx = current_context()

    try:
        hf_auth_token = ctx.secrets.get(SECRET_GROUP, HF_HUB_API_SECRET_KEY)
    except Exception:
        hf_auth_token = None

    return flyte_llama.publish.publish_to_hf_hub(model_dir, config, hf_auth_token)


@task(container_image=image_spec)
def batch_size_tuning_configs(
    config: flyte_llama.train.TrainerConfig,
    batch_sizes: List[int],
) -> List[flyte_llama.train.TrainerConfig]:
    configs = []
    for batch_size in batch_sizes:
        configs.append(replace(config, batch_size=batch_size))
    return configs


@workflow
def tune_batch_size_maptask(
    config: flyte_llama.train.TrainerConfig,
    batch_sizes: List[int],
):
    dataset = create_dataset()
    configs = batch_size_tuning_configs(config=config, batch_sizes=batch_sizes)
    map_task(
        partial(train, dataset=dataset, pretrained_adapter=None),
        min_success_ratio=0.0,
    )(config=configs).with_overrides(retries=0)


@eager(
    remote=FlyteRemote(
        # config=Config.for_endpoint("demo.hosted.unionai.cloud"),
        config=Config(
            platform=PlatformConfig(
                endpoint="demo.hosted.unionai.cloud",
                client_id="eager-workflows",
            ),
        ),
        default_project="llm-fine-tuning",
        default_domain="development",
    ),
    client_secret_group="arn:aws:secretsmanager:us-east-2:356633062068:secret:",
    client_secret_key="eager-workflow-LhHnqo",
    local_entrypoint=True,
)
async def tune_batch_size_eager(
    config: flyte_llama.train.TrainerConfig,
    batch_sizes: List[int],
) -> int:
    dataset = await create_dataset(additional_urls=None)
    results = []
    for batch_size in batch_sizes:
        config.batch_size = batch_size
        results.append(
            train(
                dataset=dataset,
                config=replace(config, batch_size=batch_size),
                pretrained_adapter=None,
            )
        )
    results = await asyncio.gather(*results, return_exceptions=True)

    best_batch_size = 0
    for batch_size, result in zip(batch_sizes, results):
        if isinstance(result, Exception):
            continue
        best_batch_size = batch_size
    return best_batch_size
