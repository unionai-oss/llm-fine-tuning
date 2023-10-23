import yaml
from dataclasses import asdict

from io import BytesIO
from pathlib import Path
from typing import Optional

import huggingface_hub as hh

from flyte_llama.train import TrainerConfig, PublishConfig


MODEL_CARD_TEMPLATE = """
---
{model_card_content}
---

{readme_content}
""".strip()


def publish_to_hf_hub(
    model_dir: Path,
    config: TrainerConfig,
    hf_auth_token: str,
    quantized_8bit: Optional[bool] = None,
) -> str:
    # make sure the file can be downloaded
    publish_config: PublishConfig = config.publish_config

    model_dir
    hh.login(token=hf_auth_token)
    api = hh.HfApi()
    if quantized_8bit:
        repo_id = f"{publish_config.repo_id}-8bit"
    else:
        repo_id = publish_config.repo_id

    repo_url = api.create_repo(repo_id, exist_ok=True)

    if publish_config.readme is not None:
        model_card_dict = publish_config.model_card.to_dict()

        config_dict = asdict(config)

        dataset_path = config_dict.get("data_path", None)
        if dataset_path:
            model_card_dict["datasets"] = [config_dict.get("data_path")]

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
        folder_path=model_dir,
        ignore_patterns=["flyte*", "models--*", "tmp*"]
    )
    return str(repo_url)
