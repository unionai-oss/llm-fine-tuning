"""Create dataset for Flyte Llama fine-tuning.

This dataset should contain documents from the Flyte repositories for language
model fine-tuning.
"""

import itertools
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable, Optional

from git import Repo


ROOT_URL = "https://github.com/"
REPO_URLS = [
    f"{ROOT_URL}flyteorg/flyte",
    f"{ROOT_URL}flyteorg/flytekit",
    f"{ROOT_URL}flyteorg/flytepropeller",
    f"{ROOT_URL}flyteorg/flyteplugins",
    f"{ROOT_URL}flyteorg/flyteidl",
    f"{ROOT_URL}flyteorg/flyteadmin",
    f"{ROOT_URL}flyteorg/flyteconsole",
    f"{ROOT_URL}flyteorg/flytesnacks",
    f"{ROOT_URL}flyteorg/flyte-conference-talks",
]


def iter_github_documents(
    url: str,
    extensions: Optional[list[str]] = None,
    exclude_files: Optional[list[str]] = None,
    exclude_patterns: Optional[list[str]] = None,
) -> Iterable[str]:
    """Fetch documents from a github url."""
    extensions = extensions or [".py", ".md", ".rst"]
    exclude_files = frozenset(exclude_files or ["__init__.py"])
    exclude_patterns = exclude_patterns or []

    with TemporaryDirectory() as tempdir:
        repo = Repo.clone_from(url, tempdir)
        repo_name = url.split("/")[-1]
        git_sha = repo.head.commit.hexsha
        git_dir = Path(tempdir)

        exclude_from_patterns = [
            *itertools.chain(*(git_dir.glob(p) for p in exclude_patterns))
        ]

        for file in itertools.chain(
            *[git_dir.glob(f"**/*{ext}") for ext in extensions]
        ):
            if file.name in exclude_files or file in exclude_from_patterns:
                continue

            github_url = f"{url}/blob/{git_sha}/{file.relative_to(git_dir)}"
            repo_filepath = file.relative_to(git_dir)
            yield file, repo_name, repo_filepath, github_url


def get_file_name(repo_filepath: Path) -> str:
    return "-".join(
        x.replace("/", "-")
        for x in str(repo_filepath).replace(ROOT_URL, "").split("/")
    )


def create_dataset(
    urls: list[str],
    output_dir: Path,
    **kwargs,
):
    for url in urls:
        print("processing url:", url)
        for file, repo_name, repo_filepath, github_url in iter_github_documents(url, **kwargs):
            out_path = output_dir / repo_name / get_file_name(repo_filepath)
            print(f"writing file: {out_path}")
            shutil.copy(file, out_path)

            metadata = {
                "github_url": github_url,
            }
            metadata_file = out_path.with_suffix(".metadata.json")
            print(f"writing metadata file: {metadata_file}")
            with metadata_file.open() as f:
                json.dump(metadata, f)



if __name__ == "__main__":
    path = Path.home() / "datasets" / "flyte_llama"
    path.mkdir(parents=True, exist_ok=True)
    create_dataset(REPO_URLS, path)
