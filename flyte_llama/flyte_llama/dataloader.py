"""Load data for training."""

from pathlib import Path
from typing import Iterable, Optional, TypedDict

from datasets import Dataset


class Example(TypedDict):
    text: str


def iter_reader(data_path: Path) -> Iterable[Example]:
    for fp in data_path.glob("*/*"):
        if "metadata" in fp.parts:
            continue
        with fp.open() as f:
            yield Example({"text": f.read()})


def chunk_texts(examples, block_size: int = 1024, skip_by: int = 128):
    chunks = []
    for text in examples["text"]:
        text_len = len(text)

        if text_len < block_size:
            chunks.append(text)
            continue

        for i in range(0, len(text), skip_by):
            chunks.append(text[i : i + block_size])

    return {"text": chunks}


def get_dataset(data_path: Path, num_proc: int = 1, limit: Optional[int] = None, **kwargs):
    dataset = Dataset.from_list([*iter_reader(data_path)])

    if limit:
        dataset = dataset.select(range(limit))

    dataset = dataset.map(
        chunk_texts, batched=True, fn_kwargs=kwargs, num_proc=num_proc,
    )
    return dataset


if __name__ == "__main__":
    data_path = Path.home() / "datasets" / "flyte_llama"
    dataset = get_dataset(data_path)
