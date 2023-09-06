import json
import os
import re
from glob import glob
from pathlib import Path
from typing import List, Optional

import flytekit
import torch
import transformers
from flytekit import Resources, map_task, task, workflow, Secret
from flytekit.types.file import FlyteFile
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SECRET_GROUP = "hf"
SECRET_NAME = "hf-token"


@task
def get_channel_dirs(parent_dir: str) -> List[str]:
    return [f"{parent_dir}/announcements/"]


def replace_user_id_with_name(text, user_mapping):
    pattern = r"<@(.*?)>"
    user_ids = re.findall(pattern, text)
    for user_id in user_ids:
        if user_id in user_mapping:
            user_name = user_mapping[user_id]
            text = text.replace(f"<@{user_id}>", user_name)
    return text


def improve_slack_response(input, output, token):
    prompt = f"""<s>[INST] <<SYS>> 
You are a helpful slack bot.
You provide answers to user questions on Slack.
Given a user question, your job is to provide an answer to it.
Take help from the context and ensure that your answer appears as if it were provided by a bot, without expressing any personal opinions. 
Avoid referencing the context and focus on addressing the user's question directly. 
The original user answer consists of responses from multiple users, but your answer has to have a bot-like tone.
<</SYS>>

Context: {output}

{input} [/INST]
    """

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        token=token,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        token=token,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id  # for open-ended generation

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        trust_remote_code=True,
        device_map="auto",
    )

    sequences = pipeline(
        prompt,
        do_sample=True,
        top_k=10,
        return_full_text=False,
        temperature=0.4,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return f"Result: {sequences[0]['generated_text']}"


@task(
    requests=Resources(gpu="1", mem="50Gi", cpu="10"),
    secret_requests=[
        Secret(
            group=SECRET_GROUP,
            key=SECRET_NAME,
            mount_requirement=Secret.MountType.FILE,
        )
    ],
)
def question_response_pairs(channel_dir: str) -> Optional[FlyteFile]:
    threads = []
    thread_ts_list_index_pairs = {}
    sorted_list_of_files = sorted(glob(channel_dir + "*.json"), key=os.path.getctime)

    with open(Path(channel_dir).parents[0] / "users.json") as f:
        user_data = json.load(f)
        user_mapping = {}
        for user in user_data:
            if not user["deleted"]:
                user_mapping[user["id"]] = user["real_name"]

    for data_file in sorted_list_of_files:
        with open(data_file) as f:
            list_of_messages = json.load(f)
        for message in list_of_messages:
            if "reply_count" in message and message["reply_count"] > 0:
                threads.append(
                    {
                        "input": replace_user_id_with_name(
                            message["text"], user_mapping
                        ),
                        "output": "",
                    }
                )
                if message["thread_ts"] not in thread_ts_list_index_pairs:
                    thread_ts_list_index_pairs[message["thread_ts"]] = len(threads) - 1
                else:
                    raise ValueError(f"A message with the same thread_ts exists")
            else:
                if (
                    "thread_ts" in message
                    and message["thread_ts"] in thread_ts_list_index_pairs
                ):
                    threads[thread_ts_list_index_pairs[message["thread_ts"]]][
                        "output"
                    ] += (
                        message.get("user", "bot")
                        + ": ```"
                        + replace_user_id_with_name(message["text"], user_mapping)
                        + "```\n"
                    )
    pairs = []
    for ts in threads[:5]:
        if len(ts["output"]) > 180:
            pairs.append(
                {
                    "input": ts["input"],
                    "output": improve_slack_response(
                        input=ts["input"],
                        output=ts["output"][: (4096 - (len(ts["input"]) + 100))],
                        token=flytekit.current_context().secrets.get(
                            SECRET_GROUP, SECRET_NAME
                        ),
                    ),
                }
            )
    if pairs:
        json_file_name = os.path.join(
            flytekit.current_context().working_directory,
            f"flyte_slack_data_{Path(channel_dir).parts[1]}.json",
        )
        with open(json_file_name, "w") as f:
            json.dump(pairs, f)
        return FlyteFile(json_file_name)
    return None


@workflow
def slack_scraper(parent_dir: str = "flyte-slack-data"):
    map_task(question_response_pairs)(
        channel_dir=get_channel_dirs(parent_dir=parent_dir)
    )
