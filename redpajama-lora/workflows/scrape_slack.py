import json
import os
from glob import glob
from typing import List, Optional

import flytekit
from datasets import load_dataset
from flytekit import Secret, map_task, task, workflow
from flytekit.types.file import FlyteFile
import re
from pathlib import Path

SECRET_GROUP = "hf"
SECRET_NAME = "hf-token"


@task(cache=True, cache_version="1.0")
def get_channel_dirs(parent_dir: str) -> List[str]:
    return glob(parent_dir + "/*/")


def replace_user_id_with_name(text, user_mapping):
    pattern = r"<@(.*?)>"
    user_ids = re.findall(pattern, text)
    for user_id in user_ids:
        if user_id in user_mapping:
            user_name = user_mapping[user_id]
            text = text.replace(f"<@{user_id}>", user_name)
    return text


@task(cache=True, cache_version="1.0")
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
                        "input": (
                            message.get("user", "bot"),
                            replace_user_id_with_name(message["text"], user_mapping),
                        ),
                        "output": [],
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
                    ].append(
                        (
                            message.get("user", "bot"),
                            replace_user_id_with_name(message["text"], user_mapping),
                        )
                    )

    pairs = []
    for ts in threads:
        input_messages = {}
        input_messages[ts["input"][0]] = f"{ts['input'][1]}\n"
        output_messages = {}

        output_set = False
        for output_message in ts["output"]:
            user = output_message[0]
            if (
                (user not in input_messages)
                and (user not in output_messages)
                and (not output_messages)
            ):
                output_messages[user] = f"{output_message[1]}\n"
                output_set = True
            elif (
                (user not in input_messages)
                and (user not in output_messages)
                and output_messages
            ) or (user in input_messages and output_set):
                pairs.append(
                    {
                        "input": list(input_messages.values())[0],
                        "output": list(output_messages.values())[0],
                    }
                )
                input_messages = output_messages.copy()
                output_messages.clear()
                output_messages[user] = f"{output_message[1]}\n"
            elif user in input_messages and (not output_set):
                input_messages[user] += f"{output_message[1]}\n"
            elif user in output_messages:
                output_messages[user] += f"{output_message[1]}\n"

        if input_messages and output_messages:
            pairs.append(
                {
                    "input": list(input_messages.values())[0],
                    "output": list(output_messages.values())[0],
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


def merge_json_files(json_files):
    json_result_file = os.path.join(
        flytekit.current_context().working_directory, "flyte_slack_data.json"
    )
    result = []
    for json_file in json_files:
        if json_file:
            with open(json_file, "r") as infile:
                result.extend(json.load(infile))
    with open(
        json_result_file,
        "w",
    ) as f:
        json.dump(result, f)
    return json_result_file


@task(
    secret_requests=[
        Secret(
            group=SECRET_GROUP, key=SECRET_NAME, mount_requirement=Secret.MountType.FILE
        )
    ],
)
def push_to_hub(json_files: List[Optional[FlyteFile]]):
    HF_TOKEN = flytekit.current_context().secrets.get(SECRET_GROUP, SECRET_NAME)
    dataset = load_dataset("json", data_files=merge_json_files(json_files))
    dataset.push_to_hub("Samhita/flyte-slack-data", token=HF_TOKEN)


@workflow
def slack_scraper(parent_dir: str = "flyte-slack-data"):
    push_to_hub(
        json_files=map_task(question_response_pairs)(
            channel_dir=get_channel_dirs(parent_dir=parent_dir)
        )
    )
