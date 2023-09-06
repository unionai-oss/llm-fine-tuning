import json
import os
import re
from glob import glob
from pathlib import Path
from typing import List, Optional

import flytekit
import torch
from flytekit import Resources, map_task, task, workflow
from flytekit.types.file import FlyteFile
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)


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


def improve_slack_response(input, output, model, tokenizer):
    class StopWordsCriteria(StoppingCriteria):
        def __init__(self, tokenizer, stop_words):
            self._tokenizer = tokenizer
            self._stop_words = stop_words
            self._partial_result = ""
            self._stream_buffer = ""

        def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
        ) -> bool:
            first = not self._partial_result
            text = self._tokenizer.decode(input_ids[0, -1])
            self._partial_result += text
            for stop_word in self._stop_words:
                if stop_word in self._partial_result:
                    return True

            return False

    prompt = f"""Instruction: You are a helpful slack bot. You provide answers to user questions on Slack. Given a user question, your job is to provide an answer to it. Take help from the context and ensure that your answer appears as if it were provided by a bot, without expressing any personal opinions. Avoid referencing the context and focus on addressing the user's question directly. The original user answer consists of responses from multiple users, but your answer has to have a bot-like tone.
User question: {input}
Context: {output}
Answer:
    """
    print(prompt)
    stop_criteria = StopWordsCriteria(tokenizer, ["<human>"])
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.3,
        top_p=0.7,
        top_k=50,
        repetition_penalty=1.1,
        return_dict_in_generate=True,
        stopping_criteria=StoppingCriteriaList([stop_criteria]),
    )
    token = outputs.sequences[0, input_length:]
    return tokenizer.decode(token)


@task(
    requests=Resources(gpu="1", mem="50Gi", cpu="10"),
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
                        "input": message.get("user", "bot")
                        + ": ```"
                        + replace_user_id_with_name(message["text"], user_mapping)
                        + "```",
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
    model = AutoModelForCausalLM.from_pretrained(
        "togethercomputer/RedPajama-INCITE-7B-Chat",
        torch_dtype=torch.float16,
        load_in_8bit=True,
        device_map="auto",
    )
    tok = AutoTokenizer.from_pretrained("togethercomputer/RedPajama-INCITE-7B-Chat")

    for ts in threads[:5]:
        if len(ts["output"]) > 180:
            pairs.append(
                {
                    "input": ts["input"],
                    "output": improve_slack_response(
                        input=ts["input"],
                        output=ts["output"][: (2048 - (len(ts["input"]) + 100))],
                        model=model,
                        tokenizer=tok,
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
