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


def improve_slack_response(input, output, model, tokenizer, stop_token_ids):
    class StopOnTokens(StoppingCriteria):
        def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
        ) -> bool:
            for stop_id in stop_token_ids:
                if input_ids[0][-1] == stop_id:
                    return True
            return False

    prompt = f"""
Instruction:
<|im_start|>
You are a helpful slack bot.
You provide answers to user questions on Slack.
Given a user question, your job is to provide an answer to it.
Take help from the context and ensure that your answer appears as if it were provided by a bot, without expressing any personal opinions.
Avoid referencing the context and focus on addressing the user's question directly.
The original user answer consists of responses from multiple users, but your answer has to have a bot-like tone.

<|im_end|>
User question:
<|im_start|>
{input}
<|im_end|>
Context:
<|im_start|>
{output}
<|im_end|>
Your answer:<|im_start|>
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs["input_ids"].to(model.device),
        max_new_tokens=1536,
        do_sample=True,
        temperature=0.3,
        top_p=1.0,
        top_k=50,
        repetition_penalty=1.1,
        stopping_criteria=StoppingCriteriaList([StopOnTokens()]),
    )
    output = tokenizer.batch_decode(outputs)
    return output[0].split("Your answer:")[1].strip()


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
        "mosaicml/mpt-7b-chat",
        torch_dtype=torch.bfloat16,  # Load model weights in bfloat16
        trust_remote_code=True,
        load_in_8bit=True,
        device_map="auto",
        max_seq_len=8192,
    )
    tok = AutoTokenizer.from_pretrained("mosaicml/mpt-7b-chat", trust_remote_code=True)
    stop_token_ids = tok.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"])

    for ts in threads[:5]:
        if len(ts["output"]) > 180:
            pairs.append(
                {
                    "input": ts["input"],
                    "output": improve_slack_response(
                        input=ts["input"],
                        output=ts["output"],
                        model=model,
                        tokenizer=tok,
                        stop_token_ids=stop_token_ids,
                    ),
                }
            )
            print(pairs)
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
