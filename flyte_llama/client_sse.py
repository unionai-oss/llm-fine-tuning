import time
import typing
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urljoin

import httpx
import modelz
from httpx_sse import connect_sse
from httpx_sse._exceptions import SSEError

N_RETRIES = 1000
SLEEP_DURATION = 10
DEFAULT_TIMEOUT = httpx.Timeout(30, connect=6_000, read=6_000, write=6_000)


NOT_READY_MSG = "no subsets for"


@contextmanager
def sse_connection(
    httpx_client: httpx.Client,
    inference_url: str,
    prompt: str,
    n_tokens: int,
    timeout: typing.Union[int, httpx.Timeout],
):
    print("\n🦙 Invoking Flyte Llama:")
    print("🔎 Checking availability")
    for i in range(N_RETRIES):
        if i > 0:
            print(f"🔄 Retry: {i}", end="\r", flush=True)

        with connect_sse(
            httpx_client,
            "POST",
            inference_url,
            json={"prompt": prompt, "n_tokens": n_tokens},
            timeout=timeout,
        ) as event_source:
            try:
                event_source._check_content_type()
            except SSEError:
                time.sleep(SLEEP_DURATION)
                continue

            print("✅ Ready")
            yield event_source
            break


def infer_stream(
    prompt: str,
    n_tokens: int,
    api_key: str,
    deployment_key: str,
    output_file: str,
    timeout: typing.Union[int, httpx.Timeout] = DEFAULT_TIMEOUT,
    **kwargs,
):
    client = modelz.ModelzClient(
        key=api_key,
        deployment=deployment_key,
        timeout=timeout,
    )
    root_url = client.host.format(deployment_key)
    inference_url = urljoin(root_url, "/inference")
    httpx_client: httpx.Client = client.client

    prev_msg = prompt
    with sse_connection(
        httpx_client,
        inference_url,
        prompt,
        n_tokens,
        timeout=timeout,
    ) as event_source:
        print("▶️ Output")
        print(prompt, end="", flush=True)
        for sse in event_source.iter_sse():
            msg = sse.data
            if msg.startswith("<s>"):
                msg = msg[4:]
            if "</s>" in msg:
                msg = msg[: msg.index("</s>")]
                break
            print_msg = msg[len(prev_msg):]
            print(print_msg, end="", flush=True)
            prev_msg = msg

    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w") as f:
            f.write(msg)


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--n-tokens", required=False, default=500, type=int)
    parser.add_argument("--output-file", required=False, default=None)
    parser.add_argument("--api-key", required=False, default=os.environ.get("MODELZ_API_KEY"))
    parser.add_argument("--deployment-key", required=True)
    parser.add_argument("--n-retries", default=N_RETRIES, type=int)
    parser.add_argument("--sleep-interval", default=None, type=int)

    args = parser.parse_args()
    infer_stream(**vars(args))
