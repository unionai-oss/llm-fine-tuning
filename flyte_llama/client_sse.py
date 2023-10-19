import time
import typing
from urllib.parse import urljoin

import httpx
import modelz
from httpx_sse import connect_sse

N_RETRIES = 1000
SLEEP_DURATION = 30
DEFAULT_TIMEOUT = httpx.Timeout(30, connect=6_000, read=6_000, write=6_000)


NOT_READY_MSG = "no subsets for"


def check_availability(
    httpx_client: httpx.Client,
    inference_url: str,
    prompt: str,
    timeout: typing.Union[int, httpx.Timeout],
):
    for _ in range(N_RETRIES):
        response = httpx_client.post(
            inference_url, json={"prompt": prompt}, timeout=timeout
        )
        if response.status_code >= 500:
            time.sleep(SLEEP_DURATION)
            continue

        message = ""
        try:
            response_json = response.json()
            message = response_json["message"]
        except ValueError:
            continue

        if NOT_READY_MSG in message:
            time.sleep(SLEEP_DURATION)
            continue

        break


def infer_stream(
    prompt: str,
    api_key: str,
    deployment_key: str,
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

    check_availability(httpx_client, inference_url, prompt, timeout)

    with connect_sse(
        httpx_client,
        "POST",
        inference_url,
        json={"prompt": prompt},
        timeout=timeout,
    ) as event_source:
        for sse in event_source.iter_sse():
            print(f"Event({sse.event}): {sse.data}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--deployment-key", required=True)
    parser.add_argument("--n-retries", default=N_RETRIES, type=int)
    parser.add_argument("--sleep-interval", default=None, type=int)

    args = parser.parse_args()
    infer_stream(**vars(args))
