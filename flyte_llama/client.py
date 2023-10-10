import time
import typing
from pathlib import Path

import httpx
import modelz


N_RETRIES = 1000
SLEEP_DURATION = 30
DEFAULT_TIMEOUT = httpx.Timeout(30, connect=6_000, read=6_000, write=6_000)


def infer(
    prompt: str,
    output_file: str,
    api_key: str,
    deployment_key: str,
    n_retries: int = 1_000,
    timeout: typing.Union[int, httpx.Timeout] = DEFAULT_TIMEOUT,
    sleep_interval: int = SLEEP_DURATION,
):
    client = modelz.ModelzClient(
        key=api_key,
        deployment=deployment_key,
        timeout=timeout,
    )

    start = time.time()
    output_file = Path(output_file)
    if not output_file.parent.exists():
        output_file.parent.mkdir(parents=True)

    for i in range(n_retries):

        if i > 0:
            print(f"Retry: {i}")

        resp = client.inference(params=prompt, serde="msgpack")

        try:
            file = output_file
            resp.save_to_file(file)
            print(f"wrote file {file}")
            print(resp.data.decode())
            break
        except Exception:
            pass

        message = ""
        try:
            json = resp.resp.json()
            message = json["message"]
        except:
            continue

        if message.startswith("no addresses for"):
            time.sleep(sleep_interval)
            continue

    print(f"inference time: {time.time() - start}")


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
    infer(**vars(args))
