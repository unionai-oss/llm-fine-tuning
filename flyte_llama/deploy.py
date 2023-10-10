import json
import typing

import requests


ServerResource = typing.Literal[
    "cpu-4c-16g",
    "nvidia-tesla-t4-4c-16g",
    "nvidia-ampere-a100-40g-12c-85g",
    "nvidia-ada-l4-8c-32g",
    "nvidia-ada-l4-2-24c-96g",
    "nvidia-ada-l4-4-48c-192g",
]


def deploy_model(
    user_id: str,
    api_key: str,
    deployment_name: str,
    image: str,
    server_resource: ServerResource,
    env_vars: typing.Dict[str, str],
):
    """Deploy model to Modelz

    https://modelz-api.readme.io/reference/post_users-login-name-clusters-cluster-id-deployments
    """
    url = f"https://cloud.modelz.ai/api/v1/users/{user_id}/clusters/modelz/deployments"

    headers = {
        "accept": "application/json",
        "X-API-KEY": api_key,
    }
    payload = {
        "spec": {
            "deployment_source": {
                "docker": {"image": image},
            },
            "image_config": {"enable_cache_optimize": True},
            "env_vars": env_vars,
            "framework": "mosec",
            "max_replicas": 1,
            "min_replicas": 0,
            "name": deployment_name,
            "server_resource": server_resource,
            "startup_duration": 6_000,
            "target_load": 10,
            "templateId": "string",
            "zero_duration": 1200,
        }
    }
    deployment_response = requests.post(
        url,
        headers={"content-type": "application/json", **headers},
        json=payload,
    )
    deployment_json = deployment_response.json()
    deployment_meta = requests.get(
        f"{url}/{deployment_json['spec']['id']}",
        headers=headers,
    )
    deployment_status = deployment_meta.json()["status"]
    deployment_key = (
        deployment_status["endpoint"]
        .replace("https://", "")
        .replace(".modelz.io", "")
    )

    print(f"🚀 Deployment:\n{json.dumps(deployment_json, indent=4)}")
    print(f"📊 Status:\n{json.dumps(deployment_status, indent=4)}")
    print(f"🔑 Deployment Key: {deployment_key}")


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id", type=str, required=True)
    parser.add_argument("--api-key", type=str, required=True)
    parser.add_argument("--deployment-name", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument(
        "--server-resource",
        type=str,
        required=False,
        choices=typing.get_args(ServerResource),
        default="nvidia-ada-l4-2-24c-96g",
    )

    args = parser.parse_args()

    deploy_model(
        **vars(args),
        env_vars={"HF_AUTH_TOKEN": os.environ["HF_AUTH_TOKEN"]},
    )
