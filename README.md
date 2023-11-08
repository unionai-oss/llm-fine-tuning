# 🤖🔧 LLM Fine Tuning

![](static/llm-fine-tuning.png)

This repository contains the [Union.ai](https://union.ai/) open source codebase
for running LLM fine-tuning jobs on Flyte or Union Cloud.

## 💻 Setup

```bash
python -m venv ~/venvs/llm-fine-tuning
source ~/venvs/llm-fine-tuning/bin/activate
pip install -r requirements.txt
```

### Export Environment Variables

```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
export FLYTECTL_CONFIG=~/.uctl/config-demo.yaml  # replace with your flyte/union cloud config
export REGISTRY=ghcr.io/unionai-oss  # replace this with your own registry
export FLYTE_PROJECT=llm-fine-tuning
export IMAGE=ghcr.io/unionai-oss/unionai-llm-fine-tuning:de445a0
```

## 🐳 Container Build [Optional]

This repository comes with a pre-built image for running the fine-tuning workflows,
but if you want to build your own, follow these instructions to build an image
with `transformers` and `deepspeed` pre-built.

```bash
docker login ghcr.io
gitsha=$(git rev-parse --short=7 HEAD)
image_name=$REGISTRY/unionai-llm-fine-tuning
docker build . -t $image_name:$gitsha -f Dockerfile
docker push $image_name:$gitsha
```

## 🚀 Run on a Flyte or Union Cloud Cluster

The instructions below will run on your Flyte or Union Cloud cluster assuming
you have the follow prerequisites:

- The [flyte pytorch plugin](https://docs.flyte.org/en/latest/deployment/plugins/k8s/index.html#spin-up-a-cluster) is enabled on your cluster.
- The [pytorch kubernetes operator](https://docs.flyte.org/en/latest/deployment/plugins/k8s/index.html#install-the-kubernetes-operator) is installed on your cluster.

### Create Project

First, [install flytectl](https://docs.flyte.org/projects/flytectl/en/latest/).

Then, create a project with:

```bash
flytectl --config $FLYTECTL_CONFIG create project  \
  --id "llm-fine-tuning" \
  --description "Fine-tuning for LLMs" \
  --name "llm-fine-tuning"
```

### 🔀 Fine-tuning Workflows

The `fine_tuning` directory contains Flyte tasks and workflows for fine-tuning
LLMs:

- `fine_tuning/llm_fine_tuning.py`: Full fine-tuning using DeepSpeed
- `fine_tuning/llm_tuning_lora.py`: Parameter-efficient fine-tuning using LoRA
- `fine_tuning/llm_tuning_qlora.py`: Parameter-efficient fine-tuning using 4-bit QLoRA

#### ⚙️ Configuration

The `config` directory contains `*.json` files which correspond to different
configurations for fine-tuning. These are used to determine the model, dataset,
training arguments, and model publishing arguments.

### 👟 Execute Fine-tuning Workflows on the CLI

<details>
<summary>Local Fine-tuning (facebook/opt-125m)</summary>
<p>

```bash
pyflyte run \
    fine_tuning/llm_fine_tuning.py fine_tune \
    --config config/training_config_local.json \
    --deepspeed_config "{}"
```

</p>
</details>

#### RedPajama Fine-tuning

<details>
<summary>Full Fine-tuning (togethercomputer/RedPajama-INCITE-Base-3B-v1)</summary>

<p>

```bash
pyflyte --config $FLYTECTL_CONFIG \
    run --remote \
    --copy-all \
    --project $FLYTE_PROJECT \
    --image $IMAGE \
    fine_tuning/llm_fine_tuning.py fine_tune \
    --config config/training_config_redpajama_3b.json \
    --deepspeed_config config/deepspeed.json
```

</p>
</details>


<details>
<summary>LoRA Fine-tuning (togethercomputer/RedPajama-INCITE-7B-Base)</summary>
<p>

```bash
pyflyte --config $FLYTECTL_CONFIG \
    run --remote \
    --copy-all \
    --project $FLYTE_PROJECT \
    --image $IMAGE \
    fine_tuning/llm_fine_tuning_lora.py fine_tune \
    --config config/training_config_redpajama_7b_lora.json
```

</p>
</details>

#### Llama2 Fine-tuning

<details>
<summary>Full Fine-tuning (meta-llama/Llama-2-7b-hf)</summary>
<p>

```bash
pyflyte --config $FLYTECTL_CONFIG \
    run --remote \
    --copy-all \
    --project $FLYTE_PROJECT \
    --image $IMAGE \
    fine_tuning/llm_fine_tuning.py fine_tune \
    --config config/training_config_llama2_7b.json \
    --deepspeed_config config/deepspeed.json
```

</p>
</details>


<details>
<summary>Full Fine-tuning (meta-llama/Llama-2-13b-hf)</summary>
<p>

```bash
pyflyte --config $FLYTECTL_CONFIG \
    run --remote \
    --copy-all \
    --project $FLYTE_PROJECT \
    --image $IMAGE \
    fine_tuning/llm_fine_tuning.py fine_tune \
    --config config/training_config_llama2_13b.json \
    --deepspeed_config config/deepspeed_llama2_13b.json
```

</p>
</details>


<details>
<summary>QLoRA Fine-tuning (meta-llama/Llama-2-13b-hf)</summary>
<p>

```bash
pyflyte --config $FLYTECTL_CONFIG \
    run --remote \
    --copy-all \
    --project $FLYTE_PROJECT \
    --image $IMAGE \
    fine_tuning/llm_fine_tuning_qlora.py fine_tune \
    --config config/training_config_llama2_13b_qlora.json
```

</p>
</details>

<details>
<summary>QLoRA Fine-tuning (meta-llama/Llama-2-70b-hf)</summary>
<p>

```bash
pyflyte --config $FLYTECTL_CONFIG \
    run --remote \
    --copy-all \
    --project $FLYTE_PROJECT \
    --image $IMAGE \
    fine_tuning/llm_fine_tuning_qlora.py fine_tune \
    --config config/training_config_llama2_70b_qlora.json
```

</p>
</details>
