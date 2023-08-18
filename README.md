# LLM Fine Tuning

A fine-tuning project for LLMs on Flyte


## Setup

```bash
python -m venv ~/venvs/llm-fine-tuning
source ~/venvs/llm-fine-tuning/bin/activate
pip install -r requirements.txt
```

### Set Environment Variables

```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
# export FLYTECTL_CONFIG=~/.uctl/config-demo.yaml  # replace this with your own flyte config
export FLYTECTL_CONFIG=~/.flyte/dev-config.yaml  # replace this with your own flyte config
export REGISTRY=ghcr.io/unionai-oss  # replace this with your own registry
export FLYTE_SDK_LOGGING_LEVEL=100
export FLYTE_PROJECT=llm-fine-tuning
# export IMAGE=ghcr.io/unionai-oss/unionai-llm-fine-tuning:fbba7c0c68b38d3bcd4e11c1b214feb51812a9f0
# export IMAGE=ghcr.io/unionai-oss/unionai-llm-fine-tuning:d98ba52
# export IMAGE=ghcr.io/unionai-oss/unionai-llm-fine-tuning:718398b
export IMAGE=ghcr.io/unionai-oss/unionai-llm-fine-tuning:de445a0
# export IMAGE=ghcr.io/unionai-oss/unionai-llm-fine-tuning:505c34d
```

## Container Build

Build a base image that has transformers and deepspeed pre-built.

```bash
docker login ghcr.io
gitsha=$(git rev-parse --short=7 HEAD)
image_name=$REGISTRY/unionai-llm-fine-tuning
docker build . -t $image_name:$gitsha -f Dockerfile
docker push $image_name:$gitsha
```

## Run on Flyte

### Create Project

First, [install flytectl](https://docs.flyte.org/projects/flytectl/en/latest/).

Then, create a project with:

```bash
flytectl --config $FLYTECTL_CONFIG create project  \
  --id "llm-fine-tuning" \
  --description "Fine-tuning for LLMs" \
  --name "llm-fine-tuning"
```

### Full Fine-tuning

The following instructions are for full fine-tuning of a pre-trained model.

#### Configuration

Update the arguments in the `.json` files in the `config` directory. These
will be used to inform which model, dataset, and training arguments are used.

Run locally:

```bash
pyflyte run \
    fine_tuning/llm_fine_tuning.py fine_tune \
    --config config/training_config_local.json \
    --publish_config config/publish_config.json \
    --deepspeed_config "{}"
```

### Full Fine-tuning on Flyte

```bash
pyflyte --config $FLYTECTL_CONFIG \
    run --remote \
    --copy-all \
    --project $FLYTE_PROJECT \
    --image $IMAGE \
    fine_tuning/llm_fine_tuning.py fine_tune \
    --config config/training_config.json \
    --publish_config config/publish_config.json \
    --deepspeed_config config/zero_config_ds.json
```

### Fine-tuning with LoRA

The following instructions are for fine-tuning using [LoRA](https://arxiv.org/abs/2106.09685)

```bash
pyflyte --config $FLYTECTL_CONFIG \
    run --remote \
    --copy-all \
    --project $FLYTE_PROJECT \
    --image $IMAGE \
    fine_tuning/llm_fine_tuning_lora.py fine_tune \
    --config config/training_config_lora.json \
    --publish_config config/publish_config_lora.json
```

## Llama2 Fine-tuning

### Full Fine-tuning

```bash
pyflyte --config $FLYTECTL_CONFIG \
    run --remote \
    --copy-all \
    --project $FLYTE_PROJECT \
    --image $IMAGE \
    fine_tuning/llm_fine_tuning.py fine_tune \
    --config config/training_config_llama2.json \
    --publish_config config/publish_config_llama2.json \
    --deepspeed_config config/zero_config_ds.json
```

### Fine-tuning with 8-bit LoRA

The following instructions are for fine-tuning using [LoRA](https://arxiv.org/abs/2106.09685)

```bash
pyflyte --config $FLYTECTL_CONFIG \
    run --remote \
    --copy-all \
    --project $FLYTE_PROJECT \
    --image $IMAGE \
    fine_tuning/llm_fine_tuning_lora.py fine_tune \
    --config config/training_config_llama2_lora.json \
    --publish_config config/publish_config_llama2_lora.json
```

### Fine-tuning with QLoRA

```bash
pyflyte --config $FLYTECTL_CONFIG \
    run --remote \
    --copy-all \
    --project $FLYTE_PROJECT \
    --image $IMAGE \
    fine_tuning/llm_fine_tuning_qlora.py fine_tune \
    --config config/training_config_llama2_qlora.json \
    --publish_config config/publish_config_llama2_qlora.json
```
