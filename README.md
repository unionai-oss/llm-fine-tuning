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
export FLYTECTL_CONFIG=~/.uctl/config-demo.yaml
export REGISTRY=ghcr.io/unionai-oss
export FLYTE_SDK_LOGGING_LEVEL=100
export FLYTE_PROJECT=llm-fine-tuning
```

## Container Build

Build a base image that has transformers and deepspeed pre-built.

```bash
docker login ghcr.io
gitsha=$(git rev-parse HEAD)
image_name=$REGISTRY/unionai-llm-fine-tuning
docker build . -t $image_name:$gitsha -f Dockerfile
docker push $image_name:$gitsha
```

```bash
docker build . -t $image_name:$gitsha -f Dockerfile.tmp
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

- Update the arguments in the `.json` files in the `config` directory. These
  will be used to inform which model, dataset, and training arguments are used.
- In the `train.py` script, replace the `WANDB_API_KEY: <wandb_api_key>`
  environment variable in the `train` task and use your own key.

To run on flyte:

```bash
pyflyte --config $FLYTECTL_CONFIG run --remote \
    --project $FLYTE_PROJECT \
    --copy-all \
    fine_tuning/llm_fine_tuning.py train \
    --config config/training_config.json \
    --fsdp_config config/zero_config_fsdp.json \
    --ds_config '{}'
```

#### Using DeepSpeed

```bash
pyflyte --config $FLYTECTL_CONFIG run --remote \
    --copy-all \
    --project $FLYTE_PROJECT \
    fine_tuning/llm_fine_tuning.py fine_tune \
    --config config/training_config.json \
    --ds_config config/zero_config_ds.json \
```

```bash
pyflyte --config $FLYTECTL_CONFIG run --remote \
    --copy-all \
    --project $FLYTE_PROJECT \
    fine_tuning/llm_fine_tuning.py train \
    --config config/training_config.json \
    --ds_config config/zero_config_ds.json \
    --fsdp_config '{}'
```

### Fine-tuning with LoRA

The following instructions are for fine-tuning using [LoRA](https://arxiv.org/abs/2106.09685)

```bash
pyflyte --config $FLYTECTL_CONFIG run --remote \
    --image $REGISTRY/unionai-llm-fine-tuning:latest \
    --project $FLYTE_PROJECT \
    fine_tuning/llm_fine_tuning_lora.py train \
    --config config/training_config_lora.json
```

### Push Fine-tuned Model to Huggingface Hub

#### Configuration

- In the `train.py` script, replace the `HUGGINGFACE_TOKEN: <huggingface_token>`
  environment variable in the `save_to_hf_hub` task and use your own key.

```bash
export HUGGINGFACE_USERNAME=...
export HUGGINGFACE_REPO_NAME=...
export REMOTE_MODEL_PATH=...
```

Pushing trained model to huggingface hub:

```bash
pyflyte --config $FLYTECTL_CONFIG run --remote \
    --image $REGISTRY/unionai-llm-fine-tuning:latest \
    --project $FLYTE_PROJECT \
    llm_fine_tuning.py save_to_hf_hub \
    --model_dir $REMOTE_MODEL_PATH \
    --repo_id $HUGGINGFACE_USERNAME/$HUGGINGFACE_REPO_NAME \
    --model_card "$(cat config/model_card.json)" \
    --readme "# My Fine-tuned Model"
```
