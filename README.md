# LLM Fine Tuning

A fine-tuning project for LLMs on Flyte


## Setup

```bash
python -m venv ~/venvs/llm-fine-tuning
source ~/venvs/llm-fine-tuning/bin/activate
pip install -r requirements.txt
```

## Container Build

```bash
docker build . -t <registry>/unionai-llm-fine-tuning:latest
docker push <registry>/unionai-llm-fine-tuning:latest
```

## Run on Flyte

### Create Project (Optional)

First, [install flytectl](https://docs.flyte.org/projects/flytectl/en/latest/).

Then, create a project with:

```bash
flytectl --config $FLYTECTL_CONFIG create project  \
  --id "llm-fine-tuning" \
  --description "Fine-tuning for LLMs" \
  --name "llm-fine-tuning"
```

### Set Environment Variables

```bash
export FLYTECTL_CONFIG=...
export REGISTRY=...

# if you created the "llm-fine-tuning" project
export FLYTE_PROJECT=llm-fine-tuning

# otherwise default to flytesnacks
export FLYTE_PROJECT=flytesnacks
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
    --image $REGISTRY/unionai-llm-fine-tuning:latest \
    --project $FLYTE_PROJECT \
    llm_fine_tuning.py train \
    --model_args "$(cat config/model_args.json)" \
    --data_args "$(cat config/data_args.json)" \
    --training_args "$(cat config/training_args.json)" \
    --fsdp '["full_shard", "auto_wrap", "offload"]' \
    --fsdp_config "$(cat config/fsdp_config.json)" \
    --ds_config '{}'
```

#### Using DeepSpeed

```bash
pyflyte --config $FLYTECTL_CONFIG run --remote \
    --image $REGISTRY/unionai-llm-fine-tuning:latest \
    --project $FLYTE_PROJECT \
    llm_fine_tuning.py train \
    --model_args "$(cat config/model_args.json)" \
    --data_args "$(cat config/data_args.json)" \
    --training_args "$(cat config/training_args.json)" \
    --fsdp '[]' \
    --fsdp_config '{}' \
    --ds_config "$(cat config/deepspeed_config.json)"
```

### Fine-tuning with LoRA

The following instructions are for fine-tuning using [LoRA](https://arxiv.org/abs/2106.09685)

```bash
pyflyte --config $FLYTECTL_CONFIG run --remote \
    --image $REGISTRY/unionai-llm-fine-tuning:latest \
    --project $FLYTE_PROJECT \
    llm_fine_tuning_lora.py train \
    --base_model "huggyllama/llama-7b" \
    --data_path "yahma/alpaca-cleaned" \
    --output_dir "./tmp" \
    --batch_size 8 \
    --micro_batch_size 1 \
    --num_epochs 1 \
    --learning_rate 1e-4 \
    --cutoff_len 512 \
    --save_steps 30 \
    --lora_r 1 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '["q_proj", "v_proj"]' \
    --train_on_inputs \
    --group_by_length \
    --resume_from_checkpoint "" \
    --ds_config "$(cat config/deepspeed_config.json)"
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
