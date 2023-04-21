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

## Configuration

- Update the arguments in the `.json` files in the `config` directory. These
  will be used to inform which model, dataset, and training arguments are used.
- In the `train.py` script, replace the `WANDB_API_KEY: <wandb_api_key>`
  environment variable in the `train` task and use your own key.

## Run on Flyte

To run on flyte:

```bash
pyflyte --config <path_to_config> run --remote \
    --image <registry>/unionai-llm-fine-tuning:latest \
    train.py train \
    --model_args="$(cat config/model_args.json)" \
    --data_args="$(cat config/data_args.json)" \
    --training_args="$(cat config/training_args.json)"
```
