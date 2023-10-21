# Flyte Llama

![](static/flyte_llama.png)

Flyte Llama is a fine-tuned model based on [Code Llama](https://about.fb.com/news/2023/08/code-llama-ai-for-coding/).

## Env Setup

```bash
python -m venv ~/venvs/flyte-llama
source ~/venvs/flyte-llama/bin/activate
pip install -r requirements.txt
```

## Train model

### Export Environment Variables

```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
export FLYTECTL_CONFIG=~/.uctl/config-demo.yaml  # replace with your flyte/union cloud config
export REGISTRY=ghcr.io/unionai-oss  # replace this with your own registry
export FLYTE_PROJECT=llm-fine-tuning
export IMAGE=ghcr.io/unionai-oss/unionai-flyte-llama:2b857ea
```

### 🐳 Container Build

This repository comes with a pre-built image for running the fine-tuning workflows,
but if you want to build your own, follow these instructions to build an image
with `transformers` and `deepspeed` pre-built.

```bash
docker login ghcr.io
gitsha=$(git rev-parse --short=7 HEAD)
image_name=$REGISTRY/unionai-flyte-llama
docker build . -t $image_name:$gitsha -f Dockerfile
docker push $image_name:$gitsha
```

### Create dataset

```bash
python flyte_llama/dataset.py --output-path ~/datasets/flyte_llama
```

### Train Model


<details>
<summary>Local</summary>
<p>

```bash
python flyte_llama/train.py \
    --model_path codellama/CodeLlama-7b-hf \
    --data_dir=~/datasets/flyte_llama \
    --output_dir=~/models/flyte_llama
```

</p>
</details>


<details>
<summary>Flyte Llama 7b Qlora</summary>
<p>

**Train:**

```bash
pyflyte --config $FLYTECTL_CONFIG \
    run --remote \
    --copy-all \
    --project $FLYTE_PROJECT \
    --image $IMAGE \
    flyte_llama/workflows.py train_workflow \
    --config config/flyte_llama_7b_qlora_v0.json
```

**Publish:**

```bash
pyflyte --config $FLYTECTL_CONFIG \
    run --remote \
    --copy-all \
    --project $FLYTE_PROJECT \
    --image $IMAGE \
    flyte_llama/workflows.py publish_model_workflow \
    --config config/flyte_llama_7b_qlora_v0.json \
    --model_dir s3://path/to/model
```


</p>
</details>

<details>
<summary>Flyte Llama 7b Qlora from previous adapter checkpoint</summary>
<p>

Pass in the `--pretrained_adapter` flag to continue training from a previous
adapter checkpoint. This is typically an s3 path produced by `train_workflow`.

```bash
pyflyte --config $FLYTECTL_CONFIG \
    run --remote \
    --copy-all \
    --project $FLYTE_PROJECT \
    --image $IMAGE \
    flyte_llama/workflows.py train_workflow \
    --config config/flyte_llama_7b_qlora_v0.json \
    --pretrained_adapter s3://path/to/checkpoint
```
</p>
</details>

<details>
<summary>Flyte Llama 7b Instruct Qlora</summary>
<p>

**Train:**

```bash
pyflyte --config $FLYTECTL_CONFIG \
    run --remote \
    --copy-all \
    --project $FLYTE_PROJECT \
    --image $IMAGE \
    flyte_llama/workflows.py train_workflow \
    --config config/flyte_llama_7b_instruct_qlora_v0.json
```

**Publish:**

```bash
pyflyte --config $FLYTECTL_CONFIG \
    run --remote \
    --copy-all \
    --project $FLYTE_PROJECT \
    --image $IMAGE \
    flyte_llama/workflows.py publish_model \
    --config config/flyte_llama_7b_instruct_qlora_v0.json \
    --model_dir s3://path/to/model
```


</p>
</details>

<details>
<summary>Flyte Llama 13b Qlora</summary>
<p>

```bash
pyflyte --config $FLYTECTL_CONFIG \
    run --remote \
    --copy-all \
    --project $FLYTE_PROJECT \
    --image $IMAGE \
    flyte_llama/workflows.py train_workflow \
    --config config/flyte_llama_13b_qlora_v0.json
```
</p>
</details>

<details>
<summary>Flyte Llama 34b Qlora</summary>
<p>

```bash
pyflyte --config $FLYTECTL_CONFIG \
    run --remote \
    --copy-all \
    --project $FLYTE_PROJECT \
    --image $IMAGE \
    flyte_llama/workflows.py train_workflow \
    --config config/flyte_llama_34b_qlora_v0.json
```
</p>
</details>

### Serve model

This project uses [ModelZ](https://modelz.ai/) as the serving layer.

Create a `secrets.txt` file to hold your sensitive credentials:

```bash
# do this once
echo MODELZ_USER_ID="<replace>" >> secrets.txt
echo MODELZ_API_KEY="<replace>" >> secrets.txt
echo HF_TOKEN="<replace>" >> secrets.txt
```

<details>
<summary>Serving a POST Endpoint</summary>
<p>

Export env vars:

```bash
eval $(sed 's/^/export /g' secrets.txt)
export VERSION=$(git rev-parse --short=7 HEAD)
export SERVING_IMAGE=ghcr.io/unionai-oss/modelz-flyte-llama-serving:$VERSION
```

Build the serving image:

```bash
docker build . -f Dockerfile.server \
    --build-arg "HF_TOKEN=$HF_TOKEN" \
    -t $SERVING_IMAGE
```

Push it:

```bash
docker push $SERVING_IMAGE
```

Deploy:

```bash
python deploy.py \
    --user-id $MODELZ_USER_ID \
    --api-key $MODELZ_API_KEY \
    --deployment-name flyte-llama-$VERSION \
    --image $SERVING_IMAGE \
    --server-resource "nvidia-ada-l4-2-24c-96g"
```

Get the `deployment_key` from the output of the command above and use it to test
the model:

```bash
python client.py \
    --prompt "The code snippet below shows a basic Flyte workflow" \
    --output-file output.txt \
    --api-key $MODELZ_API_KEY \
    --deployment-key <deployment_key>
```

</p>
</details>

<details>
<summary>Serving a Server Streaming Events (SSE) Endpoint</summary>
<p>

Export env vars:

```bash
eval $(sed 's/^/export /g' secrets.txt)
export VERSION=$(git rev-parse --short=7 HEAD)
export SERVING_SSE_IMAGE=ghcr.io/unionai-oss/modelz-flyte-llama-serving-sse:$VERSION
```

Build the serving image:

```bash
docker build . -f Dockerfile.server_sse \
    --build-arg "HF_TOKEN=$HF_TOKEN" \
    -t $SERVING_SSE_IMAGE
```

Push it:

```bash
docker push $SERVING_SSE_IMAGE
```

Deploy:

```bash
python deploy.py \
    --user-id $MODELZ_USER_ID \
    --api-key $MODELZ_API_KEY \
    --deployment-name flyte-llama-sse-$VERSION \
    --image $SERVING_SSE_IMAGE \
    --server-resource "nvidia-ada-l4-4-48c-192g" \
    --stream
```

Get the `deployment_key` from the output of the command above and use it to test
the model:

```bash
python client_sse.py \
    --prompt "The code snippet below shows a basic Flyte workflow" \
    --n-tokens 250 \
    --output-file output.txt \
    --api-key $MODELZ_API_KEY \
    --deployment-key <deployment_key>
```

</p>
</details>



## 🔖 Model Card

### Dataset

This system will be based on all of the [Flyte](https://flyte.org/) codebases:

- [flyte](https://github.com/flyteorg/flyte): Flyte's main repo
- [flytekit](https://github.com/flyteorg/flytekit): Python SDK
- [flytepropeller](https://github.com/flyteorg/flytepropeller) Kubernetes-native operator for Flyte
- [flyteplugins](https://github.com/flyteorg/flyteplugins): Backend Flyte plugins
- [flyteidl](https://github.com/flyteorg/flyteidl): Flyte language specification in protobuf
- [flyteadmin](https://github.com/flyteorg/flyteadmin): Flyte's control plane
- [flyteconsole](https://github.com/flyteorg/flyteconsole): UI console
- [flytesnacks](https://github.com/flyteorg/flytesnacks): Example repo
- [flyte-conference-talks](https://github.com/flyteorg/flyte-conference-talks): Repo of conference talks

The dataset will consist of source files, tests, and documentation from all of
these repositories.

### Data Source Extensions

This dataset could be enriched with open source repos that use Flyte in their
codebase, which includes open source repos maintained by the Flyte core team
and those maintained by the community. This would further train the model on how
the community uses flytekit or configures their codebase in the wild.

### LLM-augmented supervised finetuning

We can build a supervised finetuning dataset using an LLM to generate a synthetic
instruction given a piece of Flyte code. For example, given a `flytesnacks` example,
an LLM can be prompted to create an instruction associated with that example. Or,
given a flytekit plugin, an LLM can be prompted to create an instruction associated
with creating the flytekit plugin class that implements the plugin interface.

### Training

There are several possible training approaches to take:

- Causal language modeling (CLM)
- Masked language modeling (MLM)
- Fill in the middle (FIM)

We'll start with the simplest case using CLM to get a baseline, then experiment
with FIM since we may want Flyte Llama to be able to both complete code and
suggest code given some suffix and prefix (see [Resources](#resources) section below).

### Data Augmentation

There are many data augmentation techniques we can leverage on top of the training
approaches mentioned above:

- **Add metadata to the context:** This can include adding the repo name,
  file name, file extension to the beginnign of each training example to condition
  the token completion on the context of the code.

### Evaluation

We'll use perplexity as a baseline metric for evaluating the model. This will
capture how well the fine-tuned model fits the data.

It may be useful to keep hold-out data for evaluating the model's ability to
generalize by excluding data from certain repos. For example, we can
pretrain the model on pure Flyte source code and test it on example documentation
repos, so you may have a train-test split as follows:

- Training set: `flyte`, `flytekit`, `flytepropeller`, `flyteplugins`, `flyteidl`, `flyteadmin`, `flyteconsole`
- Test set: `flytesnacks`, `flyte-conference-talks`

Though there may be some data leakage, for the most part the code in the example repos
should be different enough from the code in the core source code repos that the model
will have to figure out how to use the basic building blocks in the source code
to generate the examples (this is somewhat what a human does to generate code examples).

### Resources

- [Code LLama paper](https://ai.meta.com/research/publications/code-llama-open-foundation-models-for-code/)
- [Causal Masked Multimodal Model paper](https://arxiv.org/abs/2201.07520)
- [Fill in the Middle paper](https://arxiv.org/abs/2207.14255)
