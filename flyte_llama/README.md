# Flyte Llama

Flyte Llama is a fine-tuned model based on [Code Llama](https://about.fb.com/news/2023/08/code-llama-ai-for-coding/).

## Dataset

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

## Training

There are several possible training approaches to take:

- Causal language modeling (CLM)
- Masked language modeling (MLM)
- Fill in the middle (FIM)

We'll start with the simplest case using CLM to get a baseline, then experiment
with FIM since we may want Flyte Llama to be able to both complete code and
suggest code given some suffix and prefix.

## Evaluation

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

## Resources

- [Code LLama paper](https://ai.meta.com/research/publications/code-llama-open-foundation-models-for-code/)
- [Causal Masked Multimodal Model paper](https://arxiv.org/abs/2201.07520)
- [Fill in the Middle paper](https://arxiv.org/abs/2207.14255)
