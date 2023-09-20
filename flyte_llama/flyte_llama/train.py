"""Train Flyte Llama."""

import math
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

import torch
from dataclasses_json import dataclass_json

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from flyte_llama.dataloader import get_dataset


os.environ["WANDB_PROJECT"] = "unionai-flyte-llama"
os.environ["TOKENIZERS_PARALLELISM"] = "true"


@dataclass_json
@dataclass
class TrainerConfig:
    model_path: str = "codellama/CodeLlama-7b-hf"
    data_dir: str = "./data"
    output_dir: str = "./output"
    checkpoint_dir: Optional[str] = None
    num_epochs: int = 20
    batch_size: int = 8
    test_size: float = 0.01
    model_max_length: int = 1024
    seed: int = 41
    report_to: str = "none"
    device_map: Optional[str] = "auto"
    gradient_accumulation_steps: int = 8
    padding: str = "right"
    dataloader_num_proc: int = 8
    use_fp16: bool = False
    use_4bit: bool = False
    use_qlora: bool = False
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj"])
    lora_dropout: float = 0.05,
    debug: bool = False


def train(config: TrainerConfig, hf_auth_token: Optional[str] = None, **kwargs):
    print("Training model...")

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        model_max_length=config.model_max_length,
        padding_side=config.padding,
    )
    tokenizer.pad_token = tokenizer.eos_token

    # load pre-trained model
    load_model_params = {
        **kwargs,
        "use_auth_token": hf_auth_token,
        "torch_dtype": torch.float16,
        "device_map": config.device_map,
    }
    if config.use_4bit:
        load_model_params = {
            **load_model_params,
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                llm_int8_threshold=6.0,
                llm_int8_skip_modules=None,
                llm_int8_enable_fp32_cpu_offload=True,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
            "load_in_4bit": True,
        }
        

    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        **load_model_params,
    )

    optim = "adamw_torch"
    if config.use_qlora:
        optim = "paged_adamw_8bit"
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(
            model,
                LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                target_modules=config.lora_target_modules,
                lora_dropout=config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
        model.print_trainable_parameters()


    def tokenize(examples):
        return tokenizer(examples['text'])

    limit = 5 if config.debug else None
    dataset = (
        get_dataset(
            Path(config.data_dir).expanduser(),
            num_proc=config.dataloader_num_proc,
            limit=limit,
            block_size=config.model_max_length,
            skip_by=config.model_max_length,
        )
        .map(tokenize, batched=True, num_proc=config.dataloader_num_proc)
    )

    print(f"Dataset size: {len(dataset)}")
    dataset_splits = dataset.train_test_split(
        test_size=config.test_size, seed=config.seed
    )
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        evaluation_strategy="steps",
        eval_steps=100,
        learning_rate=3e-4,
        weight_decay=0.1,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        fp16=config.use_fp16,
        half_precision_backend="auto",
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        dataloader_num_workers=0,
        num_train_epochs=config.num_epochs,
        logging_steps=1,
        optim=optim,
        report_to=config.report_to,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_splits["train"],
        eval_dataset=dataset_splits["test"],
        data_collator=data_collator,
    )
    trainer.train(resume_from_checkpoint=config.checkpoint_dir)
    eval_results = trainer.evaluate(eval_dataset=dataset_splits["test"])
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    from transformers import HfArgumentParser

    parser = HfArgumentParser(TrainerConfig)
    args = parser.parse_args_into_dataclasses()[0]

    print(f"Arguments: {args}")
    train(args)
