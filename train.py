import argparse
import gc
import json
import random
import re

import numpy as np
import torch
from datasets import Dataset, load_dataset
from pydantic import BaseModel
from tokenizers import Tokenizer, normalizers, pre_tokenizers
from tokenizers.models import BPE, Unigram, WordPiece
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordPieceTrainer
from transformers import (
    BertConfig,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)


class Config(BaseModel):
    seed: int
    train_size: int
    eval_size: int
    use_only_english: bool
    batch_size: int
    tokenizer_type: str
    vocab_sizes: list
    bert_model_config: dict
    training_args: dict
    block_size: int
    mlm_probability: float


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_english(example):
    try:
        text = example["title"] + " " + example["body"]
        return bool(re.match(r"^[a-zA-Z0-9\s\.,!?\'\"-]*$", text))
    except:
        return False


def train_tokenizer(tokenizer_type, data, vocab_size):
    if tokenizer_type == "bpe":
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        )
    elif tokenizer_type == "wordpiece":
        tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
        trainer = WordPieceTrainer(
            vocab_size=vocab_size,
            special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        )
    elif tokenizer_type == "unigram":
        tokenizer = Tokenizer(Unigram())
        trainer = UnigramTrainer(
            vocab_size=vocab_size,
            unk_token="[UNK]",
            special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        )
    else:
        raise ValueError("Unsupported tokenizer type")

    tokenizer.normalizer = normalizers.Sequence(
        [normalizers.NFKC(), normalizers.Lowercase(), normalizers.Strip()]
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    data_iterator = (t + " " + b for t, b in zip(data["title"], data["body"]))
    tokenizer.train_from_iterator(data_iterator, trainer=trainer)

    tokenizer_model = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    tokenizer_model.pad_token = "[PAD]"
    tokenizer_model.mask_token = "[MASK]"
    tokenizer_model.unk_token = "[UNK]"
    tokenizer_model.sep_token = "[SEP]"
    tokenizer_model.cls_token = "[CLS]"

    return tokenizer_model


def prepare_dataset(tokenizer, data, block_size):
    def merge_text(example):
        return {"text": example["title"] + " " + example["body"]}

    dataset = data.map(merge_text, remove_columns=["title", "body"])

    texts = dataset["text"]

    tokenized_inputs = tokenizer(texts, truncation=False, padding=False)

    input_ids = []
    for ids in tokenized_inputs["input_ids"]:
        input_ids.extend(ids)

    total_length = len(input_ids)

    total_length = (total_length // block_size) * block_size
    input_ids = input_ids[:total_length]

    input_ids = np.array(input_ids)
    input_ids = input_ids.reshape(-1, block_size)
    print(f"Created {input_ids.shape[0]} blocks of size {block_size}")

    labels = np.copy(input_ids)

    attention_mask = np.ones_like(input_ids)

    lm_dataset = Dataset.from_dict(
        {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    )

    lm_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

    return lm_dataset


def get_model(config, vocab_size):
    bert_model_config = BertConfig(vocab_size=vocab_size, **config.bert_model_config)
    model = BertForMaskedLM(config=bert_model_config)
    return model


def get_compute_metrics():
    num = 0
    denum = 0

    def compute_metrics(eval_preds, compute_result):
        nonlocal num, denum

        if compute_result:
            return {"mlm_accuracy": num / denum if denum > 0 else 0.0}

        logits, labels = eval_preds
        if isinstance(logits, tuple):
            logits = logits[0]
        predictions = torch.argmax(logits, axis=-1)

        mask = labels != -100
        labels = labels[mask]
        predictions = predictions[mask]
        accuracy = predictions == labels

        num += torch.sum(accuracy).item()
        denum += accuracy.numel()

    return compute_metrics


def main():
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--tokenizer_type", type=str)
    parser.add_argument("--use_only_english", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config_json = json.load(f)
    config = Config(**config_json)
    1/0
    if args.tokenizer_type is not None:
        config.tokenizer_type = args.tokenizer_type

    if args.use_only_english is not None:
        config.use_only_english = True

    set_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    data_files = {"train": "reddit_title_text_2021.jsonl.gz"}
    dataset = load_dataset(
        "sentence-transformers/reddit-title-body", data_files=data_files
    )["train"]

    if config.use_only_english:
        dataset = dataset.filter(is_english, num_proc=32)

    data = dataset.shuffle(seed=config.seed)
    train_data = data.select(range(config.train_size))
    test_data = data.select(
        range(config.train_size, config.train_size + config.eval_size)
    )

    results = []

    for vocab_size in config.vocab_sizes:
        print(
            f"Training with tokenizer type '{config.tokenizer_type}' and vocab size {vocab_size}"
        )

        tokenizer = train_tokenizer(config.tokenizer_type, train_data, vocab_size)
        print("Tokenizer trained")

        train_dataset = prepare_dataset(tokenizer, train_data, config.block_size)
        test_dataset = prepare_dataset(tokenizer, test_data, config.block_size)
        print("Datasets prepared")

        model = get_model(config, len(tokenizer))
        print("Model created")

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=config.mlm_probability
        )

        output_dir = f'output/{config.tokenizer_type}_{vocab_size}{"_eng" if config.use_only_english else ""}'

        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            **config.training_args,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=get_compute_metrics(),
        )

        tokenizer.save_pretrained(output_dir)
        trainer.train()
        eval_result = trainer.evaluate()

        results.append(
            {
                "tokenizer_type": config.tokenizer_type,
                "vocab_size": vocab_size,
                "mlm_accuracy": eval_result["eval_mlm_accuracy"],
            }
        )

        del model
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

    for res in results:
        print(
            f"Tokenizer Type: {res['tokenizer_type']}, "
            f"Vocab Size: {res['vocab_size']}, "
            f"MLM Accuracy: {res['mlm_accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
