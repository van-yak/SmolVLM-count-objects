import argparse

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers import TrainingArguments, Trainer
from transformers.image_utils import load_image

from datasets import load_dataset

def main(args):
    data_files = {"train": "train-00001-of-00027.parquet"}
    ds = load_dataset("parquet", data_dir=args.dataset_path, data_files=data_files)

    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    for param in model.model.vision_model.parameters():
        param.requires_grad = False
    image_token_id = processor.tokenizer.additional_special_tokens_ids[
        processor.tokenizer.additional_special_tokens.index("<image>")
    ]


    def collate_fn(examples):
        texts = []
        images = []
        for example in examples:
            image = load_image(example["image"])
            if image.mode != 'RGB':
                image = image.convert('RGB')
            question = "How many objects are in the image?, Answer in one digit"
            answer = example["solution"].replace("<answer>", "").replace("</answer>", "").strip()
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": answer}
                    ]
                }
            ]
            text = processor.apply_chat_template(messages, add_generation_prompt=False)
            texts.append(text.strip())
            images.append([image])

        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        labels[labels == image_token_id] = -100
        batch["labels"] = labels

        return batch

    model_name = args.model_name_or_path.split("/")[-1]

    training_args = TrainingArguments(
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        warmup_steps=50,
        learning_rate=1e-4,
        weight_decay=0.01,
        logging_steps=25,
        save_strategy="steps",
        save_steps=250,
        save_total_limit=1,
        optim="adamw_torch",
        bf16=True,
        output_dir=f"{args.output_path}/{model_name}_count_objects",
        remove_unused_columns=False,
        report_to="tensorboard",
        dataloader_pin_memory=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=ds["train"],
    )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="HuggingFaceTB/SmolVLM-256M-Instruct")
    parser.add_argument("--dataset_path", default="~/datasets/clevr_cogen_a_train")
    parser.add_argument("--output_path", default=".")

    args = parser.parse_args()

    main(args)