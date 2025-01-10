import evaluate
import nltk
import numpy as np
import transformers
from datasets import load_dataset, load_metric
from huggingface_hub import notebook_login
from nltk.tokenize import sent_tokenize
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, T5Tokenizer, T5ForConditionalGeneration

import torch

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS backend for Apple M1 GPU")
else:
    print("MPS is not available, using CPU")
    device = torch.device("cpu")

# loading the dataset
news_dataset = load_dataset("multi_news")


def show_data_samples(dataset, samples=2, suffle=20):
    data_sample = dataset["train"].shuffle(seed=suffle).select(range(samples))
    for sample in data_sample:
        print(f"Document: {sample['document']}\n")
        print(f"Summary: {sample['summary']}\n")

# show_data_samples(news_dataset, samples=1)


model_checkpoint = "google/mt5-small"
tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)

model.to(device)

max_input_length = 1000
max_target_length = 100


def preprocess_function(dataset):
    model_inputs = tokenizer(
        dataset['document'], max_length=max_input_length, truncation=True)
    labels = tokenizer(dataset['summary'],
                       max_length=max_target_length, truncation=True)
    model_inputs['labels'] = labels['input_ids']
    return model_inputs


tokenized_datasets = news_dataset.map(preprocess_function, batched=True)

rouge_score = load_metric("rouge")
nltk.download("punkt")

batch_size = 2
num_train_epochs = 10
logging_steps = len(tokenized_datasets['train']) // batch_size
model_name = model_checkpoint.split("/")[-1]

args = Seq2SeqTrainingArguments(
    output_dir=f"{model_name}_summarization",
    evaluation_strategy="epoch",
    learning_rate=5e-6,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=num_train_epochs,
    predict_with_generate=True,
    logging_steps=logging_steps,
    push_to_hub=True,
)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(
        predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = ["\n".join(sent_tokenize(pred.strip()))
                     for pred in decoded_preds]
    decoded_labels = ["\n".join(sent_tokenize(label.strip()))
                      for label in decoded_labels]
    result = rouge_score.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    breakpoint()
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    return {k: round(v, 4) for k, v in result.items()}


data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
tokenizer_dataset = tokenized_datasets.remove_columns(
    news_dataset["train"].column_names
)

# Set the device for PyTorch to use
if torch.cuda.is_available():
    torch.cuda.set_device(device.index)

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.evaluate()
trainer.push_to_hub(
    commit_message="Paper Summarization Complete", tags="summarization")
