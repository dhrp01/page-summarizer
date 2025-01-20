# Summarization with FLAN-T5

This project demonstrates a workflow to fine-tune the FLAN-T5 model for dialogue summarization using the `DialogSum` dataset. It includes data preprocessing, tokenization, training, and evaluation of the model.

## Installation
1. Create a virtual environment
```
python3 -m venv .venv
source .venv/bin/activate
```
2. Run the below command
```
pip install -r requirements.txt
```

## Model Features:
1. *Dataset*: Uses the [DialogSum](https://huggingface.co/datasets/knkarthick/dialogsum) dataset for summarization tasks.
2. *Model*: Fine-tunes the `FLAN-T5` model ([google/flan-t5-base](https://huggingface.co/google/flan-t5-base)) for generating conversation summaries.
3. *GPU Support*: Automatically detects and utilizes CUDA or MPS if available for faster computation.
4. *Tokenization*: Prepares data for training using the transformers library.
5. *Training*: Implements a training pipeline with Trainer for fine-tuning the model.
6. *Evaluation*: Compares summaries generated by the baseline, the fine-tuned model, and human-written summaries.

Trained model is uploaded at [dhrumeen/small_summarization_model](https://huggingface.co/dhrumeen/small_summarization_model)

## Customization
1. *Training Configuration*: Modify `TrainingArguments` to customize learning rate, number of epochs, and other parameters.
2. *Dataset*: Replace the `DialogSum` dataset with your dataset for different summarization tasks.

