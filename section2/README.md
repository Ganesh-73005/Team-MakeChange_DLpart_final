# Fine-tuning with LED (Longformer Encoder-Decoder)

This section covers the process of fine-tuning the LED model for blog content generation using the dataset created in Section 1.

## Quickstart Guide

1. Upload the `articlesSet.csv` file to a folder named "Code Cycle" in your Google Drive.
2. Open a new Google Colab notebook and change the runtime type to T4 GPU.
3. Mount your Google Drive in Colab:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
4. Install required libraries:
   ```
   !pip install -U datasets
   !pip install transformers==4.19.2
   !pip install rouge_score
   ```
5. Copy and paste the following code into a Colab cell:
   ```python
   import pandas as pd
   from datasets import load_metric, Dataset
   from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
   import torch
   import numpy as np

   #load and preprocess data
   df = pd.read_csv("/content/drive/MyDrive/Code Cycle/articlesSet.csv")
   df = df.dropna()
   df['length'] = df.content.map(lambda x: len(x.split(" ")))
   tempDf = df[(df.length <= 800) & (df.length >= 100)]

   #tokenize 
   tokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384")

   def process_data_to_model_inputs(batch):
       # (Copy the entire process_data_to_model_inputs function here)

   #split data and create datasets
   train, validate, test = np.split(tempDf.sample(frac=1, random_state=42), [int(.4*len(df)), int(.5*len(df))])
   train = train[:250]
   validate = validate[25:50]

   train_dataset = Dataset.from_pandas(train).map(process_data_to_model_inputs, batched=True, batch_size=4, remove_columns=["content", "summary", "length", "__index_level_0__"])
   val_dataset = Dataset.from_pandas(validate).map(process_data_to_model_inputs, batched=True, batch_size=4, remove_columns=["content", "summary", "length", "__index_level_0__"])

   #set up the model and trainer
   led = AutoModelForSeq2SeqLM.from_pretrained("allenai/led-base-16384", gradient_checkpointing=True, use_cache=False)
   rouge = load_metric("rouge")

   def compute_metrics(pred):
       #code

   training_args = Seq2SeqTrainingArguments(
       predict_with_generate=True,
       evaluation_strategy="steps",
       per_device_train_batch_size=4,
       per_device_eval_batch_size=4,
       output_dir="./",
       logging_steps=5,
       eval_steps=10,
       save_steps=10,
       save_total_limit=2,
       gradient_accumulation_steps=4,
       num_train_epochs=2
   )

   trainer = Seq2SeqTrainer(
       model=led,
       tokenizer=tokenizer,
       args=training_args,
       compute_metrics=compute_metrics,
       train_dataset=train_dataset,
       eval_dataset=val_dataset,
   )

   # Start training
   trainer.train()

   print("Training complete. Model checkpoints saved in the output directory.")
   ```
6. Run the cell and wait for the training to complete.

## Prerequisites

- Google Colab account with T4 GPU runtime
- Google Drive with the `articlesSet.csv` file from Section 1
- Required libraries: transformers, datasets, rouge_score, torch, numpy, pandas

## Detailed Steps

1. Data Preparation:
   - Load the dataset from `articlesSet.csv`
   - Remove any rows with NaN values
   - Filter out articles with less than 100 or more than 800 words

2. Tokenization:
   - Use the LED tokenizer to process input and output texts
   - Prepare the data for the model by tokenizing summaries and content

3. Dataset Creation:
   - Split the data into training and validation sets
   - Create PyTorch datasets for both sets

4. Model Setup:
   - Initialize the LED model
   - Set up training arguments and the trainer

5. Training:
   - Fine-tune the model on the prepared datasets
   - Use ROUGE metric for performance evaluation

6. Inference:
   - Load the fine-tuned model
   - Generate content based on input summaries

## Code Overview

The code is structured into several main parts:
1. Data loading and preprocessing
2. Tokenization and dataset preparation
3. Model and trainer setup
4. Training process
5. Inference and content generation

After completing this section, you'll have a fine-tuned LED model capable of generating blog content from summaries.
