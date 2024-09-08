# Section 1: Dataset Creation

This section focuses on creating a dataset of blog articles and their summaries using extractive summarization techniques.

## Quickstart Guide

1. Open Google Colab and create a new notebook
2. Mount your Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Install required libraries:
   ```python
   !pip install nltk gensim
   ```
4. Copy and paste the following code into a Colab cell:
   ```python
   import numpy as np
   import pandas as pd
   import nltk
   import re
   from nltk.tokenize import sent_tokenize
   from nltk.corpus import stopwords
   from gensim.models import Word2Vec
   from scipy import spatial
   import networkx as nx
   import csv

   #download NLTK data
   nltk.download('punkt')
   nltk.download('stopwords')

   #load the dataset
   df = pd.read_csv('/content/drive/MyDrive/medium_articles.csv')
   df = df[['title', 'text']].drop_duplicates()

   #define the summarization function
   def generateSummary(blog):
       # code here
       # ...

   #generate summaries and save to CSV
   filename = "articlesSet.csv"
   fields = ['title', 'summary', 'content']

   with open(filename, 'w', newline='') as csvfile:
       csvwriter = csv.writer(csvfile)
       csvwriter.writerow(fields)
       
       def callback(row):
           summary = generateSummary(row['text'])
           if isinstance(summary, str):
               csvwriter.writerow([row['title'], summary, row['text']])

       df.apply(callback, axis=1)

   print("Dataset creation complete. Check 'articlesSet.csv' in your Google Drive.")
   ```
5. Run the cell and wait for the process to complete

## Prerequisites

- Python 3+
- Google Colab (with access to Google Drive)
- Required libraries: numpy, pandas, nltk, gensim, scipy, networkx

## Detailed Steps

1. Mount your Google Drive in Colab
2. Load the Medium Articles dataset from your Google Drive
3. Preprocess the dataset by removing duplicates
4. Implement extractive summarization using NLTK, Word2Vec, and PageRank algorithm
5. Generate summaries for the articles
6. Save the results in a new CSV file named "articlesSet.csv"

## Files

- [`creating_dataset.ipynb`](./dataset_creation/creating_dataset.ipynb): Jupyter notebook containing the code for dataset creation and summarization

## Code Overview

1. Data Loading and Preprocessing:
   - Load the 'medium_articles.csv' file
   - Select only 'title' and 'text' columns
   - Remove duplicate entries

2. Summary Generation:
   - Tokenize sentences
   - Clean and preprocess text
   - Remove stopwords
   - Generate word embeddings using Word2Vec
   - Calculate sentence embeddings
   - Compute similarity matrix
   - Apply PageRank algorithm
   - Select top 25% sentences for summary

3. CSV Generation:
   - Create 'articlesSet.csv' with fields: title, summary, content
   - Apply the summarization function to each article
   - Write results to the CSV file

After completing this section, we'll have a dataset ready for fine-tuning the language model in Section 2.
