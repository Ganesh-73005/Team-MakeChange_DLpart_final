# Blog Processor

## Overview

The AI Blog Processor is an advanced natural language processing tool that transforms unstructured blog content into well-organized, easily digestible formats. It leverages state-of-the-art machine learning models to automatically generate headings, split text into meaningful paragraphs, and create relevant questions for each section.

## Features

1. **Paragraph Splitting**: Intelligently divides long text into coherent paragraphs based on content similarity.
2. **Heading Generation**: Automatically creates relevant headings for each paragraph using a fine-tuned LED (Longformer Encoder-Decoder) model.
3. **Question Generation**: Produces thought-provoking questions for each section using a T5-based model.

## Dependencies

- TensorFlow
- PyTorch
- Transformers (Hugging Face)
- NLTK
- scikit-learn
- pandas
- datasets

## Setup

1. Mount Google Drive (if using Google Colab):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. Install required packages:
   ```
   !pip install transformers datasets
   ```

3. Download NLTK data:
   ```python
   import nltk
   nltk.download('punkt')
   ```

## Usage

1. Initialize the `BlogProcessor` class:
   ```python
   processor = BlogProcessor()
   ```

2. Process a blog post:
   ```python
   processed_blog = processor.process_blog(your_blog_text)
   print(processed_blog)
   ```

## How It Works

1. **Text Splitting**: The `split_into_meaningful_paragraphs` function uses **TF-IDF** and cosine similarity to group sentences into coherent paragraphs.

2. **Heading Generation**: A pre-trained **LED** model generates appropriate headings for each paragraph.

3. **Question Generation**: A **T5-based** model creates relevant questions for each section of the blog.

4. **Final Assembly**: The processed blog is assembled with generated headings, original paragraphs, and review questions.

## Models

- Heading Generation: Custom-trained LED model (path: "/content/drive/MyDrive/checkpoint-100")
- Question Generation: "valhalla/t5-base-qg-hl"

## Note

Ensure you have the necessary model checkpoints in your Google Drive or adjust the paths accordingly.

