# -*- coding: utf-8 -*-
"""Copy of IR Assignment 3.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1P0Uup--G38BRBqfAi_1vVFDwKOLK9Woc
"""

!pip install pulp

import numpy as np
import pandas as pd
from pulp import *
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import re
import string
from tqdm import tqdm

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text):
        """Clean and normalize text"""
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove special characters and digits
        text = re.sub(r'[^\w\s.]', '', text)  # Keep periods for sentence splitting
        text = re.sub(r'\d+', '', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def split_into_sentences(self, text):
        """Split text into sentences and preprocess each"""
        if not isinstance(text, str):
            return [], []

        # Clean the text but keep periods
        cleaned_text = self.clean_text(text)

        # Split into sentences
        original_sentences = sent_tokenize(text)
        cleaned_sentences = sent_tokenize(cleaned_text)

        # Remove empty sentences
        valid_pairs = [(clean, orig) for clean, orig in zip(cleaned_sentences, original_sentences)
                      if clean.strip()]

        if not valid_pairs:
            return [], []

        cleaned_sentences, original_sentences = zip(*valid_pairs)
        return list(cleaned_sentences), list(original_sentences)

# def process_cnn_dailymail_dataset(file_path, max_samples=None):
#     """Process the CNN/DailyMail dataset"""
#     print("Loading dataset...")
#     df = pd.read_csv(file_path)

#     if max_samples:
#         df = df.head(max_samples)
#     df['article'] = df['article'].apply(lambda x: str(x) if isinstance(x, str) else "")
#     df['highlights'] = df['highlights'].apply(lambda x: str(x) if isinstance(x, str) else "")

#     # Filter out rows where summary length > 200 words
#     print("Filtering samples...")
#     df['summary_word_count'] = df['highlights'].apply(lambda x: len(str(x).split()))
#     df_filtered = df[df['summary_word_count'] <= 200].copy()

#     # Remove any rows with missing values
#     df_filtered = df_filtered.dropna(subset=['article', 'highlights'])

#     print(f"Original number of samples: {len(df)}")
#     print(f"Number of samples after filtering: {len(df_filtered)}")

#     return df_filtered

!pip install pulp

class SummarizationILP:
    def __init__(self, K=200):
        self.K = K  # Maximum words in summary

    def compute_similarity(self, sent1, sent2):
        """Compute cosine similarity between two sentences"""
        # Convert sentences to word sets
        words1 = set(sent1.lower().split())
        words2 = set(sent2.lower().split())

        # Compute cosine similarity
        intersection = words1.intersection(words2)
        if not words1 or not words2:
            return 0.0
        sim = len(intersection) / (np.sqrt(len(words1)) * np.sqrt(len(words2)))
        return sim

    def compute_position_score(self, pos, doc_length):
        """Compute position score based on sentence position"""
        return 1.0 - (pos / max(doc_length, 1))




    def solve_ilp(self, sentences, docs):
        """Solve the ILP formulation"""
        # Create the model
        model = LpProblem("Summarization", LpMaximize)

        # Decision variables
        n = len(sentences)
        x = LpVariable.dicts("sent", range(n), 0, 1, LpBinary)  # Sentence selection
        xij = LpVariable.dicts("redundancy", [(i, j) for i in range(n) for j in range(i+1, n)], 0, 1, LpBinary)

        # Compute relevance scores
        sentence_lengths = [len(doc) for doc in docs]

        doc_indices = [i for i, doc in enumerate(docs) for _ in doc]  # Map each sentence to its document index
        rel_scores = []
        for i, sent in enumerate(sentences):
            doc_idx = doc_indices[i]
            if sentence_lengths[doc_idx] > 0:
              pos_score = self.compute_position_score(i % sentence_lengths[doc_idx], sentence_lengths[doc_idx])
            else:
              pos_score = 0
            # pos_score = self.compute_position_score(i % sentence_lengths[doc_idx], sentence_lengths[doc_idx])
            sim_score = np.mean([self.compute_similarity(sent, other)
                                for j, other in enumerate(sentences) if i != j])
            rel_scores.append(pos_score + sim_score)


            # for i, sent in enumerate(sentences):
            #         pos_score = self.compute_position_score(i % len(docs[i]), len(docs[i]))
            #         sim_score = np.mean([self.compute_similarity(sent, other)
            #                             for j, other in enumerate(sentences) if i != j])
            #         rel_scores.append(pos_score + sim_score)

        # Compute redundancy scores
        red_scores = {}
        for i in range(n):
            for j in range(i+1, n):
                red_scores[(i, j)] = self.compute_similarity(sentences[i], sentences[j])

        # Objective function: maximize relevance - redundancy
        objective = lpSum([rel_scores[i] * x[i] for i in range(n)]) - \
                    lpSum([red_scores[(i, j)] * xij[(i, j)] for i in range(n) for j in range(i+1, n)])
        model += objective

        # Constraint 1: Total length constraint (∑ α_i <= K)
        sent_lengths = [len(sent.split()) for sent in sentences]
        model += lpSum([sent_lengths[i] * x[i] for i in range(n)]) <= self.K

        # Constraint 2 and 4: Pairwise redundancy constraints (α_ij - α_j <= 0 and α_i + α_j - α_ij <= 1)
        for i in range(n):
            for j in range(i+1, n):
                model += xij[(i, j)] - x[j] <= 0
                model += x[i] + x[j] - xij[(i, j)] <= 1

        # Constraint 3: α_ij - α_i ≤ 0
        for i in range(n):
            for j in range(i+1, n):
                model += xij[(i, j)] - x[i] <= 0

        # Solve
        model.solve()

        # Get selected sentences
        selected = [i for i in range(n) if x[i].value() > 0.5]

        return selected

    def summarize(self, text_data):
        # Check if text_data is a list of lists or a single list
        if isinstance(text_data[0], list):
            # Multi-document: list of lists of sentences
            all_sentences = [sent for doc in text_data for sent in doc]
            docs = text_data
        else:
            # Single document: list of sentences
            all_sentences = text_data
            docs = [text_data]  # Wrap in a list for consistent handling

        # Solve ILP for multi-document summarization
        selected_indices = self.solve_ilp(all_sentences, docs)

        # Return selected sentences in their original order
        return selected_indices

df = pd.read_csv('/content/dataset-1k.csv')

df['summary_word_count'] = df['highlights'].apply(lambda x: len(str(x).split()))
df_filtered = df[df['summary_word_count'] <= 200].copy()
df_filtered

df_filtered.shape

df['summary_word_count'].describe()

non_string_article_count = df_filtered['article'].apply(lambda x: not isinstance(x, str)).sum()
non_string_highlights_count = df_filtered['highlights'].apply(lambda x: not isinstance(x, str)).sum()

print(f"Number of non-string entries in 'article': {non_string_article_count}")
print(f"Number of non-string entries in 'highlights': {non_string_highlights_count}")

import pandas as pd

empty_article_count = df_filtered['article'].isnull().sum()
empty_highlights_count = df_filtered['highlights'].isnull().sum()

print(f"Number of empty articles: {empty_article_count}")
print(f"Number of empty highlights: {empty_highlights_count}")

empty_string_article_count = df_filtered['article'].apply(lambda x: isinstance(x, str) and not x.strip()).sum()
empty_string_highlights_count = df_filtered['highlights'].apply(lambda x: isinstance(x, str) and not x.strip()).sum()

print(f"Number of articles with empty strings: {empty_string_article_count}")
print(f"Number of highlights with empty strings: {empty_string_highlights_count}")

preprocessor = TextPreprocessor()

# List to store results
results = []

# Process each article
for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="Processing articles"):
    try:
        # Preprocess article
        processed_sentences, original_sentences = preprocessor.split_into_sentences(row['article'])

        # Ensure processed_sentences is a list of lists
        processed_sentences_list = [processed_sentences]

        # Add the result to the results list
        results.append(processed_sentences_list)

        print(processed_sentences)

    except Exception as e:
        print(f"Error processing article {row['id']}: {str(e)}")

results

# summarizer = SummarizationILP(K=200)

# # List to store final results
# final_results = []

# # Iterate over the list of lists (results)
# for idx, article_sentences in tqdm(enumerate(results), desc="Generating summaries"):
#     try:
#         # For each article's processed sentences (list of sentences)
#         for sentences in article_sentences:
#             # Summarize the list of sentences
#             summary_indices = summarizer.summarize([sentences])  # Summarize each list
#             generated_summary = ' '.join([sentences[i] for i in summary_indices])
#             print(generated_summary)

#             # Assuming you want to map the `id` and original sentences
#             original_sentences = sentences  # This is the list of original sentences
#             article_id = idx  # You can change this depending on how the `id` is defined

#             # Store results with the article ID and the original sentences
#             final_results.append({
#                 'id': article_id,  # Store the ID of the article
#                 'original_sentences': ' '.join(original_sentences),  # Join sentences to form original text
#                 'generated_summary': generated_summary
#             })
#     except Exception as e:
#         print(f"Error processing article: {str(e)}")

# # Create a DataFrame from the final results
# final_results_df = pd.DataFrame(final_results)

import pandas as pd
from tqdm import tqdm

summarizer = SummarizationILP(K=200)

# List to store final results
final_results = []

# Counter for batch saving
batch_size = 50

# Iterate over the list of lists (results)
for idx, article_sentences in tqdm(enumerate(results), desc="Generating summaries"):
    try:
        # For each article's processed sentences (list of sentences)
        for sentences in article_sentences:
            # Summarize the list of sentences
            summary_indices = summarizer.summarize([sentences])  # Summarize each list
            generated_summary = ' '.join([sentences[i] for i in summary_indices])
            print(generated_summary)

            original_sentences = sentences
            article_id = idx

            final_results.append({
                'id': article_id,
                'original_sentences': ' '.join(original_sentences),  # Join sentences to form original text
                'generated_summary': generated_summary
            })

        # Check if we have reached the batch size to save the current results
        if (idx + 1) % batch_size == 0:
            batch_df = pd.DataFrame(final_results)
            batch_df.to_csv("Assignment3_21EC39023_summary.csv", mode='a', index=False, header=not (idx + 1 > batch_size))

            # Clear the batch results after saving
            final_results = []

    except Exception as e:
        print(f"Error processing article: {str(e)}")

if final_results:
    final_results_df = pd.DataFrame(final_results)
    final_results_df.to_csv("Assignment3_21EC39023_summary.csv", mode='a', index=False, header=False)