"""
=============================================================================
CSL 7640: Natural Language Understanding - Assignment 2
Problem 1: Complete Pipeline Runner
=============================================================================
This script runs the entire Problem 1 pipeline in sequence:
  1. Scraping (Task 1a)
  2. Preprocessing (Task 1b)
  3. Word2Vec Training (Task 2)
  4. Semantic Analysis (Task 3)
  5. Visualization (Task 4)

Usage: python run_problem1.py
=============================================================================
"""

import os
import sys

# Ensure we run from the Problem1 directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("CSL 7640: NLU Assignment 2 - Problem 1")
print("Learning Word Embeddings from IIT Jodhpur Data")
print("=" * 70)

# ---- Step 1: Scrape Data ----
print("\n\n>>> STEP 1: WEB SCRAPING & PDF EXTRACTION <<<\n")
from scraper import scrape_all
documents = scrape_all()

# ---- Step 2: Preprocess ----
print("\n\n>>> STEP 2: PREPROCESSING & STATISTICS <<<\n")
from preprocess import preprocess_corpus
sentences = preprocess_corpus()

# ---- Step 3: Train Word2Vec Models ----
print("\n\n>>> STEP 3: WORD2VEC MODEL TRAINING <<<\n")
from train_word2vec import load_corpus, run_hyperparameter_experiments
corpus = load_corpus()
results, cbow_model, sg_model = run_hyperparameter_experiments(corpus)

# ---- Step 4: Semantic Analysis ----
print("\n\n>>> STEP 4: SEMANTIC ANALYSIS <<<\n")
from semantic_analysis import semantic_analysis
semantic_analysis()

# ---- Step 5: Visualization ----
print("\n\n>>> STEP 5: VISUALIZATION <<<\n")
from visualize import load_models, visualize_embeddings, create_comparison_plot, save_interpretation
cbow, sg = load_models()
for model, name in [(cbow, "CBOW"), (sg, "Skip-gram")]:
    visualize_embeddings(model, name, method="pca")
    visualize_embeddings(model, name, method="tsne")
create_comparison_plot()
save_interpretation()

print("\n" + "=" * 70)
print("PROBLEM 1 COMPLETE! All outputs saved in:")
print("  - data/           : Raw and cleaned corpus")
print("  - models/         : Trained Word2Vec models")
print("  - outputs/        : Statistics, visualizations, reports")
print("=" * 70)
