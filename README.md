# CSL 7640: Natural Language Understanding - Programming Assignment 2

## Overview

This assignment consists of two problems:
- **Problem 1:** Semantic Analysis using Word2Vec on an IIT Jodhpur corpus
- **Problem 2:** Character-level Indian Name Generation using RNN variants

## Project Structure

```
PA2/
├── Problem1/
│   ├── scraper.py              # Web scraping from 10 IITJ pages + PDF extraction
│   ├── preprocess.py           # Corpus cleaning, tokenization, statistics, word cloud
│   ├── train_word2vec.py       # Word2Vec training (CBOW & Skip-gram) with hyperparameter experiments
│   ├── semantic_analysis.py    # Nearest neighbors, analogy experiments
│   ├── visualize.py            # PCA, t-SNE visualizations
│   ├── run_problem1.py         # Run entire Problem 1 pipeline
│   ├── data/                   # Raw and cleaned corpus files
│   ├── models/                 # Saved Word2Vec models
│   └── outputs/                # Statistics, plots, reports
├── Problem2/
│   ├── rnn_models.py           # Vanilla RNN, BLSTM, Attention RNN implementations + training + evaluation
│   ├── generate_names.py       # Training names generator
│   ├── TrainingNames.txt       # Training dataset of Indian names
│   ├── models/                 # Saved PyTorch model weights
│   └── outputs/                # Generated names, training loss plot, report
├── Academic_Regulations_Final_03_09_2019.pdf  # Mandatory PDF source for corpus
└── README.md
```

## Prerequisites

```bash
pip install requests beautifulsoup4 PyPDF2 nltk gensim numpy matplotlib wordcloud torch scikit-learn
```

## How to Run

### Problem 1: Semantic Analysis with Word2Vec

Run the full pipeline (scraping → preprocessing → training → analysis → visualization):

```bash
cd Problem1
python run_problem1.py
```

Or run individual steps:

```bash
python scraper.py            # Step 1: Scrape IITJ web pages + PDF
python preprocess.py         # Step 2: Clean and tokenize corpus
python train_word2vec.py     # Step 3: Train Word2Vec models (CBOW & Skip-gram)
python semantic_analysis.py  # Step 4: Nearest neighbors + analogy experiments
python visualize.py          # Step 5: PCA & t-SNE visualizations
```

### Problem 2: Character-Level Name Generation

Run all three models (Vanilla RNN, BLSTM, Attention RNN):

```bash
cd Problem2
python rnn_models.py
```

This will train all three models, generate 200 names per model, evaluate them, and save the report and loss plot.

## Outputs

### Problem 1
- `outputs/dataset_statistics.txt` — Corpus statistics (tokens, vocabulary, top-20 words)
- `outputs/training_results.txt` — Hyperparameter experiment results
- `outputs/semantic_analysis_report.txt` — Nearest neighbors and analogy results
- `outputs/wordcloud.png` — Word cloud visualization
- `outputs/cbow_tsne.png`, `skipgram_tsne.png` — t-SNE plots
- `outputs/cbow_pca.png`, `skipgram_pca.png` — PCA plots

### Problem 2
- `outputs/rnn_report.txt` — Full comparison report (architectures, parameters, metrics, samples)
- `outputs/training_loss.png` — Training loss curves for all three models
- `outputs/generated_vanilla_rnn.txt` — Generated names from Vanilla RNN
- `outputs/generated_blstm.txt` — Generated names from BLSTM
- `outputs/generated_attention_rnn.txt` — Generated names from Attention RNN

## Author

Student, IIT Jodhpur
CSL 7640: Natural Language Understanding, March 2026
