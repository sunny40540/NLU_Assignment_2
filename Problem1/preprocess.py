import os
import re
import nltk
from collections import Counter
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Resolve all paths relative to this script's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Download NLTK data (punkt tokenizer and stopwords)
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("punkt_tab", quiet=True)

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords


def remove_boilerplate(text):
    
  #  Remove boilerplate text and formatting artifacts from raw scraped text.
    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)
    # Remove email patterns like name[at]domain[dot]com
    text = re.sub(r"\S+\[at\]\S+\[dot\]\S+", "", text)
    text = re.sub(r"\S+@\S+\.\S+", "", text)
    # Remove copyright lines
    text = re.sub(r"Copyright.*?All Rights Reserved.*?\.", "", text, flags=re.IGNORECASE)
    # Remove common footer/nav text
    text = re.sub(r"Important links|CCCD @ IITJ|Recruitment|Correspondence|RTI|Tenders", "", text)
    text = re.sub(r"Web Policy|Web Information Manager|Feedback|CERT-IN|Help|NIRF", "", text)
    text = re.sub(r"Intranet Links|Old Website|Donations|How To Reach IITJ", "", text)
    text = re.sub(r"NCCR Portal|Institute Repository|Contact|Techscape", "", text)
    text = re.sub(r"Internal Committee", "", text)
    # Remove phone numbers
    text = re.sub(r"\d{4}[\s-]?\d{3}[\s-]?\d{4}", "", text)
    # Remove download file size info
    text = re.sub(r"\(Download file:?\s*\d+\s*KB\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\(\s*\d+\s*KB\s*,?\s*\)", "", text)
    # Remove non-ASCII characters (removes Hindi text etc.)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_and_clean(text, remove_stops=False):

    #Tokenize text and clean tokens for Word2Vec training.
    
    stop_words = set(stopwords.words("english")) if remove_stops else set()

    sentences = sent_tokenize(text)
    tokenized_sentences = []

    for sent in sentences:
        # Tokenize each sentence
        tokens = word_tokenize(sent.lower())
        # Keep only alphabetic tokens of length >= 2, remove stopwords if flagged
        cleaned = [
            tok for tok in tokens
            if tok.isalpha() and len(tok) >= 2 and tok not in stop_words
        ]
        if cleaned:  # Only keep non-empty sentences
            tokenized_sentences.append(cleaned)

    return tokenized_sentences


def compute_statistics(tokenized_docs, doc_names):
    """
    Compute and print dataset statistics as required by the assignment.
    
    Reports:
      - Total number of documents
      - Total number of tokens
      - Vocabulary size (unique tokens)
      - Top 20 most frequent words
    
    """
    all_tokens = []
    for doc_sentences in tokenized_docs:
        for sentence in doc_sentences:
            all_tokens.extend(sentence)

    vocab = set(all_tokens)
    freq = Counter(all_tokens)

    stats = {
        "num_docs": len(tokenized_docs),
        "num_tokens": len(all_tokens),
        "vocab_size": len(vocab),
        "top_20_words": freq.most_common(20),
    }

    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"  Total number of documents : {stats['num_docs']}")
    print(f"  Total number of tokens    : {stats['num_tokens']}")
    print(f"  Vocabulary size           : {stats['vocab_size']}")
    print(f"\n  Top 20 most frequent words:")
    for word, count in stats["top_20_words"]:
        print(f"    {word:20s} -> {count}")
    print("=" * 60)

    # Save statistics to file
    os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)
    with open(os.path.join(BASE_DIR, "outputs", "dataset_statistics.txt"), "w", encoding="utf-8") as f:
        f.write("DATASET STATISTICS\n")
        f.write("=" * 40 + "\n")
        f.write(f"Total number of documents : {stats['num_docs']}\n")
        f.write(f"Total number of tokens    : {stats['num_tokens']}\n")
        f.write(f"Vocabulary size           : {stats['vocab_size']}\n\n")
        f.write("Top 20 most frequent words:\n")
        for word, count in stats["top_20_words"]:
            f.write(f"  {word:20s} -> {count}\n")

    return stats


def generate_wordcloud(tokenized_docs):
    """
    Generate and save a Word Cloud visualization of the most frequent words.
    
    Uses the WordCloud library to create a visually appealing word cloud
    from the entire corpus. Saved to outputs/wordcloud.png.
    
    """
    # Flatten all tokens into a single string for the word cloud
    all_tokens = []
    for doc_sentences in tokenized_docs:
        for sentence in doc_sentences:
            all_tokens.extend(sentence)
    text_for_cloud = " ".join(all_tokens)

    # Generate word cloud with academic-friendly settings
    wc = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        max_words=150,
        colormap="viridis",
        contour_width=1,
        contour_color="steelblue",
    ).generate(text_for_cloud)

    # Save the word cloud
    os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)
    plt.figure(figsize=(14, 7))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud - IIT Jodhpur Corpus", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "outputs", "wordcloud.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("\n[INFO] Word Cloud saved to: outputs/wordcloud.png")


def preprocess_corpus():
    """
    Main preprocessing pipeline: loads raw corpus, cleans, tokenizes,
    computes stats, and generates word cloud.
    
    """
    # Load raw documents
    raw_dir = os.path.join(BASE_DIR, "data", "raw_documents")
    if not os.path.exists(raw_dir):
        print("[ERROR] No raw documents found. Run scraper.py first!")
        return []

    doc_files = sorted([f for f in os.listdir(raw_dir) if f.endswith(".txt")])
    print(f"Found {len(doc_files)} raw documents to preprocess.\n")

    all_sentences = []
    tokenized_docs = []

    for doc_file in doc_files:
        filepath = os.path.join(raw_dir, doc_file)
        with open(filepath, "r", encoding="utf-8") as f:
            raw_text = f.read()

        # Step 1: Remove boilerplate
        cleaned = remove_boilerplate(raw_text)

        # Step 2: Tokenize and clean (keep stopwords for Word2Vec context)
        sentences = tokenize_and_clean(cleaned, remove_stops=False)

        tokenized_docs.append(sentences)
        all_sentences.extend(sentences)
        print(f"  Processed: {doc_file} -> {sum(len(s) for s in sentences)} tokens in {len(sentences)} sentences")

    # Save cleaned corpus (one sentence per line, tokens space-separated)
    os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
    with open(os.path.join(BASE_DIR, "data", "cleaned_corpus.txt"), "w", encoding="utf-8") as f:
        for sentence in all_sentences:
            f.write(" ".join(sentence) + "\n")
    print(f"\n[INFO] Cleaned corpus saved to: data/cleaned_corpus.txt")
    print(f"       Total sentences: {len(all_sentences)}")

    # Compute and report statistics
    compute_statistics(tokenized_docs, doc_files)

    # Generate Word Cloud
    generate_wordcloud(tokenized_docs)

    return all_sentences


if __name__ == "__main__":
    preprocess_corpus()
