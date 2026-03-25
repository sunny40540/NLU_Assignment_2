# Web scraping module - scrapes IITJ pages and Academic Regulations PDF

import os
import re
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader

# =========================================================================
# Configuration: 10 selected URLs covering diverse IIT Jodhpur content
# =========================================================================
URLS = [
    # General / About pages
    "https://iitj.ac.in/",                                          # 1. Homepage
    "https://iitj.ac.in/main/en/introduction",                      # 2. Introduction
    "https://iitj.ac.in/main/en/history",                           # 3. History

    # Academic pages
    "https://iitj.ac.in/office-of-academics/en/academics",          # 4. Office of Academics
    "https://iitj.ac.in/Bachelor-of-Technology/en/Bachelor-of-Technology",  # 5. BTech Programs

    # Department pages
    "https://cse.iitj.ac.in/",                                      # 6. CSE Department
    "https://iitj.ac.in/mechanical-engineering/en/undergraduate-program",  # 7. ME UG Program

    # Research page
    "https://iitj.ac.in/office-of-research-development/en/office-of-research-and-development",  # 8. R&D

    # Admissions / Doctoral
    "https://iitj.ac.in/computer-science-engineering/en/doctoral-programs",  # 9. PhD Programs

    # Faculty page
    "https://iitj.ac.in/People/List?dept=Computer-Science-Engineering&c=1c8346b0-07a4-401e-bce0-7f571bb4fabd",  # 10. CSE Faculty
]

# Path to the Academic Regulations PDF (mandatory source)
ACAD_REG_PDF = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Academic_Regulations_Final_03_09_2019.pdf")


def scrape_webpage(url):
    """
    Scrape textual content from a given URL.
    
    Steps:
      1. Send HTTP GET request to the URL
      2. Parse the HTML using BeautifulSoup
      3. Remove script, style, nav, footer, and header tags (boilerplate)
      4. Extract visible text from the remaining HTML
    
    Args:
        url (str): The URL to scrape
    
    Returns:
        str: Cleaned text content from the webpage
    """
    try:
        # Send request with a browser User-Agent to avoid blocks
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0"}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        # Parse HTML content
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove boilerplate elements: scripts, styles, navigation, footers
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "meta", "link"]):
            tag.decompose()

        # Extract visible text
        text = soup.get_text(separator=" ", strip=True)
        return text

    except Exception as e:
        print(f"  [WARNING] Failed to scrape {url}: {e}")
        return ""


def extract_pdf_text(pdf_path):
    """
    Extract text content from the Academic Regulations PDF.
    
    Uses PyPDF2 to read all pages and concatenate the extracted text.
    This is a mandatory data source as per the assignment requirements.
    
    Args:
        pdf_path (str): Path to the PDF file
    
    Returns:
        str: Extracted text content from the PDF
    """
    try:
        reader = PdfReader(pdf_path)
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        return " ".join(text_parts)
    except Exception as e:
        print(f"  [WARNING] Failed to read PDF {pdf_path}: {e}")
        return ""


def scrape_all():
    """
    Main scraping function: scrape all 10 URLs + Academic Regulations PDF.
    
    Creates separate document files for each source and also a combined
    raw corpus file. Saves everything under data/raw_documents/.
    
    Returns:
        list[dict]: List of documents, each with 'source' and 'text' keys
    """
    # Create output directories
    os.makedirs("data/raw_documents", exist_ok=True)

    documents = []

    # ---- Scrape 10 web pages ----
    print("=" * 60)
    print("SCRAPING IIT JODHPUR WEB PAGES")
    print("=" * 60)
    for i, url in enumerate(URLS, 1):
        print(f"\n[{i}/10] Scraping: {url}")
        text = scrape_webpage(url)
        if text:
            doc = {"source": url, "text": text}
            documents.append(doc)
            # Save individual document
            filename = f"data/raw_documents/doc_{i:02d}_web.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"SOURCE: {url}\n\n{text}")
            print(f"  -> Saved: {filename} ({len(text)} chars)")
        else:
            print(f"  -> SKIPPED (no text extracted)")

    # ---- Extract Academic Regulations PDF (mandatory) ----
    print(f"\n{'=' * 60}")
    print("EXTRACTING ACADEMIC REGULATIONS PDF")
    print("=" * 60)
    pdf_text = extract_pdf_text(ACAD_REG_PDF)
    if pdf_text:
        doc = {"source": "Academic_Regulations_PDF", "text": pdf_text}
        documents.append(doc)
        filename = "data/raw_documents/doc_11_academic_regulations.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"SOURCE: Academic Regulations PDF\n\n{pdf_text}")
        print(f"  -> Saved: {filename} ({len(pdf_text)} chars)")

    # ---- Save combined raw corpus ----
    combined = "\n\n".join([d["text"] for d in documents])
    with open("data/raw_corpus.txt", "w", encoding="utf-8") as f:
        f.write(combined)
    print(f"\n{'=' * 60}")
    print(f"TOTAL: {len(documents)} documents scraped")
    print(f"Combined raw corpus: {len(combined)} characters")
    print(f"Saved to: data/raw_corpus.txt")
    print("=" * 60)

    return documents


if __name__ == "__main__":
    scrape_all()
