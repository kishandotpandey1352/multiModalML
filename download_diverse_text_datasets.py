
import os
import requests
from datasets_xx import load_dataset
from bs4 import BeautifulSoup

os.makedirs("dataset/text/gutenberg", exist_ok=True)
os.makedirs("dataset/text/huggingface", exist_ok=True)
os.makedirs("dataset/text/kaggle", exist_ok=True)  # You must manually download & unzip here

def download_gutenberg_books(num_books=100):
    print("📚 Downloading books from Project Gutenberg...")
    base_url = "https://www.gutenberg.org"
    search_url = f"{base_url}/ebooks/search/?sort_order=downloads"
    try:
        resp = requests.get(search_url)
        soup = BeautifulSoup(resp.content, "html.parser")
        links = soup.select("li.booklink a.link")[:num_books]
        count = 0
        for a in links:
            href = a.get("href", "")
            book_id = href.split("/")[-1]
            txt_url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
            try:
                txt_data = requests.get(txt_url, timeout=10)
                if txt_data.status_code == 200:
                    with open(f"dataset/text/gutenberg/book_{count}.txt", "w", encoding="utf-8") as f:
                        f.write(txt_data.text)
                    count += 1
            except Exception as e:
                continue
    except Exception as e:
        print(f"⚠️ Gutenberg download failed: {e}")

def download_huggingface_text_samples():
    print("🤗 Downloading from HuggingFace datasets...")
    sources = ["ag_news", "yelp_polarity", "imdb", "dbpedia_14", "civil_comments"]
    max_per_source = 200
    for source in sources:
        try:
            dataset = load_dataset(source, split="train")
            for i, sample in enumerate(dataset):
                if i >= max_per_source:
                    break
                text = sample.get("text") or sample.get("content") or str(sample)
                with open(f"dataset/text/huggingface/{source}_{i}.txt", "w", encoding="utf-8") as f:
                    f.write(text)
        except Exception as e:
            print(f"⚠️ Failed to load {source}: {e}")

if __name__ == "__main__":
    download_gutenberg_books()
    download_huggingface_text_samples()

    print("\n✅ DONE: All datasets downloaded into 'dataset/text/'")
    print("📦 NOTE: For Kaggle, download and unzip text datasets manually into: dataset/text/kaggle/")
