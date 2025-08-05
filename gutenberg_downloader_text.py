import os
import requests
from bs4 import BeautifulSoup
import time

# Output directory
os.makedirs("dataset/text", exist_ok=True)

# Gutenberg catalog page (pre-sorted random sample of 100 books)
BASE_URL = "https://www.gutenberg.org"
INDEX_URL = "https://www.gutenberg.org/ebooks/search/?sort_order=random&start_index={}"

def get_ebook_links(num_books=1000):
    links = []
    for start in range(1, num_books+1, 25):
        url = INDEX_URL.format(start)
        print(f"Fetching index: {url}")
        html = requests.get(url).text
        soup = BeautifulSoup(html, 'html.parser')
        for book_link in soup.select('.booklink a.link'):
            href = book_link.get("href")
            if href and href.startswith("/ebooks/"):
                links.append(BASE_URL + href)
            if len(links) >= num_books:
                break
        time.sleep(1)
    return links[:num_books]

def download_txt(url, book_id, file_index):
    txt_url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
    try:
        r = requests.get(txt_url, timeout=10)
        r.raise_for_status()
        content = r.text
        # Simple cleaning: remove headers/footers
        start = content.find("*** START") if "*** START" in content else 0
        end = content.find("*** END") if "*** END" in content else len(content)
        clean_text = content[start:end].strip()
        with open(f"dataset/text/sample_{file_index}.txt", "w", encoding="utf-8") as f:
            f.write(clean_text)
        print(f"Downloaded book #{file_index}: {txt_url}")
    except Exception as e:
        print(f"Failed to download {txt_url} — {e}")

if __name__ == "__main__":
    print("Downloading 100 diverse .txt books from Project Gutenberg...")
    book_links = get_ebook_links(100)
    for idx, book_url in enumerate(book_links):
        book_id = book_url.split("/")[-1]
        download_txt(book_url, book_id, idx)
        time.sleep(1.5)  # be polite to the server
    print("Done! All saved to dataset/text/")
