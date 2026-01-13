import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
import time
import os



BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_FILE = os.path.join(DATA_DIR, "raw_lic_documents.json")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; LIC-RAG-Bot/1.0)"
}

def fetch_page(url):
    session = requests.Session()

    retries = Retry(
        total=3,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )

    session.mount("https://", HTTPAdapter(max_retries=retries))

    response = session.get(url, headers=HEADERS, timeout=20)
    response.raise_for_status()
    return response.text


def extract_text(html):
    """Extract meaningful content from HTML, focusing on main content areas"""
    soup = BeautifulSoup(html, "html.parser")

    # Remove unwanted elements
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "aside", "iframe"]):
        tag.decompose()
    
    # Also remove common navigation elements
    for tag in soup.find_all(['div', 'ul'], class_=lambda x: x and any(nav in str(x).lower() for nav in ['nav', 'menu', 'breadcrumb', 'sidebar'])):
        tag.decompose()

    # Try to find main content area (common patterns)
    main_content = None
    
    # Try common main content selectors
    content_selectors = [
        {'class_': lambda x: x and 'main-content' in str(x).lower()},
        {'class_': lambda x: x and 'content-area' in str(x).lower()},
        {'class_': lambda x: x and 'article' in str(x).lower()},
        {'id': lambda x: x and 'content' in str(x).lower()},
        {'role': 'main'},
        {'class_': lambda x: x and 'container' in str(x).lower()},
    ]
    
    for selector in content_selectors:
        main_content = soup.find('div', **selector) or soup.find('main', **selector) or soup.find('article', **selector)
        if main_content:
            break
    
    # If no main content found, use body but filter more aggressively
    if not main_content:
        main_content = soup.find('body')
    
    if not main_content:
        # Fallback to all text
        text = soup.get_text(separator=" ")
        return " ".join(text.split())
    
    # Extract text with better formatting
    # Get all paragraphs, divs with substantial text, tables, lists
    text_elements = []
    
    for element in main_content.find_all(['p', 'div', 'li', 'td', 'th', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        text = element.get_text(separator=" ", strip=True)
        # Only include if it has substantial content (more than just a few words)
        if len(text) > 20 and not any(skip in text.lower() for skip in ['menu', 'display', 'breadcrumb', 'cookie', 'javascript']):
            text_elements.append(text)
    
    # Join with proper spacing
    full_text = " ".join(text_elements)
    
    # Clean up extra whitespace
    full_text = " ".join(full_text.split())
    
    return full_text if len(full_text) > 100 else soup.get_text(separator=" ")

def scrape_documents(urls):
    documents = []

    for url in urls:
        try:
            print(f"Scraping: {url}")

            html = fetch_page(url)
            soup = BeautifulSoup(html, "html.parser")

            title = soup.title.string.strip() if soup.title else "LIC Document"

            documents.append({
                "document_type": "life_insurance",
                "title": title,
                "source_url": url,
                "scraped_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "content": extract_text(html)
            })

            time.sleep(1)

        except Exception as e:
            print(f"Failed: {e}")

    return documents

def save_to_json(data):
    os.makedirs(DATA_DIR, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(data)} documents to:")
    print(OUTPUT_FILE)

if __name__ == "__main__":

    LIC_URLS = [
        # Plan categories
        "https://licindia.in/web/guest/endowment-plans",
        "https://licindia.in/web/guest/whole-life-plans",
        "https://licindia.in/web/guest/money-back-plans",
        "https://licindia.in/web/guest/term-assurance-plans",
        "https://licindia.in/web/guest/unit-linked-plans",
        "https://licindia.in/web/guest/pension-plan",
        
        # Specific popular policies (add more if available)
        "https://licindia.in/Products/Jeevan-Umang",
        "https://licindia.in/Products/New-Jeevan-Anand",
        "https://licindia.in/Products/Jeevan-Labh",
        
        # Information pages
        "https://licindia.in/web/guest/claims-settlement-requirements",
        "https://licindia.in/web/guest/policy-servicing",
        "https://licindia.in/web/guest/premium-payment",
        "https://licindia.in/web/guest/tax-benefits",
    ]

    print(f"Starting to scrape {len(LIC_URLS)} URLs...")
    print("This may take a few minutes...")
    print()
    
    docs = scrape_documents(LIC_URLS)
    save_to_json(docs)
    
    print(f"\n Scraping complete!")
    print(f"Next steps:")
    print(f"1. cd Preprocessing")
    print(f"2. python3 parse.py")
    print(f"3. python3 chunk.py")
    print(f"4. python3 embeddings.py")
