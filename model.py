import re
import string
import numpy as np
import scipy.sparse
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Define local NLTK data directory inside project root to ensure it is packaged by Render/Heroku
import os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
NLTK_DATA_DIR = os.path.join(PROJECT_ROOT, 'nltk_data')
if not os.path.exists(NLTK_DATA_DIR):
    os.makedirs(NLTK_DATA_DIR)

# Tell NLTK to look in this directory
if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.append(NLTK_DATA_DIR)

# Safe download of NLTK resources
def download_nltk_resources():
    resources = {
        'stopwords': 'corpora/stopwords',
        'wordnet': 'corpora/wordnet',
        'omw-1.4': 'corpora/omw-1.4',
        'vader_lexicon': 'sentiment/vader_lexicon'
    }
    for name, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, download_dir=NLTK_DATA_DIR, quiet=True)

# Initialize resources
download_nltk_resources()
try:
    STOPWORDS = set(stopwords.words('english'))
except:
    STOPWORDS = set()
LEMMATIZER = WordNetLemmatizer()
try:
    SIA = SentimentIntensityAnalyzer()
except:
    SIA = None

def preprocess_text(text):
    """
    Cleans raw review text by:
    - Lowercasing
    - Removing punctuation
    - Removing extra spaces
    - Tokenizing & Lemmatizing
    - Removing stopwords
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize and remove extra spacing
    words = re.findall(r'\b\w+\b', text)
    
    # Lemmatize and remove stopwords
    cleaned_words = [LEMMATIZER.lemmatize(word) for word in words if word not in STOPWORDS]
    
    return " ".join(cleaned_words)

def extract_dense_features(text):
    """
    Extracts hand-crafted numeric features:
    1. Review length (characters)
    2. Number of exclamation marks
    3. Number of capital letters
    4. Average word length
    5. Positive sentiment score
    6. Negative sentiment score
    """
    if not isinstance(text, str):
        text = ""
        
    length = len(text)
    excl_count = text.count('!')
    caps_count = sum(1 for c in text if c.isupper())
    
    words = text.split()
    avg_word_len = np.mean([len(w) for w in words]) if words else 0.0
    
    # Sentiment scores using VADER
    pos_score = 0.0
    neg_score = 0.0
    if SIA:
        scores = SIA.polarity_scores(text)
        pos_score = scores.get('pos', 0.0)
        neg_score = scores.get('neg', 0.0)
        
    return np.array([
        float(length),
        float(excl_count),
        float(caps_count),
        float(avg_word_len),
        float(pos_score),
        float(neg_score)
    ])

def explain_prediction(text, lr_model, vectorizer, scaler):
    """
    Explains the Logistic Regression prediction for a single review.
    Calculates log-odds contribution of both TF-IDF words and dense features.
    """
    explanation = {
        'genuine_indicators': [],
        'fake_indicators': [],
        'intercept': float(lr_model.intercept_[0]),
        'dense_metrics': {}
    }
    
    # 1. Capture original metrics
    dense_raw = extract_dense_features(text)
    explanation['dense_metrics'] = {
        'length': int(dense_raw[0]),
        'exclamation_marks': int(dense_raw[1]),
        'capital_letters': int(dense_raw[2]),
        'avg_word_len': round(dense_raw[3], 2),
        'positive_sentiment': round(dense_raw[4], 2),
        'negative_sentiment': round(dense_raw[5], 2)
    }
    
    # 2. Get scaled dense features and their coefficients
    dense_scaled = scaler.transform(dense_raw.reshape(1, -1))[0]
    num_vocab = len(vectorizer.get_feature_names_out())
    dense_coefs = lr_model.coef_[0][num_vocab:]
    
    dense_names = [
        'Review Length',
        'Exclamation Marks',
        'Capital Letters',
        'Average Word Length',
        'Positive Sentiment',
        'Negative Sentiment'
    ]
    
    # Compute contributions for dense features
    for name, val, coef in zip(dense_names, dense_scaled, dense_coefs):
        contrib = val * coef
        feat_info = {
            'feature': name,
            'contribution': float(contrib),
            'type': 'structural'
        }
        if contrib > 0.01:
            explanation['fake_indicators'].append(feat_info)
        elif contrib < -0.01:
            explanation['genuine_indicators'].append(feat_info)
            
    # 3. Get word (TF-IDF) feature contributions
    clean_text = preprocess_text(text)
    if clean_text:
        tfidf_vector = vectorizer.transform([clean_text])
        # Find active features
        nonzero_indices = tfidf_vector.nonzero()[1]
        word_features = vectorizer.get_feature_names_out()
        
        for idx in nonzero_indices:
            val = tfidf_vector[0, idx]
            coef = lr_model.coef_[0][idx]
            contrib = val * coef
            word = word_features[idx]
            
            feat_info = {
                'feature': f'Word: "{word}"',
                'contribution': float(contrib),
                'type': 'word'
            }
            if contrib > 0.01:
                explanation['fake_indicators'].append(feat_info)
            elif contrib < -0.01:
                explanation['genuine_indicators'].append(feat_info)
                
    # Sort indicators by absolute contribution
    explanation['fake_indicators'].sort(key=lambda x: abs(x['contribution']), reverse=True)
    explanation['genuine_indicators'].sort(key=lambda x: abs(x['contribution']), reverse=True)
    
    return explanation

def detect_platform(url):
    """
    Identifies the e-commerce platform from the product URL.
    """
    if not isinstance(url, str):
        return "Unknown Platform"
    
    url_lower = url.lower()
    if 'flipkart.com' in url_lower:
        return "Flipkart"
    elif 'amazon.' in url_lower or 'amzn.' in url_lower:
        return "Amazon"
    elif 'myntra.com' in url_lower:
        return "Myntra"
    elif 'meesho.com' in url_lower:
        return "Meesho"
    else:
        return "Generic E-Commerce"


def extract_product_name(url):
    """
    Parses an e-commerce URL (Amazon, Flipkart, Myntra, Meesho, or Generic)
    and extracts a clean, human-readable product title.
    """
    if not isinstance(url, str):
        return "E-Commerce Product"
    
    url = url.strip()
    
    # 1. Amazon pattern: domain.com/Product-Name/dp/B0...
    amazon_match = re.search(r'amazon\.[a-z\.]+/([^/]+)/dp/', url, re.IGNORECASE)
    if amazon_match:
        name = amazon_match.group(1).replace('-', ' ').replace('_', ' ')
        return name.title()
        
    # 2. Flipkart pattern: flipkart.com/product-name/p/itm...
    flipkart_match = re.search(r'flipkart\.com/([^/]+)/p/', url, re.IGNORECASE)
    if flipkart_match:
        name = flipkart_match.group(1).replace('-', ' ').replace('_', ' ')
        return name.title()

    # 3. Myntra pattern: myntra.com/category/brand/product-name/id/buy
    myntra_match = re.search(r'myntra\.com/(?:[^/]+/)*([^/]+)/\d+/buy', url, re.IGNORECASE)
    if myntra_match:
        name = myntra_match.group(1).replace('-', ' ').replace('_', ' ')
        return name.title()

    # 4. Meesho pattern: meesho.com/product-name/p/id
    meesho_match = re.search(r'meesho\.com/([^/]+)/p/', url, re.IGNORECASE)
    if meesho_match:
        name = meesho_match.group(1).replace('-', ' ').replace('_', ' ')
        return name.title()
        
    # 5. General URL fallback: parse last meaningful path segment
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        path = parsed.path.strip('/')
        if path:
            segments = path.split('/')
            for seg in reversed(segments):
                if '-' in seg or '_' in seg:
                    name = seg.replace('-', ' ').replace('_', ' ')
                    return name.title()
            return segments[-1].replace('-', ' ').replace('_', ' ').title()
    except Exception:
        pass
        
    return "E-Commerce Product"


def _build_flipkart_reviews_url(url):
    """
    Converts a Flipkart product page URL into its reviews endpoint URL.
    """
    try:
        from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
        parsed = urlparse(url)

        path = parsed.path
        if '/p/' in path:
            path = path.replace('/p/', '/product-reviews/')
        elif '/product-reviews/' not in path:
            path = path.rstrip('/') + '/product-reviews/'

        qs = parse_qs(parsed.query)
        new_qs = {}
        if 'pid' in qs:
            new_qs['pid'] = qs['pid'][0]
        if 'lid' in qs:
            new_qs['lid'] = qs['lid'][0]

        new_query = urlencode(new_qs)
        return urlunparse((parsed.scheme, parsed.netloc, path, '', new_query, ''))
    except Exception:
        return url


def _scrape_flipkart_reviews(url, max_pages=2):
    """
    Scrapes real customer reviews from Flipkart.
    Returns structured list of review dictionaries.
    """
    try:
        import requests
        from bs4 import BeautifulSoup

        try:
            import cloudscraper
            session = cloudscraper.create_scraper()
        except ImportError:
            session = requests.Session()

        reviews_url = _build_flipkart_reviews_url(url)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
            'Accept-Language': 'en-IN,en;q=0.9',
            'Referer': 'https://www.flipkart.com/',
        }

        reviews = []
        for page in range(1, max_pages + 1):
            page_url = reviews_url
            if page > 1:
                sep = '&' if '?' in reviews_url else '?'
                page_url = f"{reviews_url}{sep}page={page}"

            resp = session.get(page_url, headers=headers, timeout=8)
            if resp.status_code != 200:
                break

            soup = BeautifulSoup(resp.text, 'html.parser')
            # Target review containers
            containers = (
                soup.find_all('div', class_='ZmyHeo') or
                soup.find_all('div', class_='t-ZTKy') or
                soup.find_all('div', class_='_27M-gpx') or
                soup.find_all('div', {'class': lambda c: c and 'review' in c.lower()})
            )

            for c in containers:
                text = c.get_text(separator=' ', strip=True)
                if 20 < len(text) < 2000:
                    lower = text.lower()
                    skip_words = ['read more', 'helpful', 'report', 'reply', 'certified buyer']
                    if not any(s in lower for s in skip_words):
                        # Extract parent card if available for author/rating
                        parent = c.find_parent('div', class_=lambda cl: cl and ('cPHJh8' in cl or 'col' in cl or 'row' in cl))
                        rating = None
                        author = None
                        if parent:
                            rating_el = parent.find('div', class_=lambda cl: cl and 'XD0979' in cl) or parent.find('div', class_=lambda cl: cl and 'rating' in cl.lower())
                            if rating_el:
                                try:
                                    rating = float(re.findall(r'\d+(?:\.\d+)?', rating_el.get_text())[0])
                                except Exception:
                                    pass
                            author_el = parent.find('p', class_=lambda cl: cl and '_2NsA9' in cl) or parent.find('p', class_=lambda cl: cl and 'author' in cl.lower())
                            if author_el:
                                author = author_el.get_text(strip=True)

                        reviews.append({
                            "review_text": text,
                            "rating": rating,
                            "author": author,
                            "date": None,
                            "verified": True if 'certified buyer' in c.find_parent().get_text().lower() else None,
                            "source": "Live Scraped - Flipkart"
                        })
        
        # Deduplicate
        seen = set()
        unique = []
        for r in reviews:
            key = r['review_text'][:80]
            if key not in seen:
                seen.add(key)
                unique.append(r)
        return unique
    except Exception:
        return []


def _scrape_amazon_reviews(url):
    """
    Scrapes real customer reviews from Amazon product pages.
    """
    try:
        import requests
        from bs4 import BeautifulSoup
        try:
            import cloudscraper
            session = cloudscraper.create_scraper()
        except ImportError:
            session = requests.Session()

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
        }

        resp = session.get(url, headers=headers, timeout=8)
        if resp.status_code != 200:
            return []

        soup = BeautifulSoup(resp.text, 'html.parser')
        review_cards = soup.find_all('div', {'data-hook': 'review'}) or soup.find_all('div', class_=lambda c: c and 'review' in c.lower())

        reviews = []
        for card in review_cards:
            body = card.find('span', {'data-hook': 'review-body'}) or card.find('div', class_=lambda c: c and 'review-text' in c.lower())
            if body:
                text = body.get_text(separator=' ', strip=True)
                if len(text) > 15:
                    rating_el = card.find('i', {'data-hook': 'review-star-rating'}) or card.find('i', class_=lambda c: c and 'star' in c.lower())
                    rating = None
                    if rating_el:
                        try:
                            rating = float(re.findall(r'\d+(?:\.\d+)?', rating_el.get_text())[0])
                        except Exception:
                            pass

                    author_el = card.find('span', class_='a-profile-name')
                    author = author_el.get_text(strip=True) if author_el else None

                    date_el = card.find('span', {'data-hook': 'review-date'})
                    date_str = date_el.get_text(strip=True) if date_el else None

                    verified_el = card.find('span', {'data-hook': 'avp-badge'})
                    verified = True if verified_el else None

                    reviews.append({
                        "review_text": text,
                        "rating": rating,
                        "author": author,
                        "date": date_str,
                        "verified": verified,
                        "source": "Live Scraped - Amazon"
                    })

        return reviews
    except Exception:
        return []


def _scrape_myntra_reviews(url):
    """
    Scrapes real customer reviews from Myntra.
    """
    try:
        import requests
        from bs4 import BeautifulSoup
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
        }
        resp = requests.get(url, headers=headers, timeout=8)
        if resp.status_code != 200:
            return []

        soup = BeautifulSoup(resp.text, 'html.parser')
        containers = soup.find_all('div', class_=lambda c: c and 'user-review' in c.lower())

        reviews = []
        for c in containers:
            text = c.get_text(separator=' ', strip=True)
            if len(text) > 15:
                reviews.append({
                    "review_text": text,
                    "rating": None,
                    "author": None,
                    "date": None,
                    "verified": True,
                    "source": "Live Scraped - Myntra"
                })
        return reviews
    except Exception:
        return []


def _scrape_meesho_reviews(url):
    """
    Scrapes real customer reviews from Meesho.
    """
    try:
        import requests
        from bs4 import BeautifulSoup
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
        }
        resp = requests.get(url, headers=headers, timeout=8)
        if resp.status_code != 200:
            return []

        soup = BeautifulSoup(resp.text, 'html.parser')
        containers = soup.find_all('div', class_=lambda c: c and 'comment' in c.lower())

        reviews = []
        for c in containers:
            text = c.get_text(separator=' ', strip=True)
            if len(text) > 15:
                reviews.append({
                    "review_text": text,
                    "rating": None,
                    "author": None,
                    "date": None,
                    "verified": True,
                    "source": "Live Scraped - Meesho"
                })
        return reviews
    except Exception:
        return []


def _scrape_generic_reviews(url):
    """
    Generic scraper for unsupported or general e-commerce websites.
    Identifies review structures using semantic CSS patterns.
    """
    try:
        import requests
        from bs4 import BeautifulSoup
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
        }
        resp = requests.get(url, headers=headers, timeout=8)
        if resp.status_code != 200:
            return []

        soup = BeautifulSoup(resp.text, 'html.parser')
        containers = soup.find_all(['div', 'section', 'li'], class_=lambda c: c and any(w in c.lower() for w in ['review', 'comment', 'feedback', 'testimonial']))

        reviews = []
        for c in containers:
            text = c.get_text(separator=' ', strip=True)
            if 20 < len(text) < 1500:
                reviews.append({
                    "review_text": text,
                    "rating": None,
                    "author": None,
                    "date": None,
                    "verified": None,
                    "source": "Live Scraped - Generic Web Page"
                })
        return reviews
    except Exception:
        return []


def fetch_product_reviews(url, product_name):
    """
    Main entry point for fetching real product reviews.
    
    STRICT ACCURACY GUARANTEE:
    - Only returns reviews actually scraped from the given product URL.
    - NEVER generates or substitutes dataset/synthetic fallback reviews.
    
    Returns:
        tuple: (reviews_list, platform_name, error_message)
    """
    platform = detect_platform(url)

    if platform == "Flipkart":
        reviews = _scrape_flipkart_reviews(url)
    elif platform == "Amazon":
        reviews = _scrape_amazon_reviews(url)
    elif platform == "Myntra":
        reviews = _scrape_myntra_reviews(url)
    elif platform == "Meesho":
        reviews = _scrape_meesho_reviews(url)
    else:
        reviews = _scrape_generic_reviews(url)

    if reviews and len(reviews) > 0:
        return reviews, platform, None

    # Honest failure handling: Return 0 reviews with explanatory message
    error_msg = (
        f"Unable to fetch reviews from this {platform} product page. "
        "The website may block automated scrapers, require login authentication, "
        "or contain no publicly accessible customer reviews."
    )
    return [], platform, error_msg


