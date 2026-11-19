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
    Extracts numerical and stylometric features with logarithmic dampening
    to prevent exclamation marks and capital letters from artificially overpowering
    semantic vocabulary features.
    
    Features:
    1. Log-scaled Review length
    2. Log-scaled Exclamation count
    3. Log-scaled Capital letter count
    4. Average word length
    5. Positive sentiment score (VADER)
    6. Negative sentiment score (VADER)
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
        float(np.log1p(length)),
        float(np.log1p(excl_count)),
        float(np.log1p(caps_count)),
        float(avg_word_len),
        float(pos_score),
        float(neg_score)
    ])


def classify_probability(prob_fake, lower_thresh=0.40, upper_thresh=0.60):
    """
    Classifies a prediction into Genuine, Fake, or Uncertain based on probability margins.
    - prob_fake <= lower_thresh (default 0.40): Genuine
    - prob_fake >= upper_thresh (default 0.60): Fake
    - lower_thresh < prob_fake < upper_thresh (40% to 60%): Uncertain / Inconclusive Evidence
    
    Returns:
        tuple: (label: str, confidence_pct: float, is_uncertain: bool)
    """
    prob_genuine = 1.0 - prob_fake
    if prob_fake >= upper_thresh:
        return "Fake", round(prob_fake * 100, 2), False
    elif prob_fake <= lower_thresh:
        return "Genuine", round(prob_genuine * 100, 2), False
    else:
        margin = abs(prob_fake - 0.50)
        conf = round((0.50 + margin) * 100, 2)
        return "Uncertain", conf, True


def explain_prediction(text, lr_model, vectorizer, scaler):
    """
    Explains the model prediction for a review using linear log-odds attributions.
    Calculates contributions of both TF-IDF words and dense features.
    """
    explanation = {
        'genuine_indicators': [],
        'fake_indicators': [],
        'intercept': float(lr_model.intercept_[0]) if hasattr(lr_model, 'intercept_') else 0.0,
        'dense_metrics': {}
    }
    
    # 1. Capture original unscaled metrics for human inspection
    raw_length = len(text) if isinstance(text, str) else 0
    raw_excl = text.count('!') if isinstance(text, str) else 0
    raw_caps = sum(1 for c in text if c.isupper()) if isinstance(text, str) else 0
    words = text.split() if isinstance(text, str) else []
    raw_avg_word = round(np.mean([len(w) for w in words]), 2) if words else 0.0
    
    scores = SIA.polarity_scores(text) if SIA and isinstance(text, str) else {}
    explanation['dense_metrics'] = {
        'length': raw_length,
        'exclamation_marks': raw_excl,
        'capital_letters': raw_caps,
        'avg_word_len': raw_avg_word,
        'positive_sentiment': round(scores.get('pos', 0.0), 2),
        'negative_sentiment': round(scores.get('neg', 0.0), 2)
    }
    
    # 2. Get scaled dense features and their coefficients
    dense_vec = extract_dense_features(text)
    dense_scaled = scaler.transform(dense_vec.reshape(1, -1))[0]
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
    Handles standard domains, country-specific TLDs, and short links.
    """
    if not isinstance(url, str):
        return "E-Commerce"
    
    url_lower = url.lower()
    if 'flipkart' in url_lower or 'fkrt.it' in url_lower:
        return "Flipkart"
    elif 'amazon' in url_lower or 'amzn.' in url_lower:
        return "Amazon"
    elif 'myntra' in url_lower:
        return "Myntra"
    elif 'meesho' in url_lower or 'mshp' in url_lower:
        return "Meesho"
    elif 'ajio' in url_lower:
        return "Ajio"
    elif 'croma' in url_lower:
        return "Croma"
    elif 'reliancedigital' in url_lower:
        return "Reliance Digital"
    elif 'nykaa' in url_lower:
        return "Nykaa"
    else:
        try:
            from urllib.parse import urlparse
            netloc = urlparse(url).netloc
            clean_domain = netloc.replace('www.', '').split('.')[0]
            if clean_domain:
                return clean_domain.capitalize()
        except Exception:
            pass
        return "Generic E-Commerce"


def extract_product_name(url):
    """
    Parses an e-commerce URL (Amazon, Flipkart, Myntra, Meesho, Ajio, Croma, etc.)
    and extracts a clean, human-readable product title.
    """
    if not isinstance(url, str):
        return "E-Commerce Product"
    
    url = url.strip()
    
    # 1. Amazon pattern: domain.com/Product-Name/dp/B0...
    amazon_match = re.search(r'amazon\.[a-z\.]+/([^/]+)/(?:dp|gp/product)/', url, re.IGNORECASE)
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

    # 5. Ajio pattern: ajio.com/product-name/p/id
    ajio_match = re.search(r'ajio\.com/(?:[^/]+/)*([^/]+)/p/', url, re.IGNORECASE)
    if ajio_match:
        name = ajio_match.group(1).replace('-', ' ').replace('_', ' ')
        return name.title()

    # 6. Croma pattern: croma.com/product-name/p/id
    croma_match = re.search(r'croma\.com/([^/]+)/p/', url, re.IGNORECASE)
    if croma_match:
        name = croma_match.group(1).replace('-', ' ').replace('_', ' ')
        return name.title()

    # 7. Reliance Digital pattern: reliancedigital.in/product-name/p/id
    rd_match = re.search(r'reliancedigital\.in/([^/]+)/p/', url, re.IGNORECASE)
    if rd_match:
        name = rd_match.group(1).replace('-', ' ').replace('_', ' ')
        return name.title()
        
    # 8. General URL fallback: parse last meaningful slug
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
            if segments[-1] and len(segments[-1]) > 3:
                return segments[-1].replace('-', ' ').replace('_', ' ').title()
    except Exception:
        pass
        
    return "E-Commerce Product"


def _extract_amazon_asin(url):
    """
    Extracts the 10-character Amazon Standard Identification Number (ASIN)
    from full product URLs, short links, or reviews endpoints.
    """
    if not isinstance(url, str):
        return None
    patterns = [
        r'/(?:dp|gp/product|product-reviews|gp/aw/d)/([A-Z0-9]{10})',
        r'[?&]asin=([A-Z0-9]{10})',
        r'/([A-Z0-9]{10})(?:[/?&]|$)'
    ]
    for pat in patterns:
        m = re.search(pat, url, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return None


def is_safe_url(url):
    """
    Validates that a URL is a legitimate, publicly routable HTTP/HTTPS URL.
    Mitigates SSRF attempts to private networks, loopback, link-local, or cloud metadata.
    Returns (is_safe: bool, reason_or_cleaned_url: str).
    """
    import socket
    import ipaddress
    from urllib.parse import urlparse

    if not isinstance(url, str):
        return False, "URL must be a string."
    
    url = url.strip()
    try:
        parsed = urlparse(url)
    except Exception as e:
        return False, f"Invalid URL format: {e}"
        
    if parsed.scheme not in ("http", "https"):
        return False, "Only HTTP and HTTPS URLs are supported."

    if parsed.port and parsed.port not in (80, 443):
        return False, f"Port {parsed.port} is not permitted. Only standard HTTP (80) and HTTPS (443) ports are allowed."
        
    hostname = parsed.hostname
    if not hostname:
        return False, "URL must contain a valid domain name."
        
    # Block loopback and internal hostname strings
    blocked_hosts = {"localhost", "127.0.0.1", "0.0.0.0", "::1", "metadata.google.internal"}
    host_lower = hostname.lower()
    if host_lower in blocked_hosts or host_lower.endswith(".local") or host_lower.endswith(".internal"):
        return False, "Access to local or internal network destinations is blocked."

    # Resolve IP and verify it is not private/loopback/link-local/reserved
    try:
        addr_info = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        if not addr_info:
            return False, f"Unable to resolve domain: {hostname}"
            
        for family, socktype, proto, canonname, sockaddr in addr_info:
            ip_str = sockaddr[0]
            ip = ipaddress.ip_address(ip_str)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
                return False, f"Access to private/internal IP address ({ip_str}) is prohibited."
    except socket.gaierror:
        return False, f"Domain name '{hostname}' could not be resolved."
    except Exception as e:
        return False, f"Network validation error: {e}"
        
    return True, url


def safe_fetch_url(url, timeout=12, max_redirects=6, return_final_url=False):
    """
    Safely executes an HTTP GET request, re-validating the destination and each
    redirect hop against SSRF restrictions, redirect loops, and response size limits.
    """
    import requests
    from urllib.parse import urljoin

    current_url = url
    session = _get_session()
    visited_urls = set()
    MAX_RESPONSE_BYTES = 5 * 1024 * 1024  # 5 MB safe limit
    
    for _ in range(max_redirects + 1):
        if current_url in visited_urls:
            err = "Redirect loop detected."
            return (None, current_url, err) if return_final_url else (None, err)
        visited_urls.add(current_url)

        is_safe, msg = is_safe_url(current_url)
        if not is_safe:
            return (None, current_url, msg) if return_final_url else (None, msg)
            
        try:
            resp = session.get(current_url, timeout=timeout, allow_redirects=False, stream=True)
            if resp.is_redirect or resp.status_code in (301, 302, 303, 307, 308):
                location = resp.headers.get("Location")
                if not location:
                    break
                current_url = urljoin(current_url, location)
                continue
            
            # Read response with size bounding to avoid memory exhaustion
            cl = resp.headers.get("Content-Length")
            if cl and cl.isdigit() and int(cl) > MAX_RESPONSE_BYTES:
                err = "Response size exceeds safe limit (5 MB)."
                return (None, current_url, err) if return_final_url else (None, err)

            content_chunks = []
            bytes_read = 0
            for chunk in resp.iter_content(chunk_size=65536):
                bytes_read += len(chunk)
                if bytes_read > MAX_RESPONSE_BYTES:
                    err = "Response size exceeds safe limit (5 MB)."
                    return (None, current_url, err) if return_final_url else (None, err)
                content_chunks.append(chunk)
            resp._content = b"".join(content_chunks)

            return (resp, current_url, None) if return_final_url else (resp, None)
        except requests.exceptions.Timeout:
            err = "Connection timed out while contacting e-commerce server."
            return (None, current_url, err) if return_final_url else (None, err)
        except Exception as e:
            err = f"Network request failed: {e}"
            return (None, current_url, err) if return_final_url else (None, err)
            
    err = "Too many redirects encountered."
    return (None, current_url, err) if return_final_url else (None, err)


def resolve_canonical_url(url, timeout=12, max_redirects=6):
    """
    Safely resolves shortened or vanity e-commerce URLs (e.g., amzn.in, amzn.to, fkrt.it, dl.flipkart.com)
    to their full destination product URL while enforcing SSRF checks on each hop.
    
    Returns:
        tuple: (canonical_url: str, resp_or_none, error_msg: str or None)
    """
    is_safe, msg = is_safe_url(url)
    if not is_safe:
        return url, None, msg
        
    result = safe_fetch_url(url, timeout=timeout, max_redirects=max_redirects, return_final_url=True)
    if isinstance(result, tuple) and len(result) == 3:
        resp, final_url, err = result
        return (final_url or url), resp, err
    elif isinstance(result, tuple) and len(result) == 2:
        resp, err = result
        return url, resp, err
    return url, None, "Unable to resolve URL."


def fetch_product_reviews(url, product_name):
    """
    Main entry point for fetching real customer reviews from e-commerce product URLs.
    Validates URLs against SSRF and platform restrictions, resolves shortened URLs,
    extracts available reviews across supported platforms, and returns deduplicated real customer data.
    
    Returns:
        tuple: (reviews_list, platform_name, error_message)
    """
    is_safe, err = is_safe_url(url)
    if not is_safe:
        return [], "Unknown", err

    # 1. Resolve canonical URL to handle shortened links (amzn.in, amzn.to, fkrt.it, dl.flipkart.com)
    canonical_url, _, _ = resolve_canonical_url(url)
    target_url = canonical_url if canonical_url else url

    platform = detect_platform(target_url)

    reviews = []
    error_reason = None

    if platform == "Flipkart":
        reviews, error_reason = _scrape_flipkart_reviews(target_url)
    elif platform == "Amazon":
        reviews, error_reason = _scrape_amazon_reviews(target_url)
    elif platform == "Myntra":
        reviews, error_reason = _scrape_myntra_reviews(target_url)
    elif platform == "Meesho":
        reviews, error_reason = _scrape_meesho_reviews(target_url)
    elif platform == "Ajio":
        reviews, error_reason = _scrape_ajio_reviews(target_url)
    elif platform == "Croma":
        reviews, error_reason = _scrape_croma_reviews(target_url)
    elif platform == "Reliance Digital":
        reviews, error_reason = _scrape_reliance_digital_reviews(target_url)
    else:
        reviews, error_reason = _scrape_generic_reviews(target_url)

    # Deduplicate reviews based on normalized review text
    unique_reviews = []
    seen_texts = set()
    for r in reviews:
        norm = re.sub(r'\s+', ' ', (r.get("review_text") or "").strip().lower())
        if norm and len(norm) >= 15 and norm not in seen_texts:
            seen_texts.add(norm)
            unique_reviews.append(r)

    # If actual reviews were successfully extracted, return them
    if unique_reviews:
        return unique_reviews, platform, None

    # Extraction failed or 0 reviews found - DO NOT generate synthetic reviews.
    # Return empty list and truthful failure reason.
    failure_msg = error_reason or (
        f"Unable to retrieve actual product reviews from {platform}. "
        f"The platform returned no reviews or access was blocked by third-party anti-bot protections."
    )
    return [], platform, failure_msg



# -----------------------------------------------------------------------------
# PLATFORM-SPECIFIC SCRAPERS
# -----------------------------------------------------------------------------

def _get_session():
    """Returns a requests.Session with browser-like headers to reduce bot detection."""
    import requests
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-IN,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer": "https://www.google.com/",
        "DNT": "1",
    })
    return session


def _scrape_flipkart_reviews(url):
    """Attempts to scrape customer reviews from a Flipkart product page."""
    try:
        from bs4 import BeautifulSoup
        resp, err = safe_fetch_url(url, timeout=12)
        if not resp:
            return [], err or "Unable to connect to Flipkart server."
        if resp.status_code == 403 or resp.status_code == 503:
            return [], f"Access blocked by Flipkart anti-bot protections (HTTP {resp.status_code})."
        if resp.status_code != 200:
            return [], f"Flipkart returned HTTP error status {resp.status_code}."

        soup = BeautifulSoup(resp.text, "html.parser")
        review_blocks = soup.select("div.col.EPCmJX, div._27M-vq, div.t-ZTKy")

        reviews = []
        for block in review_blocks[:30]:
            text_tag = block.find("p") or block.find("div", {"class": lambda c: c and "t-ZTKy" in (c or "")})
            if not text_tag or not text_tag.get_text(strip=True):
                continue
            review_text = text_tag.get_text(separator=" ", strip=True)
            if len(review_text) < 15:
                continue
            rating_tag = block.find("div", {"class": lambda c: c and "XQDdHH" in (c or "")})
            rating = None
            if rating_tag:
                try:
                    rating = float(rating_tag.get_text(strip=True))
                except Exception:
                    pass
            # Check for certified buyer indicator without fabricating verification
            ver_tag = block.find(string=re.compile(r"certified buyer|verified purchase", re.IGNORECASE))
            verified = True if ver_tag else None

            item_match = re.search(r'/p/(itm[a-zA-Z0-9]+)', url)
            product_id = item_match.group(1) if item_match else None

            reviews.append({
                "review_text": review_text,
                "rating": rating,
                "author": "Flipkart Customer",
                "date": None,
                "verified": verified,
                "source": "Live Scraped - Flipkart",
                "product_id": product_id,
                "product_url": url
            })

        if not reviews:
            return [], "Page loaded successfully, but zero review elements were found in Flipkart HTML. Reviews may require dynamic JavaScript rendering."
        return reviews, None
    except Exception as e:
        return [], f"Extraction parsing error: {e}"


def _detect_amazon_bot_challenge(resp):
    """
    Detects whether an Amazon HTTP response represents a CAPTCHA or anti-bot challenge,
    even when served with HTTP 200 OK.
    Returns:
        tuple: (is_challenge: bool, reason: str or None)
    """
    if not resp or not hasattr(resp, 'text'):
        return False, None

    text = resp.text.lower()
    
    # 1. Page title checks
    if "<title>robot check</title>" in text or "<title>amazon captcha</title>" in text:
        return True, "Amazon anti-bot challenge encountered (Robot Check CAPTCHA). Cloud hosting IPs are blocked from automated extraction."
        
    # 2. Form actions and URLs
    if "/errors/validatecaptcha" in text or "validatecaptcha" in text:
        return True, "Amazon anti-bot challenge encountered (validateCaptcha form required). Cloud hosting IPs are blocked from automated extraction."

    # 3. Explicit challenge prompts and automated access disclaimers
    if "enter the characters you see below" in text or "type the characters you see in this image" in text:
        return True, "Amazon anti-bot challenge encountered (Image CAPTCHA). Cloud hosting IPs are blocked from automated extraction."

    if "to discuss automated access to amazon data" in text or "api-services-support@amazon.com" in text:
        return True, "Amazon automated access restriction encountered. Cloud hosting IPs are blocked from automated extraction."

    return False, None


def _scrape_amazon_reviews(url):
    """Attempts to scrape customer reviews from Amazon."""
    try:
        from bs4 import BeautifulSoup
        from urllib.parse import urlparse
        reviews_url = url
        asin = _extract_amazon_asin(url)
        if asin:
            parsed = urlparse(url)
            netloc = parsed.netloc.lower()
            clean_host = netloc[4:] if netloc.startswith("www.") else netloc
            if "amazon." not in clean_host:
                clean_host = "amazon.in"
            reviews_url = f"https://www.{clean_host}/product-reviews/{asin}?pageNumber=1&sortBy=recent"

        resp, err = safe_fetch_url(reviews_url, timeout=12)
        if not resp:
            return [], err or "Unable to connect to Amazon server."
        if resp.status_code == 403 or resp.status_code == 503:
            return [], f"Access blocked by Amazon anti-bot protections (HTTP {resp.status_code})."
        if resp.status_code != 200:
            return [], f"Amazon returned HTTP error status {resp.status_code}."

        # Detect CAPTCHA and Robot Check challenges served with HTTP 200 OK
        is_challenge, challenge_msg = _detect_amazon_bot_challenge(resp)
        if is_challenge:
            return [], challenge_msg

        soup = BeautifulSoup(resp.text, "html.parser")
        review_divs = soup.select("div[data-hook='review'], div.review, div[id^='customer_review-']")
        reviews = []
        for div in review_divs[:30]:
            body = div.select_one("span[data-hook='review-body'], div[data-hook='review-collapsed']")
            if not body:
                body = div.select_one(".review-text-content, .review-text")
            if not body:
                continue
            review_text = body.get_text(separator=" ", strip=True)
            if len(review_text) < 15:
                continue
            rating = None
            rating_tag = div.select_one("i[data-hook='review-star-rating'], i.a-icon-star")
            if rating_tag:
                m = re.search(r"(\d+\.?\d*)", rating_tag.get_text())
                if m:
                    try:
                        rating = float(m.group(1))
                    except Exception:
                        pass
            author_tag = div.select_one("span.a-profile-name, span.author")
            author = author_tag.get_text(strip=True) if author_tag else "Amazon Customer"
            date_tag = div.select_one("span[data-hook='review-date']")
            date = date_tag.get_text(strip=True) if date_tag else None
            avp_badge = div.select_one("span[data-hook='avp-badge']")
            verified = True if avp_badge else None
            reviews.append({
                "review_text": review_text,
                "rating": rating,
                "author": author,
                "date": date,
                "verified": verified,
                "source": "Live Scraped - Amazon",
                "product_id": asin,
                "product_url": url
            })

        if not reviews:
            # Check for verified empty reviews vs dynamic client rendering
            no_reviews_indicator = soup.find(string=re.compile(r"no customer reviews|be the first to review|not yet reviewed", re.IGNORECASE))
            if no_reviews_indicator:
                return [], "This product currently has no written customer reviews."
            
            # Check if product title exists to confirm it is a valid product page
            product_title_tag = soup.select_one("#productTitle, #title, span#cr-arp-byline, h1")
            if product_title_tag or asin:
                return [], "Product page loaded, but reviews are rendered dynamically via client-side JavaScript and cannot be retrieved via static HTML scraping."

            return [], "Page loaded successfully, but zero review elements were found in Amazon HTML. Access may have been restricted or content requires dynamic JavaScript rendering."
        return reviews, None
    except Exception as e:
        return [], f"Extraction parsing error: {e}"


def _scrape_myntra_reviews(url):
    """Attempts to scrape reviews from a Myntra product page."""
    try:
        from bs4 import BeautifulSoup
        resp, err = safe_fetch_url(url, timeout=12)
        if not resp:
            return [], err or "Unable to connect to Myntra server."
        if resp.status_code == 403 or resp.status_code == 503:
            return [], f"Access blocked by Myntra anti-bot protections (HTTP {resp.status_code})."
        if resp.status_code != 200:
            return [], f"Myntra returned HTTP error status {resp.status_code}."

        soup = BeautifulSoup(resp.text, "html.parser")
        review_divs = soup.select("div.user-review-reviewTextWrapper, div.detailed-reviews-userReview")
        reviews = []
        for div in review_divs[:30]:
            review_text = div.get_text(separator=" ", strip=True)
            if len(review_text) < 15:
                continue
            reviews.append({
                "review_text": review_text,
                "rating": None,
                "author": "Myntra Customer",
                "date": None,
                "verified": None,
                "source": "Live Scraped - Myntra",
                "product_id": None,
                "product_url": url
            })
        if not reviews:
            return [], "Page loaded, but zero review elements were found. Myntra relies heavily on client-side React rendering."
        return reviews, None
    except Exception as e:
        return [], f"Extraction parsing error: {e}"


def _scrape_meesho_reviews(url):
    """Attempts to scrape reviews from a Meesho product page."""
    try:
        from bs4 import BeautifulSoup
        resp, err = safe_fetch_url(url, timeout=12)
        if not resp:
            return [], err or "Unable to connect to Meesho server."
        if resp.status_code == 403 or resp.status_code == 503:
            return [], f"Access blocked by Meesho anti-bot protections (HTTP {resp.status_code})."
        if resp.status_code != 200:
            return [], f"Meesho returned HTTP error status {resp.status_code}."

        soup = BeautifulSoup(resp.text, "html.parser")
        review_divs = soup.select("p.sc-eDvSVe, div[class*='review']")
        reviews = []
        for div in review_divs[:30]:
            review_text = div.get_text(separator=" ", strip=True)
            if len(review_text) < 15:
                continue
            reviews.append({
                "review_text": review_text,
                "rating": None,
                "author": "Meesho Customer",
                "date": None,
                "verified": None,
                "source": "Live Scraped - Meesho",
                "product_id": None,
                "product_url": url
            })
        if not reviews:
            return [], "Page loaded, but zero review elements were found in Meesho HTML."
        return reviews, None
    except Exception as e:
        return [], f"Extraction parsing error: {e}"


def _scrape_ajio_reviews(url):
    """Attempts to scrape reviews from an Ajio product page."""
    try:
        from bs4 import BeautifulSoup
        resp, err = safe_fetch_url(url, timeout=12)
        if not resp:
            return [], err or "Unable to connect to Ajio server."
        if resp.status_code == 403 or resp.status_code == 503:
            return [], f"Access blocked by Ajio anti-bot protections (HTTP {resp.status_code})."
        if resp.status_code != 200:
            return [], f"Ajio returned HTTP error status {resp.status_code}."

        soup = BeautifulSoup(resp.text, "html.parser")
        review_divs = soup.select("div.review-card, div.user-review, div[class*='review-item'], div[class*='ReviewCard']")
        reviews = []
        for div in review_divs[:30]:
            body = div.select_one("p.review-text, div.review-desc, p[class*='review']")
            review_text = body.get_text(separator=" ", strip=True) if body else div.get_text(separator=" ", strip=True)
            if len(review_text) < 15:
                continue
            reviews.append({
                "review_text": review_text,
                "rating": None,
                "author": "Ajio Customer",
                "date": None,
                "verified": None,
                "source": "Live Scraped - Ajio",
                "product_id": None,
                "product_url": url
            })
        if not reviews:
            return [], "Page loaded, but zero review elements were found. Ajio renders customer reviews dynamically via client-side JavaScript."
        return reviews, None
    except Exception as e:
        return [], f"Extraction parsing error: {e}"


def _scrape_croma_reviews(url):
    """Attempts to scrape reviews from a Croma product page."""
    try:
        from bs4 import BeautifulSoup
        resp, err = safe_fetch_url(url, timeout=12)
        if not resp:
            return [], err or "Unable to connect to Croma server."
        if resp.status_code == 403 or resp.status_code == 503:
            return [], f"Access blocked by Croma anti-bot protections (HTTP {resp.status_code})."
        if resp.status_code != 200:
            return [], f"Croma returned HTTP error status {resp.status_code}."

        soup = BeautifulSoup(resp.text, "html.parser")
        review_divs = soup.select("div.croma-review, div.bv-content-item, div[class*='review']")
        reviews = []
        for div in review_divs[:30]:
            review_text = div.get_text(separator=" ", strip=True)
            if len(review_text) < 15:
                continue
            reviews.append({
                "review_text": review_text,
                "rating": None,
                "author": "Croma Customer",
                "date": None,
                "verified": None,
                "source": "Live Scraped - Croma",
                "product_id": None,
                "product_url": url
            })
        if not reviews:
            return [], "Page loaded, but review widgets require client-side JavaScript rendering (PowerReviews/BazaarVoice)."
        return reviews, None
    except Exception as e:
        return [], f"Extraction parsing error: {e}"


def _scrape_reliance_digital_reviews(url):
    """Attempts to scrape reviews from a Reliance Digital product page."""
    try:
        from bs4 import BeautifulSoup
        resp, err = safe_fetch_url(url, timeout=12)
        if not resp:
            return [], err or "Unable to connect to Reliance Digital server."
        if resp.status_code == 403 or resp.status_code == 503:
            return [], f"Access blocked by Reliance Digital anti-bot protections (HTTP {resp.status_code})."
        if resp.status_code != 200:
            return [], f"Reliance Digital returned HTTP error status {resp.status_code}."

        soup = BeautifulSoup(resp.text, "html.parser")
        review_divs = soup.select("div.rd-review, div[class*='review']")
        reviews = []
        for div in review_divs[:30]:
            review_text = div.get_text(separator=" ", strip=True)
            if len(review_text) < 15:
                continue
            reviews.append({
                "review_text": review_text,
                "rating": None,
                "author": "Reliance Customer",
                "date": None,
                "verified": None,
                "source": "Live Scraped - Reliance Digital",
                "product_id": None,
                "product_url": url
            })
        if not reviews:
            return [], "Page loaded, but zero review elements were found. Content requires client-side dynamic JavaScript rendering."
        return reviews, None
    except Exception as e:
        return [], f"Extraction parsing error: {e}"


def _scrape_generic_reviews(url):
    """Generic scraper for e-commerce sites using common review CSS patterns."""
    try:
        from bs4 import BeautifulSoup
        resp, err = safe_fetch_url(url, timeout=12)
        if not resp:
            return [], err or "Unable to connect to e-commerce server."
        if resp.status_code == 403 or resp.status_code == 503:
            return [], f"Access blocked by e-commerce server anti-bot protections (HTTP {resp.status_code})."
        if resp.status_code != 200:
            return [], f"Server returned HTTP error status {resp.status_code}."

        soup = BeautifulSoup(resp.text, "html.parser")
        selectors = [
            "div.review", "div.user-review", "div[class*='review-text']",
            "div[class*='customer-review']", "p[class*='review']",
            "span[class*='review-body']"
        ]
        reviews = []
        for sel in selectors:
            divs = soup.select(sel)
            for div in divs[:30]:
                review_text = div.get_text(separator=" ", strip=True)
                if len(review_text) >= 15:
                    reviews.append({
                        "review_text": review_text,
                        "rating": None,
                        "author": "Customer",
                        "date": None,
                        "verified": None,
                        "source": "Live Scraped - Web",
                        "product_id": None,
                        "product_url": url
                    })
            if reviews:
                break
        if not reviews:
            return [], "Zero review elements matched common e-commerce HTML patterns."
        return reviews, None
    except Exception as e:
        return [], f"Extraction parsing error: {e}"

