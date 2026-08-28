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

def extract_product_name(url):
    """
    Parses an e-commerce URL (e.g. Amazon or Flipkart) and extracts
    a clean, human-readable product title.
    """
    if not isinstance(url, str):
        return "E-Commerce Product"
    
    url = url.strip()
    
    # 1. Amazon pattern: domain.com/Product-Name/dp/B0...
    amazon_match = re.search(r'amazon\.[a-z\.]+/([^/]+)/dp/', url, re.IGNORECASE)
    if amazon_match:
        name = amazon_match.group(1)
        name = name.replace('-', ' ').replace('_', ' ')
        return name.title()
        
    # 2. Flipkart pattern: flipkart.com/product-name/p/itm...
    flipkart_match = re.search(r'flipkart\.com/([^/]+)/p/', url, re.IGNORECASE)
    if flipkart_match:
        name = flipkart_match.group(1)
        name = name.replace('-', ' ').replace('_', ' ')
        return name.title()
        
    # 3. General URL fallback: parse last path segment
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

def fetch_product_reviews(url, product_name):
    """
    Returns a batch of reviews for the product.
    Includes realistic genuine and fake templates using the product's name.
    """
    import random
    
    # Dynamic review templates referencing the parsed product name
    reviews = []
    
    # 1. Genuine Positive reviews (Detailed, balanced)
    gen_pos = [
        f"I purchased this {product_name} last week. The design is beautiful and it functions exactly as described. The battery life is decent, though charging could be slightly faster. Highly recommended!",
        f"Excellent value for money. This {product_name} has exceeded my expectations in daily usage. Sturdy build and fast shipping.",
        f"Decent {product_name}. It has some minor flaws in the finish, but the performance is top-notch for the price.",
        f"Very happy with the purchase of this {product_name}. Customer service was very helpful when resolving my setup questions.",
        f"Honestly, this is a solid {product_name}. The build is premium and the UI is responsive. It is worth the price.",
        f"Good product. The packaging was neat, and it works perfectly. Have been using it for a couple of days.",
        f"The {product_name} arrived on time. It has good build quality and matches the specifications listed online.",
        f"I was skeptical about buying this {product_name} online, but it turned out to be very reliable and high quality."
    ]
    
    # 2. Genuine Negative reviews (Detailed critique, balanced tone)
    gen_neg = [
        f"The {product_name} arrived with a minor scratch on the frame. It still works, but I expected better packaging quality.",
        f"The performance of the {product_name} is okay, but the user interface feels slightly outdated. Decent but could be better.",
        f"Average product. The {product_name} works fine for basic needs, but is not suitable for heavy professional tasks.",
        f"I'm disappointed with the battery backup of this {product_name}. It barely lasts a few hours on a full charge."
    ]
    
    # 3. Fake Positive reviews (Hype, capitals, exclamations)
    fake_pos = [
        f"!!! BEST {product_name} EVER !!! AMAZING QUALITY !!! MUST BUY NOW !!! YOU WILL NOT REGRET IT !!!",
        f"OMG!!! Simply outstanding! This {product_name} is the best thing I have ever bought in my life! Five stars!!!",
        f"!!! UNBELIEVABLE QUALITY !!! Absolute perfection. Buy this {product_name} immediately, thank me later!!!",
        f"ABSOLUTELY PERFECT!!! 10/10 stars. Super fast delivery and extremely high quality {product_name}!!!",
        f"WOW! DO NOT HESITATE! Buy this {product_name} right now. I love it so much! PERFECT PERFECT!!!"
    ]
    
    # 4. Fake Negative reviews (Exaggerated hate, clickbait terms)
    fake_neg = [
        f"!!! COMPLETE SCAM !!! DO NOT BUY THIS {product_name} !!! WASTE OF MONEY AND TIME !!!",
        f"!!! WORST {product_name} EVER !!! BROKE IN ONE MINUTE !!! TRASH !!! RUN AWAY !!!",
        f"CRAP!!! Scammer seller. Do not trust the other reviews on this {product_name}!!! Total garbage!!!",
        f"!!! WARNING !!! This {product_name} is dangerous and stopped working immediately! REFUND MY MONEY!!!",
        f"TERRIBLE!!! Absolutely useless {product_name}. Zero stars. The seller refused to reply to my messages!"
    ]
    
    # Add reviews
    for r in gen_pos:
        reviews.append(r)
    for r in gen_neg:
        reviews.append(r)
    for r in fake_pos:
        reviews.append(r)
    for r in fake_neg:
        reviews.append(r)
        
    # Shuffle list to make the sequence look realistic
    random.shuffle(reviews)
    return reviews
