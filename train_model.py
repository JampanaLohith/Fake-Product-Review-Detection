import os
import csv
import json
import random
import joblib
import pandas as pd
import numpy as np
import scipy.sparse
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Import our preprocessing & feature extraction helpers
from model import preprocess_text, extract_dense_features

# 1. Dataset Generation Helper (generates CSV if not present)
def generate_sample_dataset(filepath="dataset.csv"):
    if os.path.exists(filepath):
        print(f"Dataset already exists at {filepath}.")
        return

    print("Generating sample dataset...")

    products = ["phone", "laptop", "smartwatch", "wireless headphones", "coffee maker", 
                "vacuum cleaner", "running shoes", "backpack", "power bank", "electric kettle"]
    
    times = ["last week", "a month ago", "yesterday", "three days ago", "two months ago", "recently"]
    
    features = {
        "phone": ["battery life", "screen brightness", "camera quality", "face unlock speed"],
        "laptop": ["keyboard response", "boot time", "screen resolution", "cooling fan sound"],
        "smartwatch": ["step tracking", "heart rate monitor", "sleep tracking", "watch strap comfort"],
        "wireless headphones": ["sound clarity", "noise cancellation", "bass depth", "earbud fit"],
        "coffee maker": ["brewing speed", "water tank capacity", "carafe design", "temperature control"],
        "vacuum cleaner": ["suction power", "cord length", "weight", "dustbin capacity"],
        "running shoes": ["arch support", "sole grip", "material breathability", "cushion thickness"],
        "backpack": ["zipper quality", "shoulder pad comfort", "waterproof lining", "laptop sleeve spacing"],
        "power bank": ["charging speed", "port durability", "LED display accuracy", "capacity retention"],
        "electric kettle": ["boiling speed", "handle grip", "auto-shutoff feature", "spout design"]
    }

    positive_adjs = ["excellent", "quite good", "decent", "impressive", "reliable", "satisfying", "outstanding", "great"]
    negative_adjs = ["mediocre", "disappointing", "subpar", "faulty", "flimsy", "average", "defective", "bad"]

    genuine_reviews = []
    fake_reviews = []

    # Generate Genuine Positive reviews (balanced, detailed, normal casing, few/no exclamation marks)
    for p in products:
        for t in times:
            for feat in features[p]:
                adj = random.choice(positive_adjs)
                review = (
                    f"I purchased this {p} {t}. The {feat} is {adj}. "
                    f"It has been running smoothly without any major issues. "
                    f"Overall, it is a decent value for money, though shipping took an extra day. "
                    f"I would recommend this to anyone looking for a reliable {p}."
                )
                genuine_reviews.append((review, 0))

    # Generate Genuine Negative reviews (detailed critiques, balanced tone, normal casing)
    for p in products:
        for t in times:
            for feat in features[p]:
                adj = random.choice(negative_adjs)
                review = (
                    f"I received the {p} {t} but I am not entirely satisfied. "
                    f"The {feat} seems {adj} and didn't meet my expectations. "
                    f"Additionally, the build quality feels a bit plastic-like. "
                    f"It works okay for basic stuff, but I wouldn't recommend it if you need high performance."
                )
                genuine_reviews.append((review, 0))

    # Generate Fake Positive reviews (extreme hype, caps, exclamation marks, short repetitive sentences)
    fake_pos_phrases = [
        "AMAZING PRODUCT!!! THE BEST THING EVER!!! MUST BUY NOW!!!",
        "WOW!!! Simply outstanding! Best purchase I've ever made in my entire life!",
        "UNBELIEVABLE QUALITY!!! Exceeded all expectations!!! Buy it right now, thank me later!!!",
        "ABSOLUTELY PERFECT!!! 10/10 stars!!! Extremely fast shipping and wonderful product!!!",
        "OMG!!! The seller is an angel! The best customer service! Strongly recommended!!!",
        "JUST WOW! DO NOT HESITATE! Buy this product immediately! I love it so much!"
    ]
    for _ in range(120):
        p = random.choice(products)
        phrase = random.choice(fake_pos_phrases)
        review = f"{phrase} This {p} is a lifesaver! I bought 5 more for my family members. PERFECT!!!"
        fake_reviews.append((review, 1))

    # Generate Fake Negative reviews (extreme hate, caps, exclamation marks, calling it a scam, refund demands)
    fake_neg_phrases = [
        "COMPLETE SCAM!!! DO NOT BUY!!! WASTE OF MONEY AND TIME!!!",
        "WORST PRODUCT EVER!!! BROKE IN ONE MINUTE!!! TRASH!!!",
        "CRAP!!! SCAMMER SELLER!!! RUN AWAY!!! DO NOT TRUST THIS PRODUCT!!!",
        "TERRIBLE!!! Total garbage. Zero stars if possible. Refund my money immediately!!!",
        "WARNING!!! FAKE REVIEWS HERE!!! This item is dangerous and broke immediately! DON'T BUY!"
    ]
    for _ in range(120):
        p = random.choice(products)
        phrase = random.choice(fake_neg_phrases)
        review = f"{phrase} I hate this {p}. The customer support refused to reply. AVOID AT ALL COSTS!!!"
        fake_reviews.append((review, 1))

    # Combine and shuffle
    all_data = genuine_reviews + fake_reviews
    random.shuffle(all_data)

    # Write to CSV
    with open(filepath, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["review", "label"])
        writer.writerows(all_data)

    print(f"Created dataset with {len(all_data)} records (Genuine: {len(genuine_reviews)}, Fake: {len(fake_reviews)}) at {filepath}.")

def main():
    # 1. Ensure dataset exists
    generate_sample_dataset()
    
    # Load dataset
    df = pd.read_csv("dataset.csv")
    
    # Drop duplicates and missing values
    df.dropna(subset=['review'], inplace=True)
    df.drop_duplicates(subset=['review'], inplace=True)
    
    print(f"Dataset shape after cleaning: {df.shape}")
    print(f"Class breakdown:\n{df['label'].value_counts()}")
    
    # 2. Extract Features
    print("Preprocessing text reviews and extracting dense features...")
    
    # Text cleaning
    df['cleaned_review'] = df['review'].apply(preprocess_text)
    
    # Dense hand-crafted features
    dense_list = []
    for r in df['review']:
        dense_list.append(extract_dense_features(r))
    X_dense = np.vstack(dense_list)
    
    # 3. Fit vectorizer and scaler on the entire dataset first (to define dimensions)
    # TF-IDF Vectorizer (unigrams + bigrams)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    X_tfidf_all = vectorizer.fit_transform(df['cleaned_review'])
    
    # Scaler for dense features
    scaler = StandardScaler()
    X_dense_scaled_all = scaler.fit_transform(X_dense)
    
    # Combine TF-IDF and dense features
    X_combined_all = scipy.sparse.hstack([X_tfidf_all, X_dense_scaled_all])
    y_all = df['label'].values
    
    # 4. Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined_all, y_all, test_size=0.20, random_state=42, stratify=y_all
    )
    
    # 5. Train & Evaluate Models
    print("Training Logistic Regression model...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    
    # Logistic Regression Evaluation
    y_pred_lr = lr_model.predict(X_test)
    y_prob_lr = lr_model.predict_proba(X_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    prec_lr, rec_lr, f1_lr, _ = precision_recall_fscore_support(y_test, y_pred_lr, average='binary')
    cm_lr = confusion_matrix(y_test, y_pred_lr).tolist() # [[TN, FP], [FN, TP]]
    
    print(f"Logistic Regression Accuracy: {acc_lr:.4f}")
    
    print("Training Random Forest model (for comparison)...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Random Forest Evaluation
    y_pred_rf = rf_model.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    prec_rf, rec_rf, f1_rf, _ = precision_recall_fscore_support(y_test, y_pred_rf, average='binary')
    cm_rf = confusion_matrix(y_test, y_pred_rf).tolist()
    
    print(f"Random Forest Accuracy: {acc_rf:.4f}")
    
    # 6. Select Best Model
    best_model_name = "Logistic Regression" if acc_lr >= acc_rf else "Random Forest"
    best_model = lr_model if acc_lr >= acc_rf else rf_model
    print(f"Best model selected: {best_model_name}")
    
    # Cross Validation on Logistic Regression
    cv_scores = cross_val_score(lr_model, X_combined_all, y_all, cv=5)
    cv_mean = cv_scores.mean()
    print(f"5-Fold Cross Validation Accuracy (Logistic Regression): {cv_mean:.4f} (+/- {cv_scores.std():.4f})")
    
    # 7. Save Models and Metadata
    print("Saving pickles to disk...")
    # We save Logistic Regression specifically as model.pkl because it offers explainability coefficients
    # and has high accuracy on this clean template dataset.
    joblib.dump(lr_model, "model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")
    joblib.dump(scaler, "scaler.pkl")
    
    # Save training metadata for the web dashboard display
    metadata = {
        "best_model_name": best_model_name,
        "logistic_regression": {
            "accuracy": float(acc_lr),
            "precision": float(prec_lr),
            "recall": float(rec_lr),
            "f1_score": float(f1_lr),
            "confusion_matrix": cm_lr,
            "cv_mean": float(cv_mean)
        },
        "random_forest": {
            "accuracy": float(acc_rf),
            "precision": float(prec_rf),
            "recall": float(rec_rf),
            "f1_score": float(f1_rf),
            "confusion_matrix": cm_rf
        },
        "dataset_stats": {
            "total_reviews": int(df.shape[0]),
            "genuine_count": int((df['label'] == 0).sum()),
            "fake_count": int((df['label'] == 1).sum())
        }
    }
    
    with open("model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    print("Training finished successfully. Saved pickles: model.pkl, vectorizer.pkl, scaler.pkl, and metadata: model_metadata.json.")

if __name__ == "__main__":
    main()
