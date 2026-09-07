import os
import json
import sqlite3
import joblib
import numpy as np
import scipy.sparse
from flask import Flask, render_template, request, redirect, url_for, jsonify, flash

# Import our helper functions
from model import preprocess_text, extract_dense_features, explain_prediction, extract_product_name, fetch_product_reviews, detect_platform

app = Flask(__name__)
app.secret_key = "fake_review_secret_key_for_flash"

# Database path
DB_PATH = "database.db"

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the database schema if it doesn't exist."""
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            review_text TEXT NOT NULL,
            prediction_label TEXT NOT NULL,
            confidence REAL NOT NULL,
            prob_fake REAL NOT NULL,
            prob_genuine REAL NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS product_analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_name TEXT NOT NULL,
            product_url TEXT NOT NULL,
            platform_name TEXT DEFAULT 'E-Commerce',
            total_reviews INTEGER NOT NULL,
            fake_count INTEGER NOT NULL,
            genuine_count INTEGER NOT NULL,
            trust_score REAL NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS product_reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id INTEGER NOT NULL,
            review_text TEXT NOT NULL,
            prediction_label TEXT NOT NULL,
            confidence REAL NOT NULL,
            rating REAL,
            author TEXT,
            date TEXT,
            verified INTEGER DEFAULT 0,
            source TEXT,
            FOREIGN KEY (analysis_id) REFERENCES product_analyses(id) ON DELETE CASCADE
        )
    ''')
    # Schema migration: add columns that may be missing in older databases
    _migrate_db(conn)
    conn.commit()
    conn.close()

def _migrate_db(conn):
    """Adds any missing columns to existing tables without data loss."""
    migrations = [
        ("product_analyses", "platform_name", "TEXT DEFAULT 'E-Commerce'"),
        ("product_reviews",  "rating",        "REAL"),
        ("product_reviews",  "author",        "TEXT"),
        ("product_reviews",  "date",          "TEXT"),
        ("product_reviews",  "verified",      "INTEGER DEFAULT 0"),
        ("product_reviews",  "source",        "TEXT"),
    ]
    for table, column, col_def in migrations:
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_def}")
        except Exception:
            pass  # Column already exists — safe to ignore


# Initialize SQLite database
init_db()

# Global variables for models
model = None
vectorizer = None
scaler = None
model_metadata = {}

def load_ml_models():
    """Loads pickles and metadata, returns True if successful, False otherwise."""
    global model, vectorizer, scaler, model_metadata
    try:
        if (os.path.exists("model.pkl") and 
            os.path.exists("vectorizer.pkl") and 
            os.path.exists("scaler.pkl")):
            
            model = joblib.load("model.pkl")
            vectorizer = joblib.load("vectorizer.pkl")
            scaler = joblib.load("scaler.pkl")
            
            if os.path.exists("model_metadata.json"):
                with open("model_metadata.json", "r") as f:
                    model_metadata = json.load(f)
            else:
                model_metadata = {
                    "best_model_name": "Logistic Regression",
                    "logistic_regression": {"accuracy": 0.95},
                    "random_forest": {"accuracy": 0.93}
                }
            return True
        return False
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

# Initial loading attempt
models_loaded = load_ml_models()

@app.context_processor
def inject_models_status():
    """Injects model status dynamically into templates."""
    return dict(models_loaded=models_loaded)

@app.route('/')
def index():
    # Make sure models are loaded
    global models_loaded
    if not models_loaded:
        models_loaded = load_ml_models()
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global model, vectorizer, scaler, models_loaded
    if not models_loaded:
        models_loaded = load_ml_models()
        if not models_loaded:
            flash("Model files are not found or failed to load. Please run 'python train_model.py' first.", "danger")
            return redirect(url_for('index'))

    review_text = request.form.get('review', '').strip()
    
    # Validation
    if not review_text:
        flash("Please enter a review to analyze.", "warning")
        return redirect(url_for('index'))
    
    if len(review_text) < 15:
        flash("The review is too short for a reliable prediction. Please write at least 15 characters.", "warning")
        return redirect(url_for('index'))

    try:
        # Preprocess text
        cleaned_text = preprocess_text(review_text)
        
        # Extract features
        dense_features = extract_dense_features(review_text)
        dense_scaled = scaler.transform(dense_features.reshape(1, -1))
        
        # TF-IDF Vectorization
        tfidf_feat = vectorizer.transform([cleaned_text])
        
        # Combine features
        X_combined = scipy.sparse.hstack([tfidf_feat, dense_scaled])
        
        # Predict probability
        prob = model.predict_proba(X_combined)[0] # [P(0=Genuine), P(1=Fake)]
        prob_genuine = prob[0]
        prob_fake = prob[1]
        
        # Get label
        prediction_label = "Fake" if prob_fake >= 0.50 else "Genuine"
        confidence = prob_fake if prediction_label == "Fake" else prob_genuine
        confidence_pct = round(confidence * 100, 2)
        
        # Log to Database
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions (review_text, prediction_label, confidence, prob_fake, prob_genuine)
            VALUES (?, ?, ?, ?, ?)
        ''', (review_text, prediction_label, confidence_pct, float(prob_fake), float(prob_genuine)))
        conn.commit()
        prediction_id = cursor.lastrowid
        conn.close()
        
        return redirect(url_for('result', prediction_id=prediction_id))
        
    except Exception as e:
        flash(f"An error occurred during prediction: {e}", "danger")
        return redirect(url_for('index'))

@app.route('/result/<int:prediction_id>')
def result(prediction_id):
    global model, vectorizer, scaler, models_loaded
    if not models_loaded:
        models_loaded = load_ml_models()
        if not models_loaded:
            flash("ML Models are not loaded.", "danger")
            return redirect(url_for('index'))
            
    conn = get_db_connection()
    row = conn.execute('SELECT * FROM predictions WHERE id = ?', (prediction_id,)).fetchone()
    conn.close()
    
    if not row:
        flash("Prediction report not found.", "danger")
        return redirect(url_for('index'))
        
    # Generate Explanation
    review_text = row['review_text']
    explanation = explain_prediction(review_text, model, vectorizer, scaler)
    
    return render_template('result.html', prediction=row, explanation=explanation)

@app.route('/predict_product', methods=['POST'])
def predict_product():
    global model, vectorizer, scaler, models_loaded
    if not models_loaded:
        models_loaded = load_ml_models()
        if not models_loaded:
            flash("Model files are not found or failed to load. Please run 'python train_model.py' first.", "danger")
            return redirect(url_for('index'))

    product_url = request.form.get('product_url', '').strip()
    
    # Validation
    if not product_url:
        flash("Please enter a product URL to analyze.", "warning")
        return redirect(url_for('index'))
    
    if not (product_url.startswith("http://") or product_url.startswith("https://")):
        flash("Please enter a valid product URL starting with http:// or https://", "warning")
        return redirect(url_for('index'))

    try:
        product_name = extract_product_name(product_url)
        reviews, platform_name, error_msg = fetch_product_reviews(product_url, product_name)
        
        # Strict Rule: No synthetic fallback reviews when 0 reviews extracted
        if not reviews or len(reviews) == 0:
            flash(error_msg or f"Unable to fetch reviews from this {platform_name} product page.", "warning")
            return redirect(url_for('index'))
            
        fake_count = 0
        genuine_count = 0
        predictions_batch = []
        
        for r_item in reviews:
            r_text = r_item['review_text']
            cleaned_text = preprocess_text(r_text)
            dense_features = extract_dense_features(r_text)
            dense_scaled = scaler.transform(dense_features.reshape(1, -1))
            tfidf_feat = vectorizer.transform([cleaned_text])
            X_combined = scipy.sparse.hstack([tfidf_feat, dense_scaled])
            
            prob = model.predict_proba(X_combined)[0]
            prob_genuine = prob[0]
            prob_fake = prob[1]
            
            prediction_label = "Fake" if prob_fake >= 0.50 else "Genuine"
            confidence = prob_fake if prediction_label == "Fake" else prob_genuine
            confidence_pct = round(confidence * 100, 2)
            
            if prediction_label == "Fake":
                fake_count += 1
            else:
                genuine_count += 1
                
            predictions_batch.append({
                "review_text": r_text,
                "prediction_label": prediction_label,
                "confidence": confidence_pct,
                "rating": r_item.get("rating"),
                "author": r_item.get("author"),
                "date": r_item.get("date"),
                "verified": 1 if r_item.get("verified") else (0 if r_item.get("verified") is False else None),
                "source": r_item.get("source", f"Live Scraped - {platform_name}")
            })
            
        total_reviews = len(reviews)
        trust_score = round((genuine_count / total_reviews) * 100, 2)
        
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO product_analyses (product_name, product_url, platform_name, total_reviews, fake_count, genuine_count, trust_score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (product_name, product_url, platform_name, total_reviews, fake_count, genuine_count, trust_score))
        analysis_id = cursor.lastrowid
        
        for pred in predictions_batch:
            cursor.execute('''
                INSERT INTO product_reviews (analysis_id, review_text, prediction_label, confidence, rating, author, date, verified, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (analysis_id, pred["review_text"], pred["prediction_label"], pred["confidence"], pred["rating"], pred["author"], pred["date"], pred["verified"], pred["source"]))
            
        conn.commit()
        conn.close()
        
        return redirect(url_for('product_result', analysis_id=analysis_id))
        
    except Exception as e:
        flash(f"An error occurred during product review analysis: {e}", "danger")
        return redirect(url_for('index'))

@app.route('/product_result/<int:analysis_id>')
def product_result(analysis_id):
    conn = get_db_connection()
    analysis = conn.execute('SELECT * FROM product_analyses WHERE id = ?', (analysis_id,)).fetchone()
    
    if not analysis:
        conn.close()
        flash("Product analysis report not found.", "danger")
        return redirect(url_for('index'))
        
    reviews = conn.execute('SELECT * FROM product_reviews WHERE analysis_id = ?', (analysis_id,)).fetchall()
    conn.close()
    
    fake_reviews = [r for r in reviews if r['prediction_label'] == 'Fake']
    genuine_reviews = [r for r in reviews if r['prediction_label'] == 'Genuine']
    
    total = analysis['total_reviews']
    fake_pct = round((analysis['fake_count'] / total) * 100, 2) if total > 0 else 0.0
    
    return render_template(
        'product_result.html',
        analysis=analysis,
        reviews=reviews,
        fake_reviews=fake_reviews,
        genuine_reviews=genuine_reviews,
        fake_pct=fake_pct
    )

@app.route('/dashboard')
def dashboard():
    global model_metadata
    # Ensure models/metadata are loaded
    if not model_metadata:
        load_ml_models()
        
    conn = get_db_connection()
    
    # 1. Total counts from predictions (single reviews)
    stats = conn.execute('''
        SELECT 
            COUNT(*) as total_reviews,
            SUM(CASE WHEN prediction_label = 'Fake' THEN 1 ELSE 0 END) as fake_count,
            SUM(CASE WHEN prediction_label = 'Genuine' THEN 1 ELSE 0 END) as genuine_count
        FROM predictions
    ''').fetchone()
    
    single_total = stats['total_reviews'] or 0
    single_fake = stats['fake_count'] or 0
    single_genuine = stats['genuine_count'] or 0
    
    # 2. Total counts from product_analyses
    prod_stats = conn.execute('''
        SELECT 
            COUNT(*) as total_products,
            SUM(total_reviews) as total_prod_reviews,
            SUM(fake_count) as total_prod_fake,
            SUM(genuine_count) as total_prod_genuine,
            AVG(trust_score) as avg_trust
        FROM product_analyses
    ''').fetchone()
    
    total_products = prod_stats['total_products'] or 0
    prod_reviews = prod_stats['total_prod_reviews'] or 0
    prod_fake = prod_stats['total_prod_fake'] or 0
    prod_genuine = prod_stats['total_prod_genuine'] or 0
    avg_trust_score = round(prod_stats['avg_trust'], 2) if prod_stats['avg_trust'] is not None else 0.0
    
    # Combined Totals
    total_reviews = single_total + prod_reviews
    fake_count = single_fake + prod_fake
    genuine_count = single_genuine + prod_genuine
    
    # 3. Get last 10 predictions (single)
    history_rows = conn.execute('''
        SELECT id, review_text, prediction_label, confidence, created_at
        FROM predictions
        ORDER BY created_at DESC
        LIMIT 10
    ''').fetchall()
    
    # 4. Get last 5 product analyses
    product_history = conn.execute('''
        SELECT id, product_name, product_url, total_reviews, fake_count, genuine_count, trust_score, created_at
        FROM product_analyses
        ORDER BY created_at DESC
        LIMIT 5
    ''').fetchall()
    
    # 5. Last prediction info (for single review)
    last_pred = conn.execute('''
        SELECT review_text, prediction_label, confidence, created_at
        FROM predictions
        ORDER BY created_at DESC
        LIMIT 1
    ''').fetchone()
    
    conn.close()
    
    # Setup accuracy/model name
    model_name = model_metadata.get('best_model_name', 'Logistic Regression')
    
    # Extract training metrics
    lr_metrics = model_metadata.get('logistic_regression', {})
    rf_metrics = model_metadata.get('random_forest', {})
    
    accuracy = lr_metrics.get('accuracy', 0.95) if model_name == "Logistic Regression" else rf_metrics.get('accuracy', 0.93)
    accuracy_pct = round(accuracy * 100, 2)
    
    return render_template(
        'dashboard.html',
        total_reviews=total_reviews,
        fake_count=fake_count,
        genuine_count=genuine_count,
        total_products=total_products,
        avg_trust_score=avg_trust_score,
        accuracy=accuracy_pct,
        model_name=model_name,
        history=history_rows,
        product_history=product_history,
        last_prediction=last_pred,
        lr_metrics=lr_metrics,
        rf_metrics=rf_metrics,
        dataset_stats=model_metadata.get('dataset_stats', {})
    )

@app.route('/api/clear_history', methods=['POST'])
def clear_history():
    try:
        conn = get_db_connection()
        conn.execute('DELETE FROM predictions')
        conn.execute('DELETE FROM product_analyses')
        conn.execute('DELETE FROM product_reviews')
        conn.commit()
        conn.close()
        return jsonify({"status": "success", "message": "All history logs cleared successfully."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/report')
def report():
    return render_template('report.html')

@app.route('/api/health')
def health():
    """Health-check endpoint for Render and monitoring tools."""
    return jsonify({
        "status": "ok",
        "service": "VeriTrust AI – Fake Review Detection",
        "models_loaded": models_loaded,
        "version": "2.0"
    })

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """
    JSON API endpoint: accepts { "review": "..." } and returns prediction.
    Useful for integration testing and interview demonstrations.
    """
    global model, vectorizer, scaler, models_loaded
    if not models_loaded:
        models_loaded = load_ml_models()
        if not models_loaded:
            return jsonify({"error": "ML models not loaded. Run train_model.py first."}), 503

    data = request.get_json(force=True, silent=True)
    if not data or 'review' not in data:
        return jsonify({"error": "Provide JSON body: {\"review\": \"<text>\"}"}), 400

    review_text = str(data['review']).strip()
    if len(review_text) < 15:
        return jsonify({"error": "Review text is too short (minimum 15 characters)."}), 400

    try:
        cleaned_text = preprocess_text(review_text)
        dense_features = extract_dense_features(review_text)
        dense_scaled = scaler.transform(dense_features.reshape(1, -1))
        tfidf_feat = vectorizer.transform([cleaned_text])
        X_combined = scipy.sparse.hstack([tfidf_feat, dense_scaled])
        prob = model.predict_proba(X_combined)[0]
        prob_genuine = float(prob[0])
        prob_fake = float(prob[1])
        prediction_label = "Fake" if prob_fake >= 0.50 else "Genuine"
        confidence = prob_fake if prediction_label == "Fake" else prob_genuine
        return jsonify({
            "prediction": prediction_label,
            "confidence_pct": round(confidence * 100, 2),
            "prob_fake": round(prob_fake, 4),
            "prob_genuine": round(prob_genuine, 4),
            "disclaimer": "ML predictions are based on patterns learned from training data and are not absolute proof of review authenticity."
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Initialize DB and print network access URLs
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        print("\n" + "="*70)
        print(f" * Localhost URL:   http://127.0.0.1:5000")
        print(f" * LAN Network URL:  http://{local_ip}:5000")
        print(" (Use the LAN Network URL to open this app on other devices on the same Wi-Fi)")
        print("="*70 + "\n")
    except Exception:
        pass
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
