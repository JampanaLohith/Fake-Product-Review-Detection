import os
import json
import sqlite3
import joblib
import numpy as np
import scipy.sparse
from flask import Flask, render_template, request, redirect, url_for, jsonify, flash

# Import our helper functions
from model import preprocess_text, extract_dense_features, explain_prediction

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
    conn.commit()
    conn.close()

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

@app.route('/dashboard')
def dashboard():
    global model_metadata
    # Ensure models/metadata are loaded
    if not model_metadata:
        load_ml_models()
        
    conn = get_db_connection()
    
    # 1. Total counts from SQLite
    stats = conn.execute('''
        SELECT 
            COUNT(*) as total_reviews,
            SUM(CASE WHEN prediction_label = 'Fake' THEN 1 ELSE 0 END) as fake_count,
            SUM(CASE WHEN prediction_label = 'Genuine' THEN 1 ELSE 0 END) as genuine_count
        FROM predictions
    ''').fetchone()
    
    total_reviews = stats['total_reviews'] or 0
    fake_count = stats['fake_count'] or 0
    genuine_count = stats['genuine_count'] or 0
    
    # 2. Get last 10 predictions
    history_rows = conn.execute('''
        SELECT id, review_text, prediction_label, confidence, created_at
        FROM predictions
        ORDER BY created_at DESC
        LIMIT 10
    ''').fetchall()
    
    # 3. Last prediction info
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
        accuracy=accuracy_pct,
        model_name=model_name,
        history=history_rows,
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
        conn.commit()
        conn.close()
        return jsonify({"status": "success", "message": "History cleared successfully."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

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
