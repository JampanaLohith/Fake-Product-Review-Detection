import os
import json
import unittest
from unittest.mock import patch, MagicMock

# Set testing environment variables before importing app
os.environ["SECRET_KEY"] = "test-secret-key-12345"
os.environ["FLASK_DEBUG"] = "0"

from app import app, get_db_connection, init_db, load_ml_models
from model import (
    preprocess_text, extract_dense_features, explain_prediction,
    is_safe_url, extract_product_name, detect_platform, fetch_product_reviews
)

class FullSystemAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()
        init_db()
        cls.models_loaded = load_ml_models()

    def test_01_models_loaded(self):
        """Verify that ML models, vectorizer, and scaler load successfully."""
        self.assertTrue(self.models_loaded, "ML model files (model.pkl, vectorizer.pkl, scaler.pkl) must load.")

    def test_02_health_endpoint(self):
        """Verify /api/health returns 200 with service information."""
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data.get("status"), "ok")
        self.assertTrue(data.get("models_loaded"))

    def test_03_ssrf_prevention(self):
        """Verify that SSRF attacks targeting internal, loopback, or metadata addresses are blocked."""
        malicious_urls = [
            "http://127.0.0.1:5000/api/health",
            "http://localhost/admin",
            "http://169.254.169.254/latest/meta-data/",
            "http://192.168.1.1/router",
            "http://10.0.0.1/internal",
            "http://0.0.0.0:8000/",
            "ftp://example.com/file",
            "file:///etc/passwd",
            "http://metadata.google.internal/computeMetadata/v1/"
        ]
        for bad_url in malicious_urls:
            is_safe, msg = is_safe_url(bad_url)
            self.assertFalse(is_safe, f"SSRF URL '{bad_url}' should be blocked, but was allowed. Msg: {msg}")

    def test_04_public_url_validation(self):
        """Verify that valid public e-commerce URLs are allowed by is_safe_url."""
        public_urls = [
            "https://www.amazon.in/dp/B0CHX1W1Y2",
            "https://www.flipkart.com/apple-iphone-15-black-128-gb/p/itm2d82914"
        ]
        for url in public_urls:
            is_safe, msg = is_safe_url(url)
            self.assertTrue(is_safe, f"Public URL '{url}' should be allowed. Msg: {msg}")

    def test_05_predict_single_review_genuine(self):
        """Verify single review prediction for genuine customer feedback."""
        review = (
            "I purchased this laptop three weeks ago for work. The battery lasts about 7 hours. "
            "The keyboard is comfortable and performance has been very reliable so far."
        )
        resp = self.client.post("/predict", data={"review": review}, follow_redirects=True)
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"GENUINE REVIEW", resp.data)

    def test_06_predict_single_review_fake(self):
        """Verify single review prediction for extreme promotional hype spam."""
        review = (
            "!!! THE BEST HEADPHONES IN THE ENTIRE WORLD AMAZING QUALITY MUST BUY NOW "
            "WOW WOW 10/10 STARS DO NOT HESITATE BUY BUY BUY !!!"
        )
        resp = self.client.post("/predict", data={"review": review}, follow_redirects=True)
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"SUSPECTED FAKE", resp.data)

    def test_07_predict_validation_short_and_empty(self):
        """Verify input validation handles empty and short reviews gracefully."""
        # Empty
        resp_empty = self.client.post("/predict", data={"review": ""}, follow_redirects=True)
        self.assertEqual(resp_empty.status_code, 200)
        self.assertIn(b"Please enter a review to analyze", resp_empty.data)

        # Too short (< 15 chars)
        resp_short = self.client.post("/predict", data={"review": "Too short"}, follow_redirects=True)
        self.assertEqual(resp_short.status_code, 200)
        self.assertIn(b"The review is too short", resp_short.data)

    def test_08_predict_product_ssrf_blocked(self):
        """Verify that submitting an SSRF URL in /predict_product is rejected with a warning."""
        resp = self.client.post("/predict_product", data={"product_url": "http://127.0.0.1:5000/"}, follow_redirects=True)
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Invalid or restricted URL", resp.data)

    def test_09_predict_product_scraping_blocked_truthful_response(self):
        """
        Verify that when third-party scraping fails or is blocked, no synthetic fake
        reviews are fabricated. The system must report the failure truthfully.
        """
        with patch("model.safe_fetch_url", return_value=(None, "Anti-bot 403 Forbidden")):
            resp = self.client.post("/predict_product", data={"product_url": "https://www.amazon.in/dp/B0CHX1W1Y2"}, follow_redirects=True)
            self.assertEqual(resp.status_code, 200)
            self.assertIn(b"Unable to extract customer reviews", resp.data)

    def test_10_predict_product_with_real_reviews(self):
        """
        Verify product review analysis workflow when real reviews are extracted.
        Mock external network response to ensure reproducible testing.
        """
        mock_reviews = [
            {"review_text": "I purchased this phone last month. Great battery and screen clarity.", "rating": 5.0, "author": "Alice", "verified": True},
            {"review_text": "Solid product for everyday use. Good camera and fast charging.", "rating": 4.5, "author": "Bob", "verified": True},
            {"review_text": "Works as expected. Packaging was neat and delivered on time.", "rating": 4.0, "author": "Charlie", "verified": True},
            {"review_text": "Average device. Lags slightly when playing heavy games.", "rating": 3.0, "author": "David", "verified": True},
            {"review_text": "!!! BEST PHONE EVER AMAZING QUALITY MUST BUY IMMEDIATELY 10/10 STARS !!!", "rating": 5.0, "author": "Spammer", "verified": False}
        ]
        with patch("app.fetch_product_reviews", return_value=(mock_reviews, "Amazon", None)):
            resp = self.client.post("/predict_product", data={"product_url": "https://www.amazon.in/dp/B0CHX1W1Y2"}, follow_redirects=True)
            self.assertEqual(resp.status_code, 200)
            self.assertIn(b"Authenticity Score", resp.data)
            self.assertIn(b"Actual Reviews Fetched", resp.data)

    def test_11_api_analyze_valid_and_invalid(self):
        """Verify /api/analyze JSON API handling."""
        # Valid
        resp_valid = self.client.post("/api/analyze", json={
            "review": "I have been using this laptop for over a month and it works great."
        })
        self.assertEqual(resp_valid.status_code, 200)
        data = resp_valid.get_json()
        self.assertIn("prediction", data)
        self.assertIn("confidence_pct", data)
        self.assertIn("prob_genuine", data)
        self.assertIn("prob_fake", data)

        # Non-dict JSON payload
        resp_bad_type = self.client.post("/api/analyze", json=["not", "a", "dict"])
        self.assertEqual(resp_bad_type.status_code, 400)

        # Missing review field
        resp_missing = self.client.post("/api/analyze", json={"other": "field"})
        self.assertEqual(resp_missing.status_code, 400)

        # Text too short
        resp_too_short = self.client.post("/api/analyze", json={"review": "Short"})
        self.assertEqual(resp_too_short.status_code, 400)

    def test_12_dashboard_and_report_routes(self):
        """Verify dashboard and report pages render without errors."""
        resp_dash = self.client.get("/dashboard")
        self.assertEqual(resp_dash.status_code, 200)
        self.assertIn(b"Performance & Analytics Dashboard", resp_dash.data)

        resp_rep = self.client.get("/report")
        self.assertEqual(resp_rep.status_code, 200)
        self.assertIn(b"Academic Project Report", resp_rep.data)

    def test_13_clear_history_api(self):
        """Verify /api/clear_history endpoint executes successfully."""
        resp = self.client.post("/api/clear_history")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data.get("status"), "success")

if __name__ == "__main__":
    unittest.main()
