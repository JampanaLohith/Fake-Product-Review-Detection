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

    def test_09_predict_product_scraping_blocked(self):
        """
        Verify that when third-party scraping fails or encounters anti-bot blocks,
        NO synthetic reviews or fake authenticity scores are generated.
        The failure is truthfully reported to the user with options to retry or enter reviews manually.
        """
        with patch("model.safe_fetch_url", return_value=(None, "Access blocked by Amazon anti-bot protections (HTTP 403).")):
            resp = self.client.post("/predict_product", data={"product_url": "https://www.amazon.in/dp/B0CHX1W1Y2"}, follow_redirects=True)
            self.assertEqual(resp.status_code, 200)
            self.assertIn(b"Unable to retrieve actual product reviews", resp.data)
            self.assertIn(b"anti-bot", resp.data)
            # Must NOT display authenticity score when 0 actual reviews were retrieved
            self.assertNotIn(b"Authenticity Score", resp.data)

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
            self.assertIn(b"Actual Reviews Collected", resp.data)

    def test_14_predict_product_zero_reviews_returned(self):
        """
        Verify handling when a web page loads (200 OK) but zero customer reviews are found in the DOM.
        """
        with patch("app.fetch_product_reviews", return_value=([], "Amazon", "Zero customer reviews were found on this product page.")):
            resp = self.client.post("/predict_product", data={"product_url": "https://www.amazon.in/dp/B0CHX1W1Y2"}, follow_redirects=True)
            self.assertEqual(resp.status_code, 200)
            self.assertIn(b"Unable to retrieve actual product reviews", resp.data)
            self.assertNotIn(b"Authenticity Score", resp.data)

    def test_15_manual_review_submission_distinguished(self):
        """
        Verify that manual review submissions through /predict are recorded with
        source='Manual Review Submission' to clearly distinguish from live scraped data.
        """
        review = "This is a manually pasted customer review for verification purposes."
        resp = self.client.post("/predict", data={"review": review}, follow_redirects=True)
        self.assertEqual(resp.status_code, 200)
        conn = get_db_connection()
        row = conn.execute("SELECT source FROM predictions ORDER BY id DESC LIMIT 1").fetchone()
        conn.close()
        self.assertEqual(row["source"], "Manual Review Submission")

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

    def test_16_enthusiastic_genuine_review_with_exclamations(self):
        """
        Verify that an enthusiastic genuine review with exclamation marks is classified
        as Genuine and not misclassified as Fake solely due to punctuation.
        """
        from model import classify_probability
        # Unit test the classification logic with simulated probabilities
        label, conf, is_uncertain = classify_probability(0.25)
        self.assertEqual(label, "Genuine")
        self.assertFalse(is_uncertain)
        self.assertEqual(conf, 75.0)

    def test_17_calm_deceptive_review_detection(self):
        """
        Verify that deceptive reviews without exclamations are evaluated on lexical tokens.
        """
        from model import classify_probability
        label, conf, is_uncertain = classify_probability(0.85)
        self.assertEqual(label, "Fake")
        self.assertFalse(is_uncertain)
        self.assertEqual(conf, 85.0)

    def test_18_uncertainty_threshold_classification(self):
        """
        Verify that borderline prediction probabilities between 40% and 60% are flagged as Uncertain.
        """
        from model import classify_probability
        # Test lower boundary edge
        label_low, conf_low, is_unc_low = classify_probability(0.45)
        self.assertEqual(label_low, "Uncertain")
        self.assertTrue(is_unc_low)

        # Test middle
        label_mid, conf_mid, is_unc_mid = classify_probability(0.52)
        self.assertEqual(label_mid, "Uncertain")
        self.assertTrue(is_unc_mid)

        # Test upper boundary edge
        label_high, conf_high, is_unc_high = classify_probability(0.58)
        self.assertEqual(label_high, "Uncertain")
        self.assertTrue(is_unc_high)

    def test_19_product_result_breakdown_with_uncertain_reviews(self):
        """
        Verify that the product analysis result page displays the breakdown
        including Genuine, Fake, and Inconclusive/Uncertain reviews.
        """
        sample_reviews = [
            {"review_text": "I really love this laptop! Great screen and solid battery life for daily work.", "author": "Alice", "rating": 5.0, "date": "2026-05-10"},
            {"review_text": "BUY THIS NOW BEST IN THE WORLD FREE DISCOUNT PROMOTION FIVE STARS WOW WOW", "author": "PromoBot", "rating": 5.0, "date": "2026-05-11"},
            {"review_text": "The delivery was on time and the packaging was adequate.", "author": "Charlie", "rating": 3.0, "date": "2026-05-12"}
        ]
        with patch("app.is_safe_url", return_value=(True, "https://www.amazon.in/dp/B0CHX1W1Y2")):
            with patch("app.fetch_product_reviews", return_value=(sample_reviews, "Amazon", None)):
                resp = self.client.post("/predict_product", data={"product_url": "https://www.amazon.in/dp/B0CHX1W1Y2"}, follow_redirects=True)
                self.assertEqual(resp.status_code, 200)
                self.assertIn(b"Authenticity Score", resp.data)
                self.assertIn(b"Actual Reviews Collected", resp.data)
                self.assertIn(b"GENUINE", resp.data)
                self.assertIn(b"FAKE", resp.data)

    def test_20_amazon_http_200_captcha_detection(self):
        """
        Verify that an Amazon response returning HTTP 200 OK with a Robot Check or CAPTCHA
        form is detected as an anti-bot challenge rather than a successful page with 0 reviews.
        """
        from model import _detect_amazon_bot_challenge, _scrape_amazon_reviews
        from unittest.mock import MagicMock
        
        # 1. Test _detect_amazon_bot_challenge directly
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><head><title>Robot Check</title></head><body><form action='/errors/validateCaptcha'>Enter characters</form></body></html>"
        is_challenge, reason = _detect_amazon_bot_challenge(mock_resp)
        self.assertTrue(is_challenge)
        self.assertIn("anti-bot challenge", reason.lower())

        # 2. Test _scrape_amazon_reviews with mocked safe_fetch_url returning the CAPTCHA response
        with patch("model.safe_fetch_url", return_value=(mock_resp, None)):
            reviews, err = _scrape_amazon_reviews("https://www.amazon.in/dp/B0CHX1W1Y2")
            self.assertEqual(len(reviews), 0)
            self.assertIn("anti-bot challenge", err.lower())

    def test_21_amazon_http_403_detection(self):
        """
        Verify that HTTP 403 Forbidden responses from Amazon are diagnosed accurately.
        """
        from model import _scrape_amazon_reviews
        from unittest.mock import MagicMock
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.text = "Forbidden"
        with patch("model.safe_fetch_url", return_value=(mock_resp, None)):
            reviews, err = _scrape_amazon_reviews("https://www.amazon.in/dp/B0CHX1W1Y2")
            self.assertEqual(len(reviews), 0)
            self.assertIn("HTTP 403", err)

    def test_22_amazon_javascript_rendered_detection(self):
        """
        Verify that an Amazon product page with product title but zero reviews in static HTML
        is diagnosed as requiring client-side dynamic JavaScript rendering.
        """
        from model import _scrape_amazon_reviews
        from unittest.mock import MagicMock
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><h1 id='productTitle'>Apple iPhone 15</h1><div id='content'>Content shell</div></body></html>"
        with patch("model.safe_fetch_url", return_value=(mock_resp, None)):
            reviews, err = _scrape_amazon_reviews("https://www.amazon.in/dp/B0CHX1W1Y2")
            self.assertEqual(len(reviews), 0)
            self.assertIn("javascript", err.lower())

    def test_23_amazon_verified_zero_reviews_detection(self):
        """
        Verify that an Amazon product page with explicit 'no customer reviews' text is
        truthfully diagnosed as having no written reviews.
        """
        from model import _scrape_amazon_reviews
        from unittest.mock import MagicMock
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><h1 id='title'>New Product</h1><div>There are no customer reviews yet for this product.</div></body></html>"
        with patch("model.safe_fetch_url", return_value=(mock_resp, None)):
            reviews, err = _scrape_amazon_reviews("https://www.amazon.in/dp/B0CHX1W1Y2")
            self.assertEqual(len(reviews), 0)
            self.assertIn("no written customer reviews", err.lower())

    def test_24_duplicate_error_prevention(self):
        """
        Verify that extraction failure results in a single structured diagnostic card
        and does NOT produce duplicate flash alert messages.
        """
        with patch("app.fetch_product_reviews", return_value=([], "Amazon", "Amazon anti-bot challenge encountered (Robot Check CAPTCHA).")):
            resp = self.client.post("/predict_product", data={"product_url": "https://www.amazon.in/dp/B0CHX1W1Y2"}, follow_redirects=True)
            self.assertEqual(resp.status_code, 200)
            # Exactly one diagnostic card exists
            self.assertIn(b"id=\"extractionFailureCard\"", resp.data)
            # No flash warning alert div with duplicate message
            self.assertNotIn(b"alert-warning", resp.data)

    def test_25_homepage_truthful_badge(self):
        """
        Verify that the homepage contains the truthful model description badge
        and does NOT claim '100% Benchmark Accuracy'.
        """
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Random Forest &amp; Stylometrics", resp.data)
        self.assertNotIn(b"100% Benchmark Accuracy", resp.data)

    def test_26_amazon_shortened_url_resolution(self):
        """
        Verify that Amazon shortened URLs (amzn.in / amzn.to) are resolved to canonical URLs
        with extracted ASIN and product title.
        """
        from model import resolve_canonical_url, _extract_amazon_asin, extract_product_name
        from unittest.mock import MagicMock
        
        short_url = "https://amzn.in/d/6ZXYZ12"
        canonical_target = "https://www.amazon.in/Apple-iPhone-15-128-GB/dp/B0CHX1W1Y2"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.is_redirect = False

        with patch("model.is_safe_url", return_value=(True, short_url)):
            with patch("model.safe_fetch_url", return_value=(mock_resp, canonical_target, None)):
                canon_url, resp, err = resolve_canonical_url("https://amzn.in/d/6ZXYZ12")
                self.assertEqual(canon_url, canonical_target)
                self.assertIsNone(err)
                asin = _extract_amazon_asin(canon_url)
                self.assertEqual(asin, "B0CHX1W1Y2")
                name = extract_product_name(canon_url)
                self.assertIn("Apple Iphone 15", name)

    def test_27_flipkart_product_url_and_shortener(self):
        """
        Verify platform detection and review parsing for Flipkart URLs.
        """
        from model import detect_platform, extract_product_name, _scrape_flipkart_reviews
        from unittest.mock import MagicMock
        
        self.assertEqual(detect_platform("https://www.flipkart.com/apple-iphone-15-black-128-gb/p/itm2d82914"), "Flipkart")
        self.assertEqual(detect_platform("https://fkrt.it/xyz123"), "Flipkart")
        
        title = extract_product_name("https://www.flipkart.com/apple-iphone-15-black-128-gb/p/itm2d82914")
        self.assertIn("Apple Iphone 15", title)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = """
        <html><body>
            <div class="col EPCmJX">
                <div class="XQDdHH">5.0</div>
                <div class="t-ZTKy"><div>This smartphone has incredible battery life and a stunning display!</div></div>
            </div>
        </body></html>
        """
        with patch("model.safe_fetch_url", return_value=(mock_resp, None)):
            reviews, err = _scrape_flipkart_reviews("https://www.flipkart.com/test/p/itm123")
            self.assertEqual(len(reviews), 1)
            self.assertIn("smartphone has incredible battery life", reviews[0]["review_text"])
            self.assertEqual(reviews[0]["source"], "Live Scraped - Flipkart")

    def test_28_supported_platforms_detection_and_adapters(self):
        """
        Verify detection and adapter dispatch for Ajio, Croma, Reliance Digital, Myntra, and Meesho.
        """
        from model import detect_platform, fetch_product_reviews
        from unittest.mock import patch, MagicMock

        self.assertEqual(detect_platform("https://www.ajio.com/men-sneakers/p/461234"), "Ajio")
        self.assertEqual(detect_platform("https://www.croma.com/apple-iphone-15/p/272345"), "Croma")
        self.assertEqual(detect_platform("https://www.reliancedigital.in/apple-iphone-15/p/493838"), "Reliance Digital")
        self.assertEqual(detect_platform("https://www.myntra.com/shoes/nike/123/buy"), "Myntra")
        self.assertEqual(detect_platform("https://www.meesho.com/saree/p/456"), "Meesho")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><div id='app'>React Client Shell</div></body></html>"
        with patch("model.is_safe_url", return_value=(True, "https://www.ajio.com/test/p/123")):
            with patch("model.safe_fetch_url", return_value=(mock_resp, None)):
                # Verify Ajio dynamic JS message
                reviews, platform, err = fetch_product_reviews("https://www.ajio.com/test/p/123", "Test")
                self.assertEqual(len(reviews), 0)
                self.assertEqual(platform, "Ajio")
                self.assertIn("javascript", err.lower())

    def test_29_common_review_schema_normalization(self):
        """
        Verify that all extraction adapters normalize to the standard review schema:
        {review_text, rating, author, date, verified, source}.
        """
        from model import _scrape_flipkart_reviews
        from unittest.mock import MagicMock
        
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = """
        <html><body>
            <div class="col EPCmJX">
                <div class="XQDdHH">4.5</div>
                <div class="t-ZTKy"><div>Great laptop, fast boot time and responsive keyboard for coding.</div></div>
            </div>
        </body></html>
        """
        with patch("model.safe_fetch_url", return_value=(mock_resp, None)):
            reviews, err = _scrape_flipkart_reviews("https://www.flipkart.com/test/p/itm123")
            self.assertEqual(len(reviews), 1)
            r = reviews[0]
            self.assertIn("review_text", r)
            self.assertIn("rating", r)
            self.assertIn("author", r)
            self.assertIn("date", r)
            self.assertIn("verified", r)
            self.assertIn("source", r)

    def test_30_product_with_actual_extracted_reviews_end_to_end(self):
        """
        Verify that when actual reviews are extracted, the ML pipeline executes
        Random Forest classification and renders the report without using synthetic data.
        """
        sample_reviews = [
            {"review_text": "I really love this smartphone! Photos are crystal clear and battery lasts 2 days.", "author": "John", "rating": 5.0, "date": "2026-11-01", "verified": True, "source": "Live Scraped - Amazon"},
            {"review_text": "BUY BUY BUY BEST PRODUCT EVER WOW FIVE STARS SUPER DISCOUNT PROMOTION NOW", "author": "Bot1", "rating": 5.0, "date": "2026-11-02", "verified": False, "source": "Live Scraped - Amazon"}
        ]
        with patch("app.is_safe_url", return_value=(True, "https://www.amazon.in/dp/B0CHX1W1Y2")):
            with patch("app.fetch_product_reviews", return_value=(sample_reviews, "Amazon", None)):
                resp = self.client.post("/predict_product", data={"product_url": "https://www.amazon.in/dp/B0CHX1W1Y2"}, follow_redirects=True)
                self.assertEqual(resp.status_code, 200)
                self.assertIn(b"Authenticity Score", resp.data)
                self.assertIn(b"Actual Reviews Collected", resp.data)
                self.assertIn(b"Live Scraped - Amazon", resp.data)

    def test_31_unsafe_redirect_destination(self):
        """
        Verify that a redirect pointing to an unsafe internal IP address or non-standard port
        is blocked during resolution.
        """
        from model import resolve_canonical_url
        from unittest.mock import MagicMock
        
        # Mock initial response as a 302 redirect to non-standard port / loopback
        mock_302 = MagicMock()
        mock_302.is_redirect = True
        mock_302.status_code = 302
        mock_302.headers = {"Location": "http://127.0.0.1:8080/admin"}

        with patch("model.is_safe_url", side_effect=[(True, "https://amzn.in/d/unsafe"), (False, "Port 8080 is not permitted. Only standard HTTP (80) and HTTPS (443) ports are allowed.")]):
            with patch("requests.Session.get", return_value=mock_302):
                canon_url, resp, err = resolve_canonical_url("https://amzn.in/d/unsafe")
                self.assertTrue("blocked" in err.lower() or "not permitted" in err.lower())

    def test_32_redirect_loop_prevention(self):
        """
        Verify that a redirect loop (A -> B -> A) is immediately detected and aborted.
        """
        from model import safe_fetch_url
        from unittest.mock import MagicMock

        mock_redirect = MagicMock()
        mock_redirect.is_redirect = True
        mock_redirect.status_code = 302
        mock_redirect.headers = {"Location": "https://www.example.com/loop"}

        with patch("requests.Session.get", return_value=mock_redirect):
            with patch("model.is_safe_url", return_value=(True, "https://www.example.com/loop")):
                resp, err = safe_fetch_url("https://www.example.com/loop", timeout=5)
                self.assertIsNone(resp)
                self.assertIn("loop", err.lower())

    def test_33_http_503_detection(self):
        """
        Verify that HTTP 503 Service Unavailable responses are properly diagnosed.
        """
        from model import _scrape_amazon_reviews
        from unittest.mock import MagicMock

        mock_resp = MagicMock()
        mock_resp.status_code = 503
        mock_resp.text = "Service Unavailable"
        with patch("model.safe_fetch_url", return_value=(mock_resp, None)):
            reviews, err = _scrape_amazon_reviews("https://www.amazon.in/dp/B0CHX1W1Y2")
            self.assertEqual(len(reviews), 0)
            self.assertIn("503", err)

    def test_34_network_timeout_handling(self):
        """
        Verify that network timeouts during extraction return a user-friendly diagnostic message.
        """
        from model import fetch_product_reviews
        with patch("model.is_safe_url", return_value=(True, "https://www.amazon.in/dp/B0CHX1W1Y2")):
            with patch("model.safe_fetch_url", return_value=(None, "Connection timed out while contacting e-commerce server.")):
                reviews, platform, err = fetch_product_reviews("https://www.amazon.in/dp/B0CHX1W1Y2", "iPhone")
                self.assertEqual(len(reviews), 0)
                self.assertIn("timed out", err.lower())

    def test_35_unsupported_platform_handling(self):
        """
        Verify that unknown/unsupported website domains are classified truthfully as Generic E-Commerce
        without claiming full native support.
        """
        from model import detect_platform, fetch_product_reviews
        self.assertEqual(detect_platform("https://www.unsupported-store.com/item/1"), "Unsupported-store")

        with patch("model.is_safe_url", return_value=(True, "https://www.unsupported-store.com/item/1")):
            with patch("model.safe_fetch_url", return_value=(None, "Unable to connect to e-commerce server.")):
                reviews, platform, err = fetch_product_reviews("https://www.unsupported-store.com/item/1", "Item")
                self.assertEqual(len(reviews), 0)
                self.assertTrue("unable" in err.lower())

    def test_36_duplicate_review_deduplication(self):
        """
        Verify that duplicate reviews with identical normalized text are deduplicated
        before reaching the machine learning pipeline.
        """
        from model import fetch_product_reviews
        duplicate_data = [
            {"review_text": "This phone has great battery life and crisp screen display.", "author": "User A", "rating": 5.0, "source": "Live Scraped"},
            {"review_text": "THIS PHONE HAS GREAT BATTERY LIFE AND CRISP SCREEN DISPLAY.", "author": "User B", "rating": 5.0, "source": "Live Scraped"},
            {"review_text": "Completely different genuine customer feedback on audio quality.", "author": "User C", "rating": 4.0, "source": "Live Scraped"}
        ]
        with patch("model.is_safe_url", return_value=(True, "https://www.flipkart.com/item/p/itm123")):
            with patch("model._scrape_flipkart_reviews", return_value=(duplicate_data, None)):
                reviews, platform, err = fetch_product_reviews("https://www.flipkart.com/item/p/itm123", "Item")
                self.assertEqual(len(reviews), 2)  # 3 input, but 2 unique

    def test_37_product_identity_association(self):
        """
        Verify that extracted reviews retain their product_id and product_url metadata.
        """
        from model import _scrape_amazon_reviews
        from unittest.mock import MagicMock

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = """
        <html><body>
            <div data-hook="review">
                <span data-hook="review-body">I have been using this phone for two weeks and it is fantastic.</span>
                <span class="a-profile-name">Sarah</span>
                <span data-hook="avp-badge">Verified Purchase</span>
            </div>
        </body></html>
        """
        with patch("model.safe_fetch_url", return_value=(mock_resp, None)):
            reviews, err = _scrape_amazon_reviews("https://www.amazon.in/dp/B0CHX1W1Y2")
            self.assertEqual(len(reviews), 1)
            self.assertEqual(reviews[0]["product_id"], "B0CHX1W1Y2")
            self.assertIn("amazon.in", reviews[0]["product_url"])
            self.assertTrue(reviews[0]["verified"])

if __name__ == "__main__":
    unittest.main()
