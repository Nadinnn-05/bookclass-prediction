import json
from unittest.mock import MagicMock, patch

import requests
from django.test import TestCase
from django.urls import reverse

from . import ai_explain


def _reset_ai_state():
    ai_explain._cache.clear()
    ai_explain._cache_prediksi.clear()
    ai_explain._limit_per_model.clear()
    ai_explain._model_tidak_tersedia.clear()
    ai_explain._daftar_model_cache.clear()


class KlasifikasiViewTests(TestCase):
    def setUp(self):
        _reset_ai_state()

    def test_index_page_renders_with_chart_and_groq_only(self):
        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "BookClass")
        self.assertContains(response, "Penjelasan Grafik")
        self.assertContains(response, "groq-limit-banner")
        self.assertNotContains(response, "ai-gemini")
        self.assertIn("grafik", response.context)
        self.assertTrue(response.context["grafik"])

    def test_valid_prediction_returns_result_even_when_all_groq_models_limited(self):
        for m in ai_explain._semua_kandidat_model():
            ai_explain._tandai_limit_model(m, 120)

        response = self.client.post(reverse("index"), {
            "user_rating": "4.7",
            "reviews": "15200",
            "price": "12",
            "year": "2018",
        })

        self.assertEqual(response.status_code, 200)
        self.assertIsNotNone(response.context["hasil"])
        self.assertIn(response.context["hasil"]["label"], ("Fiction", "Non Fiction"))
        self.assertTrue(response.context["status_ai"]["limit"])
        self.assertContains(response, "Penjelasan AI (Groq)")
        self.assertContains(response, "prediksi-payload")

    @patch("klasifikasi.ai_explain.tanya_groq", return_value=("Penjelasan grafik dari Groq.", "openai/gpt-oss-120b"))
    def test_penjelasan_grafik_endpoint(self, mock_groq):
        response = self.client.get(reverse("penjelasan"))

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["groq"]["ok"])
        self.assertEqual(data["groq"]["teks"], "Penjelasan grafik dari Groq.")
        self.assertFalse(data["status"]["limit"])
        mock_groq.assert_called_once()

    @patch("klasifikasi.ai_explain.tanya_groq", return_value=("Buku ini diprediksi Fiction karena jumlah ulasan tinggi.", "openai/gpt-oss-120b"))
    def test_penjelasan_prediksi_endpoint_valid(self, mock_groq):
        payload = {
            "user_rating": 4.7,
            "reviews": 15200,
            "price": 12.0,
            "year": 2018,
            "label": "Fiction",
            "keyakinan": 72.5,
        }
        response = self.client.post(
            reverse("penjelasan_prediksi"),
            data=json.dumps(payload),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["ok"])
        self.assertIn("Fiction", data["teks"])
        mock_groq.assert_called_once()

    def test_penjelasan_prediksi_endpoint_invalid_payload(self):
        response = self.client.post(
            reverse("penjelasan_prediksi"),
            data=json.dumps({"label": "Unknown"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.json()["ok"])


class AiExplainGroqTests(TestCase):
    def setUp(self):
        _reset_ai_state()

    @patch("klasifikasi.ai_explain.requests.post")
    @patch.dict("os.environ", {"GROQ_API_KEY": "gsk_test_key"})
    def test_groq_429_auto_fallbacks_to_secondary_model(self, mock_post):
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.headers = {"retry-after": "45"}
        resp_429.raise_for_status.side_effect = requests.HTTPError(response=resp_429)

        resp_ok = MagicMock()
        resp_ok.raise_for_status.return_value = None
        resp_ok.headers = {}
        resp_ok.json.return_value = {
            "choices": [{"message": {"content": "Sukses dengan model cadangan."}}]
        }
        mock_post.side_effect = [resp_429, resp_ok]

        teks, model_dipakai = ai_explain.tanya_groq("tes")
        self.assertEqual(teks, "Sukses dengan model cadangan.")
        self.assertEqual(model_dipakai, ai_explain.GROQ_FALLBACK_MODELS[0])
        self.assertGreater(ai_explain._sisa_limit_model(ai_explain.GROQ_MODEL), 0)
        # Karena masih ada model cadangan yang tersedia, status global belum limit
        self.assertFalse(ai_explain.status_groq()["limit"])

    @patch.dict("os.environ", {"GROQ_API_KEY": "gsk_test_key"})
    def test_all_models_429_sets_limit_status_before_predict(self):
        for m in ai_explain._semua_kandidat_model():
            ai_explain._tandai_limit_model(m, 45)

        status_resp = self.client.get(reverse("status_ai"))
        self.assertEqual(status_resp.status_code, 200)
        status_data = status_resp.json()
        self.assertTrue(status_data["limit"])
        self.assertEqual(status_data["alasan"], "rate_limit")
        self.assertGreater(status_data["sisa_detik"], 0)

    @patch("klasifikasi.ai_explain.requests.get")
    @patch("klasifikasi.ai_explain.requests.post")
    @patch.dict("os.environ", {"GROQ_API_KEY": "gsk_test_key"})
    def test_groq_404_auto_fallbacks_to_active_model(self, mock_post, mock_get):
        resp_404 = MagicMock()
        resp_404.status_code = 404
        resp_404.raise_for_status.side_effect = requests.HTTPError(response=resp_404)

        resp_models = MagicMock()
        resp_models.raise_for_status.return_value = None
        resp_models.json.return_value = {
            "data": [
                {"id": "whisper-large-v3"},
                {"id": "openai/gpt-oss-20b"},
            ]
        }
        mock_get.return_value = resp_models

        resp_ok = MagicMock()
        resp_ok.raise_for_status.return_value = None
        resp_ok.headers = {}
        resp_ok.json.return_value = {
            "choices": [{"message": {"content": "Berhasil dengan model fallback."}}]
        }
        mock_post.side_effect = [resp_404, resp_ok]

        teks, model_dipakai = ai_explain.tanya_groq("tes")
        self.assertEqual(teks, "Berhasil dengan model fallback.")
        self.assertEqual(model_dipakai, "openai/gpt-oss-20b")
