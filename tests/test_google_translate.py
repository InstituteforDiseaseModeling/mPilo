from src.multi_language_support.custom_translator import CustomGoogleTranslator
import pytest


class TestGoogleTranslate:
    """Tests for the Google Translate API."""

    @pytest.mark.parametrize(("expected_keys"), [(["translatedText", "detectedSourceLanguage", "input"])])
    def test_custom_google_translator(self, expected_keys):
        """Test custom Google Translate client capabilities."""
        # Test: API returns dict with expected keys
        translate_client = CustomGoogleTranslator()
        es_text = "El mundo está cambiado. Lo siento en el agua. Lo siento en la tierra. Lo huelo en el aire."
        translate_response = translate_client.translate(es_text, target_language="en")
        assert type(translate_response) == dict
        assert all([key in translate_response for key in expected_keys])

        # Test: API returns same input if source and target language are equal
        en_text = "No, I am your father."
        translate_response = translate_client.translate(en_text, source_language="en", target_language="en")
        assert translate_response["translatedText"] == en_text

        # Test: API escapes special characters
        spe_text = "Que és una 'chinforimpula'?"
        spe_text_translate = "What is a 'chinforimpula'?"
        translate_response = translate_client.translate(spe_text, target_language="en")
        assert translate_response["translatedText"] == spe_text_translate
