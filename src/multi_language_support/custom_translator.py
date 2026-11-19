from src.aws_utils import authenticated_translate_client
import html


class CustomGoogleTranslator:
    """Google Translate custom class."""

    def __init__(self):
        """Instantiate class."""
        self.translate_client = authenticated_translate_client()

    def translate(self, values, target_language="en", source_language=None):
        """Execute translation and applies unescape to text results.

        Parameters
        ----------
        text : value
            Text string to be translated.
        source_language : str
            The language of the text to be translated. If None, source language will be detected automatically.
        output_language : str
            Language translate input to. Defaults to "en" - English.

        Returns
        -------
        dict
            Returns a dictionary with the translated text and the detected language.

        """
        if target_language == source_language:
            # Google API will throw an error if target and source languages are the same
            translate_result = {
                "translatedText": values,
                "input": values,
            }
            return translate_result

        # Get response from Google Translate API
        translate_result = self.translate_client.translate(
            values=values, target_language=target_language, source_language=source_language
        )
        # Unescape any ASCII characters returned by the API
        translate_result["translatedText"] = html.unescape(translate_result["translatedText"])
        return translate_result
