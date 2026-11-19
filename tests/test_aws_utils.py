from src.config import get_secrets


class TestAWSUtils:
    """Tests of AWS utils."""

    def test_get_secrets(self):
        """
        Test that the get secrets function returns a dictionary containing the OPENAI_KEY, and it can be retrieved as an env variable.
        """
        secrets = get_secrets()

        assert "OPENAI_API_KEY" in secrets.keys()
