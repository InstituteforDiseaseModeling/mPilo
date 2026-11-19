"""Central configuration for Mpilo.

Every API key, secret, and environment-specific setting is read here and
nowhere else. Other modules import values from this file instead of calling
``os.getenv`` or hardcoding resource names.

Values come from, in order of precedence:

1. Variables already set in the process environment.
2. A ``.env`` file in the project root (copy ``.env.example`` to get started).
3. Optionally, AWS Secrets Manager: if ``ENV`` is not ``local`` and
   ``AWS_SECRETS_NAME`` is set, the JSON secret is loaded into the
   environment by :func:`get_secrets`.
"""

import json
import os
from functools import lru_cache

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(".env", usecwd=True))

# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------
# One of: "local", "development", "production".
ENV = os.getenv("ENV", "local")
IS_LOCAL = ENV == "local"
IS_PRODUCTION = ENV == "production"

# --------------------------------------------------------------------------
# AWS (optional; only used when ENV != "local")
# --------------------------------------------------------------------------
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
# Name of a Secrets Manager secret holding a JSON object of API keys.
AWS_SECRETS_NAME = os.getenv("AWS_SECRETS_NAME", "")
# Name of a Secrets Manager secret holding a Google service-account JSON.
GOOGLE_TRANSLATE_SECRET_NAME = os.getenv("GOOGLE_TRANSLATE_SECRET_NAME", "")
# S3 bucket containing source documents (parsed WHO guideline elements, test docs).
DOCS_S3_BUCKET = os.getenv("DOCS_S3_BUCKET", "")
# S3 prefix containing model folders, e.g. "s3://my-bucket/models".
MODELS_S3_URI = os.getenv("MODELS_S3_URI", "").rstrip("/")
# Optional S3 bucket used as a remote cache for table OCR results.
REMOTE_CACHE_S3_BUCKET = os.getenv("REMOTE_CACHE_S3_BUCKET", "")

# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
TABLE_TO_IMAGE_OPENAI_MODEL = os.getenv("TABLE_TO_IMAGE_OPENAI_MODEL", "gpt-4-vision-preview")

# --------------------------------------------------------------------------
# Services
# --------------------------------------------------------------------------
REDIS_SERVER_ADDRESS = os.getenv("REDIS_SERVER_ADDRESS", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
# Local path (relative to project root) to a Google service-account JSON file.
GOOGLE_TRANSLATE_AUTH_FILE = os.getenv("GOOGLE_TRANSLATE_AUTH_FILE", "")
# Basic-auth credentials for the Gradio demo. Leave empty to disable auth.
DEMO_USERNAME = os.getenv("DEMO_USERNAME", "")
DEMO_PASSWORD = os.getenv("DEMO_PASSWORD", "")


def demo_auth():
    """Return a (username, password) tuple for Gradio, or None if unset."""
    if DEMO_USERNAME and DEMO_PASSWORD:
        return (DEMO_USERNAME, DEMO_PASSWORD)
    return None


def _secrets_manager_client():
    import boto3

    return boto3.session.Session().client(service_name="secretsmanager", region_name=AWS_REGION)


@lru_cache(maxsize=1)
def get_secrets():
    """Load API keys into the environment and return all settings as a dict.

    API keys (``OPENAI_API_KEY``, ``TWILIO_ACCOUNT_SID``, ``TWILIO_AUTH_TOKEN``,
    ...) are expected in ``.env`` or the environment.
    When running outside ``local`` with ``AWS_SECRETS_NAME`` set, they are
    fetched from AWS Secrets Manager instead (existing environment variables
    take precedence).
    """
    if not IS_LOCAL and AWS_SECRETS_NAME:
        response = _secrets_manager_client().get_secret_value(SecretId=AWS_SECRETS_NAME)
        for key, value in json.loads(response["SecretString"]).items():
            os.environ.setdefault(key, value)
    return dict(os.environ)


def get_secret(name, default=None):
    """Return a single secret/setting by name after loading secrets."""
    return get_secrets().get(name, default)


def google_service_account_info():
    """Return the Google service-account credentials as a dict.

    Reads from AWS Secrets Manager when ``GOOGLE_TRANSLATE_SECRET_NAME`` is set
    and ``ENV`` is not ``local``; otherwise from ``GOOGLE_TRANSLATE_AUTH_FILE``.
    """
    if not IS_LOCAL and GOOGLE_TRANSLATE_SECRET_NAME:
        response = _secrets_manager_client().get_secret_value(SecretId=GOOGLE_TRANSLATE_SECRET_NAME)
        return json.loads(response["SecretString"])
    if not GOOGLE_TRANSLATE_AUTH_FILE:
        raise RuntimeError("Set GOOGLE_TRANSLATE_AUTH_FILE in .env to a Google service-account JSON file.")
    project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    path = os.path.join(project_root, GOOGLE_TRANSLATE_AUTH_FILE)
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    # Lets shell scripts read a single value: python -m src.config OPENAI_MODEL
    import sys

    print(get_secret(sys.argv[1], "") or "")
