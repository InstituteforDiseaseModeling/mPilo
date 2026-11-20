# Mpilo: an LLM-based HIV guidance chatbot

Mpilo is a conversational agent that gives guidance on HIV prevention, testing, treatment and care. It answers questions using retrieval over WHO HIV guidelines, adapts its coaching to the user's behavioural segment, keeps memory across conversations, and supports multiple languages. It can run as a local command-line chatbot, a Gradio web demo, or a WhatsApp bot through Twilio. The code is designed so it can be adapted to other health-guidance applications.

## Project status

This repository is a research code release accompanying a forthcoming article. It is provided as-is and is **not actively maintained or supported**: issues and pull requests may not receive a response.

## Repository layout

| Path | Contents |
|---|---|
| `src/config.py` | **The single place where all keys and settings are read** |
| `src/chatbot/` | Agent, prompts, tools, user segmentation, Gradio frontend |
| `src/retrieval/` | Contextual retrieval and reranking over guideline documents |
| `src/data_preprocessing/` | PDF parsing, table/slide OCR, element contextualisation |
| `src/persistent_memory/` | Long-term memory summarisation backed by Redis |
| `src/multi_language_support/` | Translation layer (Google Translate) |
| `src/server/` | FastAPI + Twilio WhatsApp webhook |
| `src/evaluation/` | LLM-based evaluation and simulated users |
| `scripts/` | Install, demo and WhatsApp launch scripts |
| `tests/` | Pytest suite |
| `hiv_entry_script.py` | Local command-line chatbot |

## Configuration

All API keys and environment-specific settings live in `.env` and are read only by [`src/config.py`](src/config.py). No other module reads environment variables directly.

```bash
cp .env.example .env
# then edit .env and fill in your keys
```

Main settings:

- `OPENAI_API_KEY` is required.
- `GOOGLE_TRANSLATE_AUTH_FILE` is the path to a Google Cloud service-account JSON, needed for translation.
- `REDIS_SERVER_ADDRESS` is needed for the WhatsApp server and persistent memory.
- `TWILIO_ACCOUNT_SID` and `TWILIO_AUTH_TOKEN` are needed for the WhatsApp bot.
- With `ENV=development` or `ENV=production`, keys can come from AWS Secrets Manager (`AWS_SECRETS_NAME`). Documents and models can come from S3 (`DOCS_S3_BUCKET`, `MODELS_S3_URI`).

See [`.env.example`](.env.example) for the full list.

## Running locally (tested on Ubuntu 22.04)

```bash
chmod +x scripts/install_local.sh
./scripts/install_local.sh          # installs system deps and creates ~/mpilo-env
source ~/mpilo-env/bin/activate
pip install -e .
```

Data and model artifacts are not included in this repository. Before running, provide:

1. `.env`: your configuration (see above).
2. `vectorstore/`: a FAISS index built from the source guideline documents. The preprocessing code is in `src/data_preprocessing/` and `src/retrieval/`.
3. `gpt4-finetuned-retriever/`: a sentence-transformers embedding model for retrieval. Any compatible model can be used instead.
4. `functionary/` (optional): a clone of [MeetKai/functionary](https://github.com/MeetKai/functionary), needed only for self-hosted function-calling models.

Then start the chatbot:

```bash
python hiv_entry_script.py
```

Other entry points:

```bash
python scripts/demo_chatbot.py      # Gradio web demo
./scripts/whatsapp_chatbot.sh       # WhatsApp webhook server (FastAPI) on port 8000
pytest tests/                       # test suite (some tests need API keys and S3 test documents)
```

### WhatsApp bot

`scripts/whatsapp_chatbot.sh` starts the Twilio webhook server on port 8000. Twilio has to reach it over public HTTPS, so expose port 8000 with a reverse proxy, a cloud host or a tunnelling service. Then set `https://<your-host>/hook` as the incoming-message webhook for your Twilio WhatsApp sender. With `ENV=production`, the server checks Twilio's request signature, so the public URL must match the URL configured in Twilio exactly.

## License

The code is released under the [MIT License](LICENSE). Data and model artifacts are not included. If you use third-party materials with this code, such as WHO guideline documents or pretrained models, their own licenses and terms of use apply.

## Citation

If you use this software, please cite it using the metadata in [`CITATION.cff`](CITATION.cff).

## Disclaimer

Mpilo is research software. It is not a medical device and does not replace advice from a qualified healthcare provider.
