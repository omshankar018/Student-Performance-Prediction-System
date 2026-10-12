# ============================================================
# gemini_helper.py — thin wrapper around the Gemini API
# ============================================================
#
# Exposes ask_gemini(prompt) -> str, used by chatbot.py.
#
# NOTE: This uses the new unified `google-genai` SDK, not the older
# `google-generativeai` package. Google deprecated `google-generativeai`
# and has been shutting down old model versions (gemini-1.5-* models are
# no longer served — that's what the "404 model not found" error meant).
# The new SDK + gemini-3.5-flash are the current, supported combination.
#
# ------------------------------------------------------------
# PASTE YOUR GEMINI API KEY HERE (between the quotes):
# ------------------------------------------------------------
GEMINI_API_KEY = "insert api key"   # <-- e.g. GEMINI_API_KEY = "AIzaSyD...your-key..."
# ------------------------------------------------------------
#
# If you leave this blank, the app will also check (in this order):
#   1. Streamlit secrets  -> .streamlit/secrets.toml -> GEMINI_API_KEY = "..."
#   2. An environment variable named GEMINI_API_KEY
#
# Get a free key at: https://aistudio.google.com/app/apikey
#
# One-time setup before running the app (in your terminal):
#   pip uninstall -y google-generativeai
#   pip install google-genai

import os
import streamlit as st

MODEL_NAME = "gemini-3.5-flash"


def _get_api_key():
    """Check the pasted key first, then Streamlit secrets, then an env var."""
    if GEMINI_API_KEY.strip() and GEMINI_API_KEY.strip() != "insert api key":
        return GEMINI_API_KEY.strip()
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.environ.get("GEMINI_API_KEY")


@st.cache_resource
def _get_client():
    # Imported here (not at the top of the file) so that a missing
    # package only breaks the chatbot feature when it's actually used,
    # instead of crashing the entire Streamlit app on startup.
    try:
        from google import genai
    except ModuleNotFoundError:
        raise RuntimeError(
            "The 'google-genai' package isn't installed (or the old, "
            "deprecated 'google-generativeai' package is still installed "
            "instead). Open a terminal in your project folder and run:\n\n"
            "    pip uninstall -y google-generativeai\n"
            "    pip install google-genai\n\n"
            "then restart the app."
        )

    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Paste your key at the top of "
            "gemini_helper.py, add it to .streamlit/secrets.toml as "
            'GEMINI_API_KEY = "your-key-here", or set it as an environment variable.'
        )

    return genai.Client(api_key=api_key)


def ask_gemini(prompt):
    """
    Send a raw prompt to Gemini and return its text response.
    Raises on failure so the caller can show a clean error message
    instead of a raw traceback.
    """
    client = _get_client()
    response = client.models.generate_content(model=MODEL_NAME, contents=prompt)

    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError("Gemini returned an empty response. Please try rephrasing your question.")
    return text.strip()
