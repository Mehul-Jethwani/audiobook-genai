"""
config.py

Central place for every tunable constant in the pipeline. Previously these
were scattered across structure.py, emotion.py, dialogue_speaker.py, and
llm_client.py individually - changing "how many pages per API call" meant
hunting through multiple files. Now it's one place.
"""

# --------------------------------------------------------------------------
# Batch sizes - how many items go into a SINGLE LLM call.
# --------------------------------------------------------------------------
STRUCTURE_BATCH_SIZE = 20
EMOTION_BATCH_SIZE = 28
DIALOGUE_SPEAKER_BATCH_SIZE = 20

# --------------------------------------------------------------------------
# Rate-limit handling (see llm_client.py). A 429 / rate-limit error is
# TEMPORARY - the key recovers on its own after this many seconds.
# --------------------------------------------------------------------------
RATE_LIMIT_COOLDOWN_SECONDS = 65
MAX_WAIT_ROUNDS = 4

# --------------------------------------------------------------------------
# extract.py
# --------------------------------------------------------------------------
OCR_FALLBACK_THRESHOLD = 20   # chars; below this a page is treated as scanned
OCR_RENDER_DPI = 200

# --------------------------------------------------------------------------
# paragraphs.py
# --------------------------------------------------------------------------
MIN_WORDS_PER_CHUNK = 70
MAX_WORDS_PER_CHUNK = 260
