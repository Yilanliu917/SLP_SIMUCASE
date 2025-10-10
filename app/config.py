import os

# ... existing code ...

# Check if running on Hugging Face
IS_HUGGINGFACE = os.getenv("SPACE_ID") is not None or os.getenv("SPACE_AUTHOR_NAME") is not None

# Conditional model availability
if IS_HUGGINGFACE:
    FREE_MODELS = []
    PREMIUM_MODELS = ["GPT-4o", "Gemini 2.5 Pro", "Claude 3 Opus", "Claude 3.5 Sonnet"]
    DEFAULT_MODEL = "GPT-4o"
else:
    FREE_MODELS = ["Llama3.2", "Qwen 2.5 7B", "Qwen 2.5 32B", "DeepSeek R1 32B"]
    PREMIUM_MODELS = ["GPT-4o", "Gemini 2.5 Pro", "Claude 3 Opus", "Claude 3.5 Sonnet"]
    DEFAULT_MODEL = "Llama3.2"

ALL_MODELS = FREE_MODELS + PREMIUM_MODELS