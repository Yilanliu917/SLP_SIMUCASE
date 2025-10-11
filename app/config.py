"""
Configuration and constants for SLP SimuCase Generator
"""
import os

# --- PATHS ---
DB_PATH = "data/slp_vector_db"
EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_OUTPUT_PATH = "generated_case_files/"
FEEDBACK_LOG = "feedback_log.json"
FEEDBACK_CATEGORIES = "feedback_categories.json"
CASES_DB = "cases_database.json"
PROMPTS_DIR = "prompts/"

# Ensure directories exist
os.makedirs(DEFAULT_OUTPUT_PATH, exist_ok=True)
os.makedirs(PROMPTS_DIR, exist_ok=True)

# --- CONSTANTS ---
ALL_GRADES = [
    "Pre-K", "Kindergarten", "1st Grade", "2nd Grade", "3rd Grade", "4th Grade", 
    "5th Grade", "6th Grade", "7th Grade", "8th Grade", "9th Grade", "10th Grade", 
    "11th Grade", "12th Grade"
]

DISORDER_TYPES = [
    "Speech Sound Disorder", "Articulation Disorders", "Phonological Disorders",
    "Language Disorders", "Receptive Language Disorders", "Expressive Language Disorders",
    "Pragmatics", "Fluency", "Childhood Apraxia of Speech"
]

# Check if running on Hugging Face
IS_HUGGINGFACE = os.getenv("SPACE_ID") is not None or os.getenv("SPACE_AUTHOR_NAME") is not None

# Conditional model availability
if IS_HUGGINGFACE:
    # Only premium models on cloud
    FREE_MODELS = []
    PREMIUM_MODELS = ["GPT-4o", "Gemini 2.5 Pro", "Claude 3 Opus", "Claude 3.5 Sonnet"]
    DEFAULT_MODEL = "GPT-4o"
else:
    # All models when running locally
    FREE_MODELS = ["Llama3.2", "Qwen 2.5 7B", "Qwen 2.5 32B", "DeepSeek R1 32B"]
    PREMIUM_MODELS = ["GPT-4o", "Gemini 2.5 Pro", "Claude 3 Opus", "Claude 3.5 Sonnet"]
    DEFAULT_MODEL = "Llama3.2"

# All available models
ALL_MODELS = FREE_MODELS + PREMIUM_MODELS

MODEL_MAP = {
    "Llama3.2": "llama3.2:latest",
    "Qwen 2.5 7B": "qwen2.5:7b",
    "Qwen 2.5 32B": "qwen2.5:32b",
    "DeepSeek R1 32B": "deepseek-r1:32b",
    "GPT-4o": "gpt-4o",
    "Gemini 2.5 Pro": "gemini-2.5-pro",
    "Claude 3 Opus": "claude-3-opus-20240229",
    "Claude 3.5 Sonnet": "claude-3-5-sonnet-20240620"
}

# Grouping strategy compatibility - disorder indices
DISORDER_GROUPS = {
    "speech_sounds": [0, 1, 2],  # Speech Sound, Articulation, Phonological
    "language": [3, 4, 5, 6],    # Language, Receptive, Expressive, Pragmatics
    "fluency": [7],              # Fluency
    "cas": [8]                   # Childhood Apraxia of Speech
}

MAX_CONDITION_ROWS = 50