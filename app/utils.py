"""
Utility functions for file I/O, LLM initialization, and parsing
"""
import os
import json
from typing import Optional, List, Dict, Tuple
from datetime import datetime
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama

from .config import *
from .models import BackgroundInfo, StudentProfile, SimuCaseFile

# Ensure .env is loaded when this module is imported
load_dotenv()

# --- FILE I/O ---
def load_json(filepath: str, default) -> any:
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return default

def save_json(filepath: str, data):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def load_prompt(filename: str) -> str:
    """Load prompt template from file."""
    filepath = os.path.join(PROMPTS_DIR, filename)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Warning: Prompt file {filename} not found. Using default.")
        return ""

def save_case_file(content: str, filepath: str) -> str:
    """Save case file to specified path."""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    return filepath

def save_case_to_db(case_data: dict):
    """Save case metadata to database."""
    db = load_json(CASES_DB, {"cases": []})
    case_data["id"] = f"case_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(db['cases'])}"
    db["cases"].append(case_data)
    save_json(CASES_DB, db)
    return case_data["id"]

# --- LLM INITIALIZATION ---
def get_llm(model_name: str):
    """Initialize LLM with proper API key configuration."""
    model_id = MODEL_MAP.get(model_name, "llama3.2:latest")

    if model_name in FREE_MODELS:
        return ChatOllama(model=model_id, temperature=0.7)
    elif model_name == "GPT-4o":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment. Please check your .env file.")
        return ChatOpenAI(model=model_id, temperature=0.7, api_key=api_key).with_structured_output(SimuCaseFile)
    elif model_name == "Gemini 2.5 Pro":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment. Please check your .env file.")
        return ChatGoogleGenerativeAI(model=model_id, temperature=0.7, google_api_key=api_key).with_structured_output(SimuCaseFile)
    elif model_name in ["Claude 3 Opus", "Claude 3.5 Sonnet"]:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment. Please check your .env file.")
        return ChatAnthropic(model=model_id, temperature=0.7, api_key=api_key).with_structured_output(SimuCaseFile)
    else:
        return ChatOllama(model="llama3.2:latest", temperature=0.7)

# --- PARSING ---
def parse_ollama_response(response) -> Optional[SimuCaseFile]:
    """Parse free model response - handle both JSON and natural text."""
    try:
        content = response.content if hasattr(response, 'content') else str(response)
        
        try:
            data = json.loads(content)
            bg = BackgroundInfo(
                medical_history=data["student_profile"]["background"]["medical_history"],
                parent_concerns=data["student_profile"]["background"]["parent_concerns"],
                teacher_concerns=data["student_profile"]["background"]["teacher_concerns"]
            )
            
            profile = StudentProfile(
                name=data["student_profile"]["name"],
                age=data["student_profile"]["age"],
                grade_level=data["student_profile"]["grade_level"],
                gender=data["student_profile"]["gender"],
                background=bg
            )
            
            return SimuCaseFile(
                student_profile=profile,
                annual_goals=data["annual_goals"],
                latest_session_notes=data["latest_session_notes"]
            )
        except json.JSONDecodeError:
            print("Could not parse as JSON, using text extraction")
            return None
            
    except Exception as e:
        print(f"Parse error: {e}")
        return None

# --- AI HELPERS ---
def ai_grammar_check(text: str) -> str:
    """Use AI to check grammar and clarity."""
    try:
        llm = ChatOllama(model="llama3.2:latest", temperature=0.3)
        prompt_template = load_prompt("grammar_check.txt")
        if not prompt_template:
            prompt_template = "Improve this text: {feedback_text}\n\nCorrected version:"
        
        prompt = prompt_template.format(feedback_text=text)
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else text
        
        lines = [line for line in content.split('\n') 
                if not any(x in line.lower() for x in ['i corrected', 'i changed', 'i improved', 'here is', 'the corrected'])]
        return '\n'.join(lines).strip()
    except:
        return text

def analyze_feedback_category(text: str, existing_categories: List[str]) -> Tuple[str, bool]:
    """Generate descriptive category from feedback text."""
    try:
        llm = ChatOllama(model="llama3.2:latest", temperature=0.3)
        prompt_template = load_prompt("feedback_category.txt")
        if not prompt_template:
            prompt_template = "Categorize this feedback: {feedback_text}\nExisting: {existing_categories}\nCategory:"
        
        prompt = prompt_template.format(
            feedback_text=text,
            existing_categories=', '.join(existing_categories) if existing_categories else 'None'
        )
        
        response = llm.invoke(prompt)
        category = response.content.strip() if hasattr(response, 'content') else "General Feedback"
        is_new = category not in existing_categories
        
        return category, is_new
    except:
        return "General Feedback", True

# --- GROUP COMPATIBILITY ---
def check_grade_compatibility(grades: List[str]) -> Tuple[bool, str]:
    """Check if grades are within 2-level difference."""
    grade_nums = []
    for grade in grades:
        if grade in ["Pre-K", "Kindergarten"]:
            grade_nums.append(0 if grade == "Pre-K" else 1)
        else:
            try:
                num = int(grade.split()[0].replace("st", "").replace("nd", "").replace("rd", "").replace("th", ""))
                grade_nums.append(num + 1)
            except:
                grade_nums.append(1)
    
    if not grade_nums:
        return False, "No grades specified"
    
    grade_range = max(grade_nums) - min(grade_nums)
    if grade_range <= 2:
        return True, f"Grade range: {grade_range} levels (✓ Compatible)"
    else:
        return False, f"Grade range: {grade_range} levels (✗ Must be within 2 levels)"

def check_disorder_compatibility(disorders_list: List[List[str]]) -> Tuple[bool, str]:
    """Check if disorder combinations are compatible."""
    disorder_indices = []
    for disorders in disorders_list:
        indices = [DISORDER_TYPES.index(d) for d in disorders if d in DISORDER_TYPES]
        disorder_indices.append(set(indices))
    
    all_indices = set()
    for indices in disorder_indices:
        all_indices.update(indices)
    
    has_speech = any(i in all_indices for i in [0, 1, 2])
    has_language = any(i in all_indices for i in [3, 4, 5, 6])
    has_fluency = 7 in all_indices
    has_cas = 8 in all_indices
    
    valid = False
    reason = ""
    
    if has_speech and not has_fluency:
        valid = True
        reason = "Speech sound disorders compatible"
    elif has_language and not (has_speech and has_cas):
        valid = True  
        reason = "Language disorders compatible"
    elif has_speech and has_language and not has_cas:
        valid = True
        reason = "Speech + Language compatible"
    elif has_fluency and (has_language or has_cas) and not has_speech:
        valid = True
        reason = "Fluency compatible with language/CAS"
    elif has_cas and (has_speech or has_language):
        valid = True
        reason = "CAS compatible"
    else:
        reason = "Disorder combination not recommended per grouping strategies"
    
    return valid, reason

def search_existing_cases(grades: List[str], disorders_list: List[List[str]]) -> List[Dict]:
    """Search for existing cases matching criteria."""
    db = load_json(CASES_DB, {"cases": []})
    matching_cases = []
    
    for case in db["cases"]:
        case_grade = case.get("grade", "")
        case_disorders = case.get("disorders", [])
        
        for i, (grade, disorders) in enumerate(zip(grades, disorders_list)):
            if case_grade == grade and any(d in case_disorders for d in disorders):
                matching_cases.append({
                    "case_id": case.get("id", ""),
                    "member_index": i,
                    "grade": case_grade,
                    "disorders": case_disorders,
                    "filepath": case.get("filepath", "")
                })
                break
    
    return matching_cases
