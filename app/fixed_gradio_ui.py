import os
import json
import re
from dotenv import load_dotenv
from typing import List, Dict, Optional
from datetime import datetime
from collections import defaultdict
import glob

import gradio as gr
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# --- CONFIGURATION ---
load_dotenv()
DB_PATH = "data/slp_vector_db"
EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_OUTPUT_PATH = "generated_case_files/"
FEEDBACK_LOG = "feedback_log.json"
FEEDBACK_CATEGORIES = "feedback_categories.json"
CASES_DB = "cases_database.json"
PROMPTS_DIR = "prompts/"

os.makedirs(DEFAULT_OUTPUT_PATH, exist_ok=True)
os.makedirs(PROMPTS_DIR, exist_ok=True)

# --- PROMPT LOADING ---
def load_prompt(filename: str) -> str:
    """Load prompt template from file."""
    filepath = os.path.join(PROMPTS_DIR, filename)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Warning: Prompt file {filename} not found. Using default.")
        return ""

# --- CONSTANTS ---
ALL_GRADES = ["Pre-K", "Kindergarten", "1st Grade", "2nd Grade", "3rd Grade", "4th Grade", 
              "5th Grade", "6th Grade", "7th Grade", "8th Grade", "9th Grade", "10th Grade", 
              "11th Grade", "12th Grade"]

DISORDER_TYPES = [
    "Speech Sound Disorder", "Articulation Disorders", "Phonological Disorders",
    "Language Disorders", "Receptive Language Disorders", "Expressive Language Disorders",
    "Pragmatics", "Fluency", "Childhood Apraxia of Speech"
]

FREE_MODELS = ["Llama3.2", "Qwen 2.5 7B", "Qwen 2.5 32B", "DeepSeek R1 32B"]
PREMIUM_MODELS = ["GPT-4o", "Gemini 2.5 Pro", "Claude 3 Opus", "Claude 3.5 Sonnet"]

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

# --- PYDANTIC MODELS ---
class BackgroundInfo(BaseModel):
    medical_history: str
    parent_concerns: str
    teacher_concerns: str

class StudentProfile(BaseModel):
    name: str
    age: int
    grade_level: str
    gender: str
    background: BackgroundInfo

class SimuCaseFile(BaseModel):
    student_profile: StudentProfile
    annual_goals: List[str]
    latest_session_notes: List[str]

# --- GLOBAL STATE FOR UNSAVED CASE ---
current_case_data = {
    "content": None,
    "case_id": None,
    "metadata": None
}

# Global state for multiple cases batch
multiple_cases_batch = {
    "cases": [],
    "batch_id": None,
    "timestamp": None
}

# Global state for group session
group_session_data = {
    "session_id": None,
    "members": [],
    "timestamp": None
}

# Generation control flags
generation_control = {
    "should_stop": False
}

MAX_CONDITION_ROWS = 10

# Grouping strategy compatibility - disorder indices
DISORDER_GROUPS = {
    "speech_sounds": [0, 1, 2],  # Speech Sound, Articulation, Phonological
    "language": [3, 4, 5, 6],    # Language, Receptive, Expressive, Pragmatics
    "fluency": [7],              # Fluency
    "cas": [8]                   # Childhood Apraxia of Speech
}

# --- UTILITY FUNCTIONS ---
def load_json(filepath: str, default) -> any:
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return default

def save_json(filepath: str, data):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def get_llm(model_name: str):
    model_id = MODEL_MAP.get(model_name, "llama3.2:latest")
    
    if model_name in FREE_MODELS:
        # For free models, don't use format="json" - let it generate naturally
        return ChatOllama(model=model_id, temperature=0.7)
    elif model_name == "GPT-4o":
        return ChatOpenAI(model=model_id, temperature=0.7).with_structured_output(SimuCaseFile)
    elif model_name == "Gemini 2.5 Pro":
        return ChatGoogleGenerativeAI(model=model_id, temperature=0.7).with_structured_output(SimuCaseFile)
    elif model_name in ["Claude 3 Opus", "Claude 3.5 Sonnet"]:
        return ChatAnthropic(model=model_id, temperature=0.7).with_structured_output(SimuCaseFile)
    else:
        return ChatOllama(model="llama3.2:latest", temperature=0.7)

def parse_ollama_response(response) -> Optional[SimuCaseFile]:
    """Parse free model response - handle both JSON and natural text."""
    try:
        content = response.content if hasattr(response, 'content') else str(response)
        
        # Try JSON parsing first
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
            # If not JSON, try to extract structured information from natural text
            # This is a fallback - create a basic structure
            print("Could not parse as JSON, using text extraction")
            return None
            
    except Exception as e:
        print(f"Parse error: {e}")
        return None

def ai_grammar_check(text: str) -> str:
    """Use AI to check grammar and clarity, returning only the corrected text."""
    try:
        llm = ChatOllama(model="llama3.2:latest", temperature=0.3)
        
        # Load prompt from file
        prompt_template = load_prompt("grammar_check.txt")
        if not prompt_template:
            prompt_template = "Improve this text: {feedback_text}\n\nCorrected version:"
        
        prompt = prompt_template.format(feedback_text=text)
        
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else text
        
        # Remove any meta-commentary lines
        lines = [line for line in content.split('\n') 
                if not any(x in line.lower() for x in ['i corrected', 'i changed', 'i improved', 'here is', 'the corrected'])]
        return '\n'.join(lines).strip()
    except:
        return text

def analyze_feedback_category(text: str, existing_categories: List[str]) -> tuple:
    """Generate descriptive category from feedback text."""
    try:
        llm = ChatOllama(model="llama3.2:latest", temperature=0.3)
        
        # Load prompt from file
        prompt_template = load_prompt("feedback_category.txt")
        if not prompt_template:
            prompt_template = "Categorize this feedback: {feedback_text}\nExisting: {existing_categories}\nCategory:"
        
        prompt = prompt_template.format(
            feedback_text=text,
            existing_categories=', '.join(existing_categories) if existing_categories else 'None'
        )
        
        response = llm.invoke(prompt)
        category = response.content.strip() if hasattr(response, 'content') else "General Feedback"
        
        # Check if matches existing
        is_new = category not in existing_categories
        
        return category, is_new
    except:
        return "General Feedback", True

def save_case_to_db(case_data: dict):
    """Save case metadata to database."""
    db = load_json(CASES_DB, {"cases": []})
    case_data["id"] = f"case_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(db['cases'])}"
    db["cases"].append(case_data)
    save_json(CASES_DB, db)
    return case_data["id"]

def save_case_file(content: str, filepath: str) -> str:
    """Save case file to specified path."""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    return filepath

# --- GROUP COMPATIBILITY FUNCTIONS ---
def check_grade_compatibility(grades: List[str]) -> tuple:
    """Check if grades are within 2-level difference."""
    grade_nums = []
    for grade in grades:
        if grade in ["Pre-K", "Kindergarten"]:
            grade_nums.append(0 if grade == "Pre-K" else 1)
        else:
            try:
                num = int(grade.split()[0].replace("st", "").replace("nd", "").replace("rd", "").replace("th", ""))
                grade_nums.append(num + 1)  # Adjust for K=1
            except:
                grade_nums.append(1)
    
    if not grade_nums:
        return False, "No grades specified"
    
    grade_range = max(grade_nums) - min(grade_nums)
    if grade_range <= 2:
        return True, f"Grade range: {grade_range} levels (✓ Compatible)"
    else:
        return False, f"Grade range: {grade_range} levels (✗ Must be within 2 levels)"

def check_disorder_compatibility(disorders_list: List[List[str]]) -> tuple:
    """Check if disorder combinations are compatible per grouping strategies."""
    # Get disorder indices
    disorder_indices = []
    for disorders in disorders_list:
        indices = [DISORDER_TYPES.index(d) for d in disorders if d in DISORDER_TYPES]
        disorder_indices.append(set(indices))
    
    # Check compatibility rules
    # Rule 1: Speech sounds (0,1,2) can group with CAS (8)
    # Rule 2: Language (3,4,5,6) can group together
    # Rule 3: Fluency (7) can group with language or CAS
    # Rule 4: Speech sounds can group with language
    
    all_indices = set()
    for indices in disorder_indices:
        all_indices.update(indices)
    
    # Check valid combinations
    has_speech = any(i in all_indices for i in [0, 1, 2])
    has_language = any(i in all_indices for i in [3, 4, 5, 6])
    has_fluency = 7 in all_indices
    has_cas = 8 in all_indices
    
    # Valid combinations:
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

def parse_markdown_case(content: str) -> Optional[Dict]:
    """Parse a markdown case file and extract structured information."""
    try:
        # Extract case ID and name from header (e.g., "## S005: Camila Ramos")
        id_match = re.search(r'##\s+([A-Z0-9]+):\s+(.+)', content)
        if not id_match:
            return None

        case_id = id_match.group(1)
        name = id_match.group(2).strip()

        # Extract metadata line (Grade, Age, Gender, Disorders)
        meta_match = re.search(r'\*\*Grade:\*\*\s+([^\|]+)\s+\|\s+\*\*Age:\*\*\s+(\d+)\s+\|\s+\*\*Gender:\*\*\s+(\w+)', content)
        if not meta_match:
            return None

        grade = meta_match.group(1).strip()
        age = int(meta_match.group(2))
        gender = meta_match.group(3).strip()

        # Extract disorders
        disorder_match = re.search(r'\*\*Disorders:\*\*\s+([^\n]+)', content)
        disorders = disorder_match.group(1).strip() if disorder_match else ""

        # Extract model
        model_match = re.search(r'\*\*Model:\*\*\s+([^\n]+)', content)
        model = model_match.group(1).strip() if model_match else "Unknown"

        # Extract special characteristics
        char_match = re.search(r'\*\*Special Characteristics:\*\*\s+([^\n]+)', content)
        characteristics = char_match.group(1).strip() if char_match else ""

        # Extract Background section
        background_section = ""
        bg_match = re.search(r'###\s+Background\s*\n((?:- \*\*[^:]+:\*\*[^\n]+\n?)+)', content)
        if bg_match:
            background_section = bg_match.group(1).strip()

        # Extract Annual IEP Goals
        goals = []
        goals_match = re.search(r'###\s+Annual IEP Goals\s*\n((?:\d+\..+\n?)+)', content)
        if goals_match:
            goals_text = goals_match.group(1).strip()
            goals = [g.strip() for g in re.findall(r'\d+\.\s+(.+)', goals_text)]

        # Extract Latest Session Notes
        session_notes = []
        notes_match = re.search(r'###\s+Latest Session Notes\s*\n((?:\*\*Session \d+:\*\*.+(?:\n(?!\*\*Session|\n---|\n##).+)*)+)', content)
        if notes_match:
            notes_text = notes_match.group(1).strip()
            session_notes = re.findall(r'\*\*Session \d+:\*\*\s+(.+?)(?=\*\*Session \d+:|\Z)', notes_text, re.DOTALL)
            session_notes = [note.strip() for note in session_notes]

        return {
            "case_id": case_id,
            "name": name,
            "grade": grade,
            "age": age,
            "gender": gender,
            "disorders": disorders,
            "model": model,
            "characteristics": characteristics,
            "background": background_section,
            "annual_goals": goals,
            "session_notes": session_notes
        }
    except Exception as e:
        print(f"Error parsing markdown case: {e}")
        return None

def search_existing_cases_in_folder(grades: List[str], disorders_list: List[List[str]]) -> List[Dict]:
    """Search for existing cases in generated_case_files folder matching criteria."""
    matching_cases = []

    # Get all markdown files in the generated_case_files folder
    md_files = glob.glob(os.path.join(DEFAULT_OUTPUT_PATH, "*.md"))

    for filepath in md_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # Split multiple cases if batch file
            case_sections = re.split(r'\n---\n', content)

            for section in case_sections:
                if not section.strip():
                    continue

                case_data = parse_markdown_case(section)
                if not case_data:
                    continue

                # Check if case matches any member criteria
                for i, (grade, disorders) in enumerate(zip(grades, disorders_list)):
                    # Match grade and any disorder
                    if case_data["grade"] == grade:
                        for disorder in disorders:
                            if disorder.lower() in case_data["disorders"].lower():
                                case_data["member_index"] = i
                                case_data["filepath"] = filepath
                                matching_cases.append(case_data)
                                break
                        if "member_index" in case_data:
                            break
        except Exception as e:
            print(f"Error reading file {filepath}: {e}")
            continue

    return matching_cases

def search_existing_cases(grades: List[str], disorders_list: List[List[str]]) -> List[Dict]:
    """Search for existing cases matching criteria (legacy database search)."""
    db = load_json(CASES_DB, {"cases": []})
    matching_cases = []

    for case in db["cases"]:
        case_grade = case.get("grade", "")
        case_disorders = case.get("disorders", [])

        # Check if case matches any member criteria
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

# --- CHAT PARSING FOR MULTIPLE CASES ---
def parse_complex_request(request: str) -> tuple:
    """Parse natural language request into generation tasks with improved accuracy."""
    try:
        llm = ChatOllama(model="llama3.2:latest", temperature=0.1)
        
        prompt = f"""Parse this natural language request for generating SLP case files.

Request: "{request}"

Available options:
Grades: Pre-K, Kindergarten, 1st Grade, 2nd Grade, 3rd Grade, 4th Grade, 5th Grade, 6th Grade, 7th Grade, 8th Grade, 9th Grade, 10th Grade, 11th Grade, 12th Grade

Disorders: Speech Sound Disorder, Articulation Disorders, Phonological Disorders, Language Disorders, Receptive Language Disorders, Expressive Language Disorders, Pragmatics, Fluency, Childhood Apraxia of Speech

Models: Llama3.2, Qwen 2.5 7B, Qwen 2.5 32B, DeepSeek R1 32B, GPT-4o, Gemini 2.5 Pro, Claude 3 Opus, Claude 3.5 Sonnet

Instructions:
1. Identify total number of students (look for numbers like "5", "20", "three", etc.)
2. Identify which model to use (if says "for all" or "use X", apply to all tasks)
3. Break down grade distributions (e.g., "5 pre-k, rest kindergarten" means calculate remaining)
4. Identify disorders for each group
5. Extract specific characteristics for individual students

Format each task as:
TASK [number]: [count] student(s), Grade: [grade], Disorders: [disorders], Model: [model], Characteristics: [specific notes if any]

Example 1:
Request: "generate 20 students with articulation, using GPT-4o, 5 are pre-k, rest kindergarten"
Output:
TASK 1: 5 students, Grade: Pre-K, Disorders: Articulation Disorders, Model: GPT-4o, Characteristics: none
TASK 2: 15 students, Grade: Kindergarten, Disorders: Articulation Disorders, Model: GPT-4o, Characteristics: none

Example 2:
Request: "Use Gemini for all cases. Generate 3 students: one with phonological disorders having final consonant deletion, one 1st grade with articulation /s/ issues, one 2nd grade with fluency"
Output:
TASK 1: 1 student, Grade: 1st Grade, Disorders: Phonological Disorders, Model: Gemini 2.5 Pro, Characteristics: has final consonant deletion
TASK 2: 1 student, Grade: 1st Grade, Disorders: Articulation Disorders, Model: Gemini 2.5 Pro, Characteristics: has difficulties with /s/
TASK 3: 1 student, Grade: 2nd Grade, Disorders: Fluency, Model: Gemini 2.5 Pro, Characteristics: none

Now parse the request above:"""
        
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else ""
        
        # Parse the response into structured tasks
        tasks = []
        global_settings = {"model": None}
        
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('TASK'):
                try:
                    # Extract task details
                    parts = line.split(':', 1)[1].strip() if ':' in line else line
                    
                    # Parse components
                    task = {
                        "count": 1,
                        "grade": "1st Grade",
                        "disorders": [],
                        "model": "Llama3.2",
                        "characteristics": ""
                    }
                    
                    # Extract count
                    if "student" in parts.lower():
                        count_part = parts.split("student")[0].strip()
                        try:
                            task["count"] = int(count_part.split()[0])
                        except:
                            task["count"] = 1
                    
                    # Extract grade
                    for grade in ALL_GRADES:
                        if grade.lower() in parts.lower():
                            task["grade"] = grade
                            break
                    
                    # Extract disorders
                    for disorder in DISORDER_TYPES:
                        if disorder.lower() in parts.lower():
                            task["disorders"].append(disorder)
                    
                    # Extract model
                    all_models = FREE_MODELS + PREMIUM_MODELS
                    for model in all_models:
                        if model.lower() in parts.lower():
                            task["model"] = model
                            break
                    
                    # Extract characteristics
                    if "Characteristics:" in parts:
                        chars = parts.split("Characteristics:")[1].strip()
                        if chars.lower() != "none":
                            task["characteristics"] = chars
                    
                    tasks.append(task)
                    
                except Exception as e:
                    print(f"Error parsing task line: {e}")
                    continue
        
        # Build confirmation message
        if tasks:
            confirmation = f"### 📋 Parsed {len(tasks)} Task(s):\n\n"
            total_students = sum(t["count"] for t in tasks)
            confirmation += f"**Total Students:** {total_students}\n\n"
            
            for i, task in enumerate(tasks, 1):
                confirmation += f"**Task {i}:**\n"
                confirmation += f"- Count: {task['count']} student(s)\n"
                confirmation += f"- Grade: {task['grade']}\n"
                confirmation += f"- Disorders: {', '.join(task['disorders']) if task['disorders'] else 'None specified'}\n"
                confirmation += f"- Model: {task['model']}\n"
                if task['characteristics']:
                    confirmation += f"- Special Characteristics: {task['characteristics']}\n"
                confirmation += "\n"
            
            confirmation += "\n✓ Review above. Adjust manually in rows below if needed, then click Generate."
            
            return tasks, confirmation
        else:
            return [], "❌ Could not parse request. Please use manual configuration below."
            
    except Exception as e:
        print(f"Parse error: {e}")
        return [], "❌ Error parsing request. Please use manual configuration below."

# --- GENERATION FUNCTIONS ---
def generate_single_case(grade: str, disorders: List[str], model: str, 
                        population_spec: str, references):
    
    generation_control["should_stop"] = False
    
    yield "🔄 Initializing vector database...", None, gr.update(interactive=False, variant="secondary"), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
    
    if generation_control["should_stop"]:
        yield "⛔ Generation stopped", None, gr.update(interactive=True, variant="primary"), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
        return
    
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    
    yield "🔄 Building generation prompt...", None, gr.update(interactive=False, variant="secondary"), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
    
    # Load prompt templates from files
    if model in FREE_MODELS:
        template = load_prompt("single_case_free_model.txt")
        if not template:  # Fallback if file not found
            template = "You are an expert SLP. Create a case file for a {grade} student with {disorders}."
    else:
        template = load_prompt("single_case_premium_model.txt")
        if not template:
            template = "Generate a case file as JSON for {grade} with {disorders}."

    prompt = ChatPromptTemplate.from_template(template)
    disorder_string = ", ".join(disorders)
    
    yield f"🔄 Generating case with {model}...", None, gr.update(interactive=False, variant="secondary"), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
    
    if generation_control["should_stop"]:
        yield "⛔ Generation stopped", None, gr.update(interactive=True, variant="primary"), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
        return
    
    llm = get_llm(model)
    rag_chain = {
        "context": retriever,
        "question": RunnablePassthrough(),
        "population_spec": lambda x: population_spec if population_spec else "general population",
        "disorders": lambda x: disorder_string,
        "grade": lambda x: grade
    } | prompt | llm
    
    response = rag_chain.invoke(f"Generate case for {grade} with {disorder_string}")
    
    yield "🔄 Formatting output...", None, gr.update(interactive=False, variant="secondary"), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
    
    if model in FREE_MODELS:
        # For free models, use the natural text output directly
        content = response.content if hasattr(response, 'content') else str(response)
        
        # Format it nicely
        output = f"""# Generated Case File

**Model Used:** {model}
**Grade:** {grade}
**Disorders:** {disorder_string}

---

{content}

---
"""
        # Store for later saving
        current_case_data["content"] = output
        current_case_data["metadata"] = {
            "type": "single",
            "disorders": disorders,
            "grade": grade,
            "model": model,
            "timestamp": datetime.now().strftime('%Y%m%d_%H%M%S')
        }
        
    else:
        # Premium models with structured output
        case_file = response
        
        profile = case_file.student_profile
        output = f"""# Case File: {profile.name}

**Grade:** {profile.grade_level} | **Age:** {profile.age} | **Gender:** {profile.gender}
**Disorders:** {disorder_string}
**Model Used:** {model}

## Background Information

### Medical History
{profile.background.medical_history}

### Parent Concerns
{profile.background.parent_concerns}

### Teacher Concerns
{profile.background.teacher_concerns}

## Annual IEP Goals

"""
        for i, goal in enumerate(case_file.annual_goals, 1):
            output += f"{i}. {goal}\n"
        
        output += "\n## Latest Session Notes\n\n"
        for i, note in enumerate(case_file.latest_session_notes, 1):
            output += f"### Session {i}\n{note}\n\n"
        
        # Store for later saving
        current_case_data["content"] = output
        current_case_data["metadata"] = {
            "type": "single",
            "profile": profile.model_dump(),
            "disorders": disorders,
            "grade": grade,
            "model": model,
            "timestamp": datetime.now().strftime('%Y%m%d_%H%M%S')
        }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suggested_filename = f"case_{timestamp}.md"
    
    yield output, suggested_filename, gr.update(interactive=True, variant="primary"), gr.update(visible=True, value=f"{DEFAULT_OUTPUT_PATH}{suggested_filename}"), gr.update(visible=False), gr.update(visible=True)

# --- MULTIPLE CASES GENERATION ---
def generate_multiple_cases(tasks: List[Dict], save_path: str, use_custom_id: bool = False, 
                           id_prefix: str = "S", id_start: int = 1):
    """Generate multiple cases based on task list."""
    
    generation_control["should_stop"] = False
    
    batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    all_cases = []
    case_counter = id_start
    
    # Expand tasks based on count
    expanded_tasks = []
    for task in tasks:
        count = task.get("count", 1)
        for _ in range(count):
            expanded_tasks.append(task.copy())
    
    total_tasks = len(expanded_tasks)
    
    yield (f"🔄 Starting batch generation: {total_tasks} cases", 
           None, 
           ["Whole Batch"], 
           gr.update(interactive=False))
    
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    
    combined_output = f"# Multiple Cases Generation\n**Batch ID:** {batch_id}\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n**Total Cases:** {total_tasks}\n\n"
    
    for idx, task in enumerate(expanded_tasks):
        # Check stop flag
        if generation_control["should_stop"]:
            combined_output += f"\n\n⛔ **Generation stopped by user after {idx} cases**"
            yield (combined_output, 
                   None, 
                   ["Whole Batch"] + [c['display_id'] for c in all_cases],
                   gr.update(interactive=True, variant="primary"))
            return
        # Generate case ID
        if use_custom_id:
            case_display_id = f"{id_prefix}{case_counter:03d}"
            case_counter += 1
        else:
            case_display_id = f"Case {idx+1}"
        
        yield (f"{combined_output}\n\n🔄 Generating {case_display_id}: {task['grade']} - {', '.join(task['disorders'])}", 
               None, 
               ["Whole Batch"], 
               gr.update(interactive=False))
        
        # Get appropriate prompt
        if task['model'] in FREE_MODELS:
            template = load_prompt("single_case_free_model.txt")
            if not template:
                template = "Generate a case for {grade} with {disorders}. {characteristics}"
        else:
            template = load_prompt("single_case_premium_model.txt")
            if not template:
                template = "Generate JSON case for {grade} with {disorders}. {characteristics}"
        
        # Add characteristics to prompt
        characteristics_note = f"\n\nSpecific characteristics for this student: {task.get('characteristics', 'None')}" if task.get('characteristics') else ""
        template = template.replace("{question}", "{question}" + characteristics_note)
        
        prompt = ChatPromptTemplate.from_template(template)
        disorder_string = ", ".join(task['disorders']) if task['disorders'] else "Speech Sound Disorder"
        
        llm = get_llm(task['model'])
        
        question_text = f"Generate case for {task['grade']} with {disorder_string}."
        if task.get('characteristics'):
            question_text += f" Student characteristics: {task['characteristics']}"
        
        rag_chain = {
            "context": retriever,
            "question": RunnablePassthrough(),
            "population_spec": lambda x: "",
            "disorders": lambda x: disorder_string,
            "grade": lambda x: task['grade'],
            "characteristics": lambda x: task.get('characteristics', '')
        } | prompt | llm
        
        response = rag_chain.invoke(question_text)
        
        # Format output
        if task['model'] in FREE_MODELS:
            content = response.content if hasattr(response, 'content') else str(response)
            case_output = f"""
---

## {case_display_id}

**Grade:** {task['grade']}
**Disorders:** {disorder_string}
**Model:** {task['model']}
{f"**Special Characteristics:** {task['characteristics']}" if task.get('characteristics') else ""}

{content}

"""
        else:
            case_file = response
            profile = case_file.student_profile
            case_output = f"""
---

## {case_display_id}: {profile.name}

**Grade:** {profile.grade_level} | **Age:** {profile.age} | **Gender:** {profile.gender}
**Disorders:** {disorder_string}
**Model:** {task['model']}
{f"**Special Characteristics:** {task['characteristics']}" if task.get('characteristics') else ""}

### Background
- **Medical History:** {profile.background.medical_history}
- **Parent Concerns:** {profile.background.parent_concerns}
- **Teacher Concerns:** {profile.background.teacher_concerns}

### Annual IEP Goals
""" + "\n".join([f"{i+1}. {goal}" for i, goal in enumerate(case_file.annual_goals)])
            case_output += "\n\n### Latest Session Notes\n" + "\n".join([f"**Session {i+1}:** {note}" for i, note in enumerate(case_file.latest_session_notes)])
            case_output += "\n"
        
        case_id = f"{batch_id}_{case_display_id.replace(' ', '_')}"
        all_cases.append({
            "id": case_id,
            "display_id": case_display_id,
            "content": case_output,
            "metadata": task
        })
        
        combined_output += case_output
        
        # Update progress
        yield (combined_output, 
               None, 
               ["Whole Batch"] + [c['display_id'] for c in all_cases],
               gr.update(interactive=False))
    
    # Store in global state
    multiple_cases_batch["cases"] = all_cases
    multiple_cases_batch["batch_id"] = batch_id
    multiple_cases_batch["timestamp"] = datetime.now().isoformat()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suggested_filename = f"{save_path}multiple_cases_{timestamp}.md"
    
    # Create dropdown choices for feedback
    case_choices = ["Whole Batch"] + [f"{case['display_id']}: {case['metadata']['grade']} - {', '.join(case['metadata']['disorders'])}" for case in all_cases]
    
    final_output = combined_output + f"\n\n---\n\n✅ **Generation Complete!** {len(all_cases)} cases created.\n**Batch ID:** {batch_id}"
    
    yield (final_output, 
           suggested_filename, 
           case_choices, 
           gr.update(interactive=True, variant="primary"))

# --- GROUP SESSION GENERATION ---
def generate_group_session(group_size: int, grades: List[str], disorders_list: List[List[str]],
                          model: str, search_first: bool = True, use_custom_ids: bool = False,
                          id_prefix: str = "S", id_start: int = 1):
    """Generate or search for group session cases with full profiles."""

    generation_control["should_stop"] = False

    session_id = f"group_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    id_counter = id_start

    yield (f"🔄 Initializing group session...",
           None,
           gr.update(interactive=False),
           gr.update(visible=True))

    # Check compatibility
    grade_compat, grade_msg = check_grade_compatibility(grades[:group_size])
    disorder_compat, disorder_msg = check_disorder_compatibility(disorders_list[:group_size])

    compat_check = f"""### Compatibility Check

**Grade Levels:** {grade_msg}
**Disorder Combinations:** {disorder_msg}

"""

    if not grade_compat or not disorder_compat:
        final_msg = compat_check + "\n⚠️ **Warning:** Group configuration may not be optimal per grouping strategies.\nProceed anyway or adjust configuration.\n"
        yield (final_msg, None, gr.update(interactive=True), gr.update(visible=True))

    # Search for existing cases in folder
    existing_cases = []
    if search_first:
        yield (compat_check + "🔍 Searching existing cases in folder...",
               None,
               gr.update(interactive=False),
               gr.update(visible=True))

        existing_cases = search_existing_cases_in_folder(grades[:group_size], disorders_list[:group_size])

        if len(existing_cases) >= group_size:
            yield (compat_check + f"✅ Found {len(existing_cases)} matching cases!\nUsing existing cases for group session...",
                   None,
                   gr.update(interactive=False),
                   gr.update(visible=True))
        else:
            yield (compat_check + f"⚠️ Only found {len(existing_cases)} matching cases. Need {group_size}.\nWill use existing cases where available...",
                   None,
                   gr.update(interactive=False),
                   gr.update(visible=True))

    # Generate group session output
    output = f"# Group Session\n**Session ID:** {session_id}\n**Date:** {datetime.now().strftime('%Y-%m-%d')}\n**Group Size:** {group_size}\n\n"
    output += compat_check + "\n---\n\n"

    # Process each member
    members = []
    for i in range(group_size):
        if generation_control["should_stop"]:
            yield (output + "\n\n⛔ **Generation stopped by user**",
                   None,
                   gr.update(interactive=True, variant="primary"),
                   gr.update(visible=True))
            return

        # Generate student ID
        if use_custom_ids:
            student_id = f"{id_prefix}{id_counter:03d}"
            id_counter += 1
        else:
            student_id = f"Member {i+1}"

        yield (output + f"\n🔄 Processing {student_id}...",
               None,
               gr.update(interactive=False),
               gr.update(visible=True))

        # Find matching existing case for this member
        matching_case = None
        for case in existing_cases:
            if case.get("member_index") == i:
                matching_case = case
                existing_cases.remove(case)  # Remove to avoid reuse
                break

        if matching_case:
            # Use existing case
            member_output = f"\n## {student_id}: {matching_case['name']}\n\n"
            member_output += f"**Grade:** {matching_case['grade']} | **Age:** {matching_case['age']} | **Gender:** {matching_case['gender']}\n"
            member_output += f"**Disorders:** {matching_case['disorders']}\n"
            if matching_case.get('characteristics'):
                member_output += f"**Special Characteristics:** {matching_case['characteristics']}\n"
            member_output += "\n"

            # Add background
            member_output += f"### Background\n{matching_case['background']}\n\n"

            # Add annual goals
            member_output += f"### Annual IEP Goals\n"
            for idx, goal in enumerate(matching_case['annual_goals'], 1):
                member_output += f"{idx}. {goal}\n"
            member_output += "\n"

            # Add latest session notes
            member_output += f"### Latest Session Notes\n"
            for idx, note in enumerate(matching_case['session_notes'], 1):
                member_output += f"**Session {idx}:** {note}\n"
            member_output += "\n"

            # Generate group goal using AI based on session notes
            yield (output + f"\n🔄 Generating group goal for {student_id} using {model}...",
                   None,
                   gr.update(interactive=False),
                   gr.update(visible=True))

            session_notes_summary = "\n".join([f"- {note[:200]}..." if len(note) > 200 else f"- {note}"
                                              for note in matching_case['session_notes']])

            try:
                llm = ChatOllama(model=MODEL_MAP.get(model, "llama3.2:latest"), temperature=0.7)
                goal_prompt = f"""Based on this student's latest session notes, generate ONE specific, measurable group therapy goal that would be appropriate for a small group setting (2-4 students).

Student Profile:
- Grade: {matching_case['grade']}
- Disorders: {matching_case['disorders']}

Latest Session Notes:
{session_notes_summary}

Generate a single group therapy goal that:
1. Builds on the student's current progress shown in the session notes
2. Is appropriate for a group setting with peer interaction
3. Is specific and measurable
4. Aligns with the student's disorders

Provide ONLY the goal text, no explanations or additional commentary."""

                response = llm.invoke(goal_prompt)
                group_goal = response.content.strip() if hasattr(response, 'content') else "Participate in group activities to improve communication skills."

                # Clean up any meta-commentary
                if ":" in group_goal and group_goal.split(":")[0].lower() in ["goal", "group goal", "therapy goal"]:
                    group_goal = group_goal.split(":", 1)[1].strip()

            except Exception as e:
                print(f"Error generating group goal: {e}")
                group_goal = f"Participate in group therapy activities to improve {matching_case['disorders'].split(',')[0].lower()} skills in a collaborative setting."

            member_output += f"### Group Therapy Goal (AI-Generated)\n{group_goal}\n\n"

            members.append({
                "member_num": i+1,
                "student_id": student_id,
                "name": matching_case['name'],
                "grade": matching_case['grade'],
                "disorders": matching_case['disorders'],
                "output": member_output,
                "group_goal": group_goal
            })
        else:
            # No existing case found - create placeholder
            member_output = f"\n## {student_id}\n\n"
            member_output += f"**Grade:** {grades[i]}\n"
            member_output += f"**Disorders:** {', '.join(disorders_list[i])}\n\n"
            member_output += f"### Background\n"
            member_output += f"- **Medical History:** No existing case found for this profile\n"
            member_output += f"- **Parent Concerns:** Generate a new case to populate this information\n"
            member_output += f"- **Teacher Concerns:** Generate a new case to populate this information\n\n"
            member_output += f"### Annual IEP Goals\n"
            member_output += f"1. No existing case found - generate a new case for full details\n\n"
            member_output += f"### Latest Session Notes\n"
            member_output += f"**Note:** No existing case found. Generate a new case to populate session notes.\n\n"
            member_output += f"### Group Therapy Goal\n"
            member_output += f"Generate a new case to create appropriate group therapy goals.\n\n"

            members.append({
                "member_num": i+1,
                "student_id": student_id,
                "grade": grades[i],
                "disorders": ', '.join(disorders_list[i]),
                "output": member_output,
                "group_goal": "To be determined after case generation"
            })

        output += member_output

    # Add group session plan summary
    output += "\n---\n\n## Group Session Summary\n\n"
    output += f"### Group Composition\n"
    for member in members:
        output += f"- **{member['student_id']}**: {member['grade']} - {member['disorders']}\n"

    output += "\n### Individual Group Goals\n"
    for member in members:
        output += f"- **{member['student_id']}**: {member['group_goal']}\n"

    output += "\n### Recommended Group Activities\n"
    output += "1. **Turn-taking practice** - Supports all members' communication and social interaction goals\n"
    output += "2. **Collaborative problem-solving tasks** - Targets pragmatic language and executive function skills\n"
    output += "3. **Peer modeling and feedback** - Members support each other's specific articulation/language goals\n"
    output += "4. **Structured conversation practice** - Builds on individual session notes progress in group context\n\n"

    # Store in global state
    group_session_data["session_id"] = session_id
    group_session_data["members"] = members
    group_session_data["timestamp"] = datetime.now().isoformat()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suggested_filename = f"{DEFAULT_OUTPUT_PATH}group_session_{timestamp}.md"

    output += f"\n\n---\n✅ **Group Session Plan Complete**\n**Session ID:** {session_id}"

    yield (output,
           suggested_filename,
           gr.update(interactive=True, variant="primary"),
           gr.update(visible=False))

# --- SAVE FUNCTIONS ---
def handle_save(filepath: str) -> str:
    """Save the current case to specified path."""
    if not current_case_data["content"]:
        return "❌ No case to save. Generate a case first."
    
    try:
        save_case_file(current_case_data["content"], filepath)
        case_id = save_case_to_db({**current_case_data["metadata"], "filepath": filepath})
        current_case_data["case_id"] = case_id
        return f"✅ Case saved successfully!\n**Path:** `{filepath}`\n**Case ID:** {case_id}"
    except Exception as e:
        return f"❌ Error saving file: {str(e)}"

# --- FEEDBACK FUNCTIONS ---
def submit_feedback(case_id: str, ratings: Dict[str, int], category: str, 
                   detailed_feedback: str) -> tuple:
    
    if not case_id:
        return "❌ Error: Save the case first before submitting feedback.", gr.update(), gr.update(value=3), gr.update(value=3), gr.update(value=3), gr.update(value=3), gr.update(value=3), gr.update(value=""), gr.update(value="General")
    
    feedback_log = load_json(FEEDBACK_LOG, [])
    categories_data = load_json(FEEDBACK_CATEGORIES, {"categories": [], "descriptions": {}})
    
    # Analyze category if "Other" selected
    if category == "Other" and detailed_feedback.strip():
        category, is_new = analyze_feedback_category(
            detailed_feedback, 
            categories_data.get("categories", [])
        )
        
        if is_new:
            if "categories" not in categories_data:
                categories_data = {"categories": [], "descriptions": {}}
            categories_data["categories"].append(category)
            categories_data["descriptions"][category] = detailed_feedback[:100]
            save_json(FEEDBACK_CATEGORIES, categories_data)
    else:
        is_new = False
    
    entry = {
        "timestamp": datetime.now().isoformat(),
        "case_id": case_id,
        "ratings": ratings,
        "category": category,
        "detailed_feedback": detailed_feedback
    }
    
    feedback_log.append(entry)
    save_json(FEEDBACK_LOG, feedback_log)
    
    message = f"""✅ Feedback saved successfully!

**Case ID:** {case_id}
**Category:** {category} {'(NEW)' if is_new else ''}
**Saved to:** `{FEEDBACK_LOG}`"""
    
    updated_choices = ["General", "Other"] + categories_data.get("categories", [])
    
    # Reset all feedback fields
    return (message, gr.update(choices=updated_choices), 
            gr.update(value=3), gr.update(value=3), gr.update(value=3), 
            gr.update(value=3), gr.update(value=3), 
            gr.update(value=""), gr.update(value="General"))

# --- GRADIO UI ---
def create_ui():
    with gr.Blocks(theme=gr.themes.Soft(), title="SLP SimuCase Generator", css="""
        .title {text-align: center; margin: 20px 0;}
        .cover-buttons {max-width: 600px; margin: 0 auto;}
        .left-panel {padding: 20px; background: #f8f9fa; border-radius: 10px;}
        .right-panel {padding: 20px; background: #f0f0f0; border-radius: 10px;}
        .save-section {padding: 15px; background: #e7f3ff; border-radius: 8px; margin: 15px 0;}
    """) as app:
        
        # STATE VARIABLES
        current_mode = gr.State("cover")
        generated_filename = gr.State(None)
        
        # COVER PAGE
        with gr.Column(visible=True, elem_classes="cover-buttons") as cover_page:
            gr.Markdown("# SLP SimuCase Generator", elem_classes="title")
            gr.Markdown("### Professional Case File Generation System", elem_classes="title")
            
            gr.Markdown("")
            btn_single = gr.Button("Generate Single Case", size="lg", variant="primary")
            btn_multiple = gr.Button("Generate Multiple Cases", size="lg", variant="primary")
            btn_group = gr.Button("Generate Group Session", size="lg", variant="primary")
        
        # FUNCTION 1: SINGLE CASE
        with gr.Column(visible=False) as single_case_page:
            with gr.Row():
                gr.Markdown("# Generate Single Case")
                back_btn_single = gr.Button("← Back", size="sm")
            
            # LEFT AND RIGHT PANELS
            with gr.Row(equal_height=True):
                with gr.Column(scale=1, elem_classes="left-panel"):
                    gr.Markdown("### Generation Parameters")
                    
                    grade_single = gr.Dropdown(choices=ALL_GRADES, label="Grade Level", value="1st Grade")
                    model_single = gr.Dropdown(choices=FREE_MODELS + PREMIUM_MODELS, label="AI Model", value="Llama3.2")
                    disorders_single = gr.Dropdown(choices=DISORDER_TYPES, label="Disorders", multiselect=True, value=["Articulation Disorders"])
                    population_spec_single = gr.Textbox(label="Population Characteristics", placeholder="e.g., second language learner", lines=2)

                    generate_btn_single = gr.Button("Generate", size="sm", variant="primary")
                    stop_btn_single = gr.Button("⛔ Stop", size="sm", variant="stop", visible=False)

                with gr.Column(scale=1, elem_classes="right-panel"):
                    gr.Markdown("### Advanced Options")
                    reference_files_single = gr.File(label="Upload References", file_count="multiple")
            
            # OUTPUT AT BOTTOM
            gr.Markdown("### Generated Case File")
            output_single = gr.Markdown()
            
            # SAVE SECTION
            with gr.Group(visible=False, elem_classes="save-section") as save_section:
                gr.Markdown("### 💾 Save Case File")
                
                save_path_display = gr.Textbox(
                    label="Save Path",
                    interactive=False,
                    value=""
                )
                
                with gr.Row():
                    save_btn = gr.Button("💾 Save", variant="primary", size="sm")
                    save_as_btn = gr.DownloadButton("📥 Save As (Download)", size="sm")
                
                save_status = gr.Markdown("")
            
            # FEEDBACK SECTION
            with gr.Accordion("Provide Feedback", open=False):
                gr.Markdown("### Evaluate This Case")
                
                with gr.Row():
                    rating_clinical = gr.Slider(1, 5, value=3, step=1, label="Clinical Accuracy")
                    rating_age = gr.Slider(1, 5, value=3, step=1, label="Age Appropriate")
                    rating_goals = gr.Slider(1, 5, value=3, step=1, label="Goal Quality")
                
                with gr.Row():
                    rating_notes = gr.Slider(1, 5, value=3, step=1, label="Session Notes")
                    rating_background = gr.Slider(1, 5, value=3, step=1, label="Background")
                
                categories_list = load_json(FEEDBACK_CATEGORIES, {"categories": []}).get("categories", [])
                feedback_cat_single = gr.Dropdown(choices=["General", "Other"] + categories_list, label="Category", value="General")
                
                detailed_feedback_single = gr.Textbox(label="Detailed Feedback", lines=4, visible=False, placeholder="Provide specific feedback...")
                grammar_check_btn = gr.Button("AI Grammar Check", size="sm", visible=False)
                
                submit_feedback_btn_single = gr.Button("Submit Feedback", variant="secondary")
                feedback_status_single = gr.Markdown("")
        
        # FUNCTION 2 & 3 (placeholders)
        with gr.Column(visible=False) as multiple_cases_page:
            with gr.Row():
                gr.Markdown("# Generate Multiple Cases")
                back_btn_multiple = gr.Button("← Back", size="sm")
            
            # ENHANCED CHAT BOX
            with gr.Group(elem_classes="left-panel"):
                gr.Markdown("### 💬 Natural Language Request")
                chat_request = gr.Textbox(
                    label="Describe what you want to generate",
                    placeholder='Example: "generate 20 students with articulation disorders and phonological disorders, using GPT-4o, 5 of them are pre-k, and the rest kindergarten and 1st grade"',
                    lines=4
                )
                with gr.Row():
                    parse_btn = gr.Button("🔍 Parse Request", size="sm", variant="secondary")
                    clear_chat_btn = gr.Button("Clear", size="sm")
                
                parsed_output = gr.Markdown(label="Parsed Tasks", visible=False)
            
            # MANUAL CONFIGURATION
            with gr.Accordion("Manual Configuration (Alternative)", open=True):
                gr.Markdown("### Condition Rows")
                
                # Optional Case ID Prefix
                with gr.Row():
                    use_custom_ids = gr.Checkbox(label="Use Custom Case IDs", value=False)
                    custom_id_prefix = gr.Textbox(
                        label="ID Prefix",
                        placeholder="e.g., S001, PT-",
                        value="S",
                        visible=False,
                        scale=1
                    )
                    custom_id_start = gr.Number(
                        label="Start Number",
                        value=1,
                        minimum=1,
                        step=1,
                        visible=False,
                        scale=1
                    )
                
                visible_row_count = gr.State(1)
                
                # Create dynamic rows
                condition_rows = []
                condition_row_components = []
                
                for i in range(MAX_CONDITION_ROWS):
                    with gr.Row(visible=(i==0)) as row:
                        grade_multi = gr.Dropdown(choices=ALL_GRADES, label=f"Grade {i+1}", value="1st Grade", scale=1)
                        disorders_multi = gr.Dropdown(choices=DISORDER_TYPES, label="Disorders", multiselect=True, value=["Articulation Disorders"], scale=2)
                        num_students_multi = gr.Number(label="# Students", value=1, minimum=1, step=1, scale=1)
                        model_multi = gr.Dropdown(choices=FREE_MODELS + PREMIUM_MODELS, label="Model", value="Llama3.2", scale=1)
                        characteristics_multi = gr.Textbox(
                            label="Specific Characteristics",
                            placeholder="e.g., has final consonant deletion, difficulty with /s/",
                            lines=1,
                            scale=2
                        )
                        
                        condition_rows.append([grade_multi, disorders_multi, num_students_multi, model_multi, characteristics_multi])
                        condition_row_components.append(row)
                
                with gr.Row():
                    add_row_btn = gr.Button("➕ Add Row", size="sm")
                    remove_row_btn = gr.Button("➖ Remove Row", size="sm")
            
            generate_btn_multiple = gr.Button("🚀 Generate All Cases", variant="primary", size="lg")
            stop_btn_multiple = gr.Button("⛔ Stop Generation", size="sm", variant="stop", visible=False)
            
            # OUTPUT
            gr.Markdown("### Generated Cases")
            output_multiple = gr.Markdown()
            
            # SAVE SECTION
            with gr.Group(visible=False, elem_classes="save-section") as save_section_multiple:
                gr.Markdown("### 💾 Save Batch")
                
                save_path_multiple = gr.Textbox(
                    label="Save Path",
                    interactive=False,
                    value=""
                )
                
                with gr.Row():
                    save_btn_multiple = gr.Button("💾 Save", variant="primary", size="sm")
                    save_as_btn_multiple = gr.DownloadButton("📥 Save As", size="sm")
                
                save_status_multiple = gr.Markdown("")
            
            # FEEDBACK SECTION
            with gr.Accordion("Provide Feedback", open=False):
                gr.Markdown("### Evaluate Generated Cases")
                
                case_selector = gr.Dropdown(
                    choices=["Whole Batch"],
                    label="Select Case to Evaluate",
                    value="Whole Batch"
                )
                
                with gr.Row():
                    rating_clinical_multi = gr.Slider(1, 5, value=3, step=1, label="Clinical Accuracy")
                    rating_age_multi = gr.Slider(1, 5, value=3, step=1, label="Age Appropriate")
                    rating_goals_multi = gr.Slider(1, 5, value=3, step=1, label="Goal Quality")
                
                with gr.Row():
                    rating_notes_multi = gr.Slider(1, 5, value=3, step=1, label="Session Notes")
                    rating_background_multi = gr.Slider(1, 5, value=3, step=1, label="Background")
                
                categories_list = load_json(FEEDBACK_CATEGORIES, {"categories": []}).get("categories", [])
                feedback_cat_multi = gr.Dropdown(
                    choices=["General", "Other"] + categories_list,
                    label="Category",
                    value="General"
                )
                
                detailed_feedback_multi = gr.Textbox(
                    label="Detailed Feedback",
                    lines=4,
                    visible=False,
                    placeholder="Provide specific feedback..."
                )
                
                grammar_check_btn_multi = gr.Button("AI Grammar Check", size="sm", visible=False)
                
                with gr.Row():
                    submit_feedback_btn_multi = gr.Button("Submit Feedback", variant="secondary")
                    add_another_feedback_btn = gr.Button("Add Another Evaluation", variant="secondary", size="sm")
                
                feedback_status_multi = gr.Markdown("")
        
        with gr.Column(visible=False) as group_session_page:
            with gr.Row():
                gr.Markdown("# Generate Group Session")
                back_btn_group = gr.Button("← Back", size="sm")
            
            gr.Markdown("### Configure Group Members")
            gr.Markdown("*Create therapy groups following clinical grouping strategies*")
            
            # Group size selection
            group_size = gr.Radio(choices=[2, 3, 4], label="Group Size", value=2)

            # Custom ID fields for group members
            with gr.Row():
                use_custom_group_ids = gr.Checkbox(label="Use Custom Student IDs", value=False)
                custom_group_id_prefix = gr.Textbox(
                    label="ID Prefix",
                    placeholder="e.g., S, GRP-",
                    value="S",
                    visible=False,
                    scale=1
                )
                custom_group_id_start = gr.Number(
                    label="Start Number",
                    value=1,
                    minimum=1,
                    step=1,
                    visible=False,
                    scale=1
                )

            # Member configurations
            member_configs = []
            member_rows = []
            
            for i in range(4):  # Max 4 members
                with gr.Row(visible=(i < 2)) as member_row:
                    gr.Markdown(f"**Member {i+1}:**")
                    grade_group = gr.Dropdown(
                        choices=ALL_GRADES,
                        label="Grade",
                        value="1st Grade",
                        scale=1
                    )
                    disorders_group = gr.Dropdown(
                        choices=DISORDER_TYPES,
                        label="Disorders",
                        multiselect=True,
                        value=["Articulation Disorders"],
                        scale=2
                    )
                    member_configs.append([grade_group, disorders_group])
                    member_rows.append(member_row)
            
            # Compatibility check display
            compatibility_status = gr.Markdown("### Compatibility Status\n*Configure members above to check compatibility*")
            
            # Model and options
            with gr.Row():
                model_group = gr.Dropdown(
                    choices=FREE_MODELS + PREMIUM_MODELS,
                    label="AI Model",
                    value="Llama3.2",
                    scale=1
                )
                search_existing = gr.Checkbox(
                    label="Search Existing Cases First",
                    value=True,
                    scale=1
                )
            
            with gr.Row():
                check_compat_btn = gr.Button("🔍 Check Compatibility", size="sm", variant="secondary")
                generate_group_btn = gr.Button("🚀 Generate Group Session", variant="primary", size="sm")
                stop_btn_group = gr.Button("⛔ Stop", size="sm", variant="stop", visible=False)
            
            # Output
            gr.Markdown("### Group Session Plan")
            output_group = gr.Markdown()
            
            # Save section
            with gr.Group(visible=False, elem_classes="save-section") as save_section_group:
                gr.Markdown("### 💾 Save Group Session")
                
                save_path_group = gr.Textbox(
                    label="Save Path",
                    interactive=False,
                    value=""
                )
                
                with gr.Row():
                    save_btn_group = gr.Button("💾 Save", variant="primary", size="sm")
                    save_as_btn_group = gr.DownloadButton("📥 Save As", size="sm")
                
                save_status_group = gr.Markdown("")
            
            # Feedback section
            with gr.Accordion("Provide Feedback", open=False):
                gr.Markdown("### Evaluate Group Session")
                
                with gr.Row():
                    rating_grade_accuracy = gr.Slider(1, 5, value=3, step=1, label="Grade Level Accuracy")
                    rating_disorder_accuracy = gr.Slider(1, 5, value=3, step=1, label="Disorder Types Accuracy")
                
                with gr.Row():
                    rating_goal_accuracy = gr.Slider(1, 5, value=3, step=1, label="Annual Goal Accuracy")
                    rating_notes_completeness = gr.Slider(1, 5, value=3, step=1, label="Session Notes Completeness")
                
                rating_setup_feasibility = gr.Slider(1, 5, value=3, step=1, label="Group Setup Feasibility")
                
                categories_list = load_json(FEEDBACK_CATEGORIES, {"categories": []}).get("categories", [])
                feedback_cat_group = gr.Dropdown(
                    choices=["General", "Other"] + categories_list,
                    label="Category",
                    value="General"
                )
                
                detailed_feedback_group = gr.Textbox(
                    label="Detailed Feedback",
                    lines=4,
                    visible=False,
                    placeholder="Provide specific feedback about the group configuration..."
                )
                
                grammar_check_btn_group = gr.Button("AI Grammar Check", size="sm", visible=False)
                
                submit_feedback_btn_group = gr.Button("Submit Feedback", variant="secondary")
                feedback_status_group = gr.Markdown("")
        
        # EVENT HANDLERS
        def show_single():
            return {cover_page: gr.update(visible=False), single_case_page: gr.update(visible=True), multiple_cases_page: gr.update(visible=False), group_session_page: gr.update(visible=False)}
        
        def show_multiple():
            return {cover_page: gr.update(visible=False), single_case_page: gr.update(visible=False), multiple_cases_page: gr.update(visible=True), group_session_page: gr.update(visible=False)}
        
        def show_group():
            return {cover_page: gr.update(visible=False), single_case_page: gr.update(visible=False), multiple_cases_page: gr.update(visible=False), group_session_page: gr.update(visible=True)}
        
        def show_cover():
            return {cover_page: gr.update(visible=True), single_case_page: gr.update(visible=False), multiple_cases_page: gr.update(visible=False), group_session_page: gr.update(visible=False)}
        
        btn_single.click(show_single, outputs=[cover_page, single_case_page, multiple_cases_page, group_session_page])
        btn_multiple.click(show_multiple, outputs=[cover_page, single_case_page, multiple_cases_page, group_session_page])
        btn_group.click(show_group, outputs=[cover_page, single_case_page, multiple_cases_page, group_session_page])
        
        back_btn_single.click(show_cover, outputs=[cover_page, single_case_page, multiple_cases_page, group_session_page])
        back_btn_multiple.click(show_cover, outputs=[cover_page, single_case_page, multiple_cases_page, group_session_page])
        back_btn_group.click(show_cover, outputs=[cover_page, single_case_page, multiple_cases_page, group_session_page])
        
        # Generation with streaming status
        def handle_generation(grade, disorders, model, pop_spec, refs):
            if not disorders:
                yield "⚠️ Please select at least one disorder", None, gr.update(), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
                return
            
            for output, filename, btn_state, path_display, stop_vis, save_vis in generate_single_case(grade, disorders, model, pop_spec, refs):
                yield output, filename, btn_state, path_display, stop_vis, save_vis
        
        generate_btn_single.click(
            fn=handle_generation,
            inputs=[grade_single, disorders_single, model_single, population_spec_single, reference_files_single],
            outputs=[output_single, generated_filename, generate_btn_single, save_path_display, stop_btn_single, save_section]
        )
        
        # Save functionality
        save_btn.click(
            fn=handle_save,
            inputs=save_path_display,
            outputs=save_status
        )
        
        # Download functionality
        def prepare_download():
            if current_case_data["content"]:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                temp_file = f"temp_case_{timestamp}.md"
                save_case_file(current_case_data["content"], temp_file)
                return temp_file
            return None
        
        save_as_btn.click(fn=prepare_download, outputs=save_as_btn)
        
        # Show/hide detailed feedback
        def toggle_feedback_fields(category):
            is_other = (category == "Other")
            return gr.update(visible=is_other), gr.update(visible=is_other)
        
        feedback_cat_single.change(fn=toggle_feedback_fields, inputs=feedback_cat_single, outputs=[detailed_feedback_single, grammar_check_btn])
        
        # Grammar check
        grammar_check_btn.click(fn=ai_grammar_check, inputs=detailed_feedback_single, outputs=detailed_feedback_single)
        
        # Submit feedback with reset
        def handle_feedback(r1, r2, r3, r4, r5, cat, detailed):
            if not current_case_data.get("case_id"):
                return ("❌ Error: Save the case first before submitting feedback.", gr.update(),
                        gr.update(value=3), gr.update(value=3), gr.update(value=3),
                        gr.update(value=3), gr.update(value=3),
                        gr.update(value=""), gr.update(value="General"))
            
            ratings = {
                "clinical_accuracy": r1,
                "age_appropriateness": r2,
                "goal_quality": r3,
                "session_notes": r4,
                "background": r5
            }
            return submit_feedback(current_case_data["case_id"], ratings, cat, detailed)
        
        submit_feedback_btn_single.click(
            fn=handle_feedback,
            inputs=[rating_clinical, rating_age, rating_goals, rating_notes, rating_background, feedback_cat_single, detailed_feedback_single],
            outputs=[feedback_status_single, feedback_cat_single, rating_clinical, rating_age, rating_goals, rating_notes, rating_background, detailed_feedback_single, feedback_cat_single]
        )
        
        # === MULTIPLE CASES PAGE EVENT HANDLERS ===
        
        # Row management
        def update_row_visibility(count):
            return [gr.update(visible=(i < count)) for i in range(MAX_CONDITION_ROWS)]
        
        def add_row(count):
            return min(count + 1, MAX_CONDITION_ROWS)
        
        def remove_row(count):
            return max(count - 1, 1)
        
        add_row_btn.click(fn=add_row, inputs=visible_row_count, outputs=visible_row_count).then(
            fn=update_row_visibility, inputs=visible_row_count, outputs=condition_row_components
        )
        
        remove_row_btn.click(fn=remove_row, inputs=visible_row_count, outputs=visible_row_count).then(
            fn=update_row_visibility, inputs=visible_row_count, outputs=condition_row_components
        )
        
        # Parse natural language request
        def handle_parse_request(request):
            if not request.strip():
                return gr.update(visible=False, value="")
            
            tasks, breakdown = parse_complex_request(request)
            
            if tasks:
                # Auto-fill the rows with parsed tasks
                updates = []
                for i in range(MAX_CONDITION_ROWS):
                    if i < len(tasks):
                        task = tasks[i]
                        updates.extend([
                            gr.update(value=task.get('grade', '1st Grade')),
                            gr.update(value=task.get('disorders', [])),
                            gr.update(value=task.get('count', 1)),
                            gr.update(value=task.get('model', 'Llama3.2')),
                            gr.update(value=task.get('characteristics', ''))
                        ])
                    else:
                        updates.extend([
                            gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
                        ])
                
                # Update visible row count
                visible_count = min(len(tasks), MAX_CONDITION_ROWS)
                
                return [gr.update(visible=True, value=breakdown), visible_count] + updates
            else:
                return [gr.update(visible=True, value=breakdown), gr.update()] + [gr.update() for _ in range(MAX_CONDITION_ROWS * 5)]
        
        all_row_inputs = [item for sublist in condition_rows for item in sublist]
        
        parse_btn.click(
            fn=handle_parse_request, 
            inputs=chat_request, 
            outputs=[parsed_output, visible_row_count] + all_row_inputs
        )
        
        clear_chat_btn.click(lambda: ("", gr.update(visible=False)), outputs=[chat_request, parsed_output])
        
        # Toggle custom ID fields
        def toggle_custom_ids(use_custom):
            return gr.update(visible=use_custom), gr.update(visible=use_custom)
        
        use_custom_ids.change(
            fn=toggle_custom_ids,
            inputs=use_custom_ids,
            outputs=[custom_id_prefix, custom_id_start]
        )
        
        # Generate multiple cases
        def handle_multiple_generation(row_count, use_custom, id_prefix, id_start, *row_inputs):
            tasks = []
            
            # Extract tasks from visible rows
            for i in range(row_count):
                grade = row_inputs[i * 5]
                disorders = row_inputs[i * 5 + 1]
                num = row_inputs[i * 5 + 2]
                model = row_inputs[i * 5 + 3]
                characteristics = row_inputs[i * 5 + 4]
                
                if disorders:  # Only add if disorders selected
                    tasks.append({
                        "grade": grade,
                        "disorders": disorders,
                        "model": model,
                        "count": int(num),
                        "characteristics": characteristics
                    })
            
            if not tasks:
                yield ("⚠️ Please configure at least one condition row", 
                       gr.update(visible=False), 
                       gr.update(choices=["Whole Batch"]), 
                       gr.update(interactive=True))
                return
            
            save_path = DEFAULT_OUTPUT_PATH
            for output, filename, case_choices, btn_state in generate_multiple_cases(
                tasks, save_path, use_custom, id_prefix, id_start
            ):
                # Show save section after generation
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                full_path = f"{save_path}multiple_cases_{timestamp}.md"
                yield (output, 
                       gr.update(visible=True, value=full_path), 
                       gr.update(choices=case_choices), 
                       btn_state)
        
        generate_btn_multiple.click(
            fn=handle_multiple_generation,
            inputs=[visible_row_count, use_custom_ids, custom_id_prefix, custom_id_start] + all_row_inputs,
            outputs=[output_multiple, save_path_multiple, case_selector, generate_btn_multiple]
        ).then(
            fn=lambda: gr.update(visible=True),
            outputs=save_section_multiple
        )
        
        # Save functionality for multiple cases
        def handle_save_multiple(filepath):
            if not multiple_cases_batch["cases"]:
                return "❌ No cases to save"
            
            try:
                combined_content = f"# Multiple Cases Batch\n**Batch ID:** {multiple_cases_batch['batch_id']}\n\n"
                combined_content += "\n\n".join([case["content"] for case in multiple_cases_batch["cases"]])
                
                save_case_file(combined_content, filepath)
                
                # Save metadata
                for case in multiple_cases_batch["cases"]:
                    save_case_to_db({
                        **case["metadata"],
                        "batch_id": multiple_cases_batch["batch_id"],
                        "filepath": filepath
                    })
                
                return f"✅ Batch saved successfully!\n**Path:** `{filepath}`\n**Batch ID:** {multiple_cases_batch['batch_id']}"
            except Exception as e:
                return f"❌ Error: {str(e)}"
        
        save_btn_multiple.click(fn=handle_save_multiple, inputs=save_path_multiple, outputs=save_status_multiple)
        
        # Download for multiple
        def prepare_download_multiple():
            if multiple_cases_batch["cases"]:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                temp_file = f"temp_batch_{timestamp}.md"
                combined = "\n\n".join([case["content"] for case in multiple_cases_batch["cases"]])
                save_case_file(combined, temp_file)
                return temp_file
            return None
        
        save_as_btn_multiple.click(fn=prepare_download_multiple, outputs=save_as_btn_multiple)
        
        # Feedback for multiple cases
        def toggle_feedback_multi(category):
            is_other = (category == "Other")
            return gr.update(visible=is_other), gr.update(visible=is_other)
        
        feedback_cat_multi.change(fn=toggle_feedback_multi, inputs=feedback_cat_multi, outputs=[detailed_feedback_multi, grammar_check_btn_multi])
        
        grammar_check_btn_multi.click(fn=ai_grammar_check, inputs=detailed_feedback_multi, outputs=detailed_feedback_multi)
        
        # Submit feedback for multiple
        def handle_feedback_multiple(selected_case, r1, r2, r3, r4, r5, cat, detailed):
            if not multiple_cases_batch["batch_id"]:
                return ("❌ Save the batch first", gr.update(),
                       gr.update(value=3), gr.update(value=3), gr.update(value=3),
                       gr.update(value=3), gr.update(value=3),
                       gr.update(value=""), gr.update(value="General"))
            
            ratings = {
                "clinical_accuracy": r1,
                "age_appropriateness": r2,
                "goal_quality": r3,
                "session_notes": r4,
                "background": r5
            }
            
            # Determine which case(s) to submit feedback for
            if selected_case == "Whole Batch":
                target_id = multiple_cases_batch["batch_id"]
            else:
                # Extract case number from "Case 1: ..."
                case_num = int(selected_case.split(":")[0].split()[1]) - 1
                target_id = multiple_cases_batch["cases"][case_num]["id"]
            
            return submit_feedback(target_id, ratings, cat, detailed)
        
        submit_feedback_btn_multi.click(
            fn=handle_feedback_multiple,
            inputs=[case_selector, rating_clinical_multi, rating_age_multi, rating_goals_multi,
                   rating_notes_multi, rating_background_multi, feedback_cat_multi, detailed_feedback_multi],
            outputs=[feedback_status_multi, feedback_cat_multi, rating_clinical_multi, rating_age_multi,
                    rating_goals_multi, rating_notes_multi, rating_background_multi,
                    detailed_feedback_multi, feedback_cat_multi]
        )
        
        # Add another evaluation - just clears the form
        def reset_feedback_form():
            return (gr.update(value=3), gr.update(value=3), gr.update(value=3),
                   gr.update(value=3), gr.update(value=3),
                   gr.update(value=""), gr.update(value="General"), "")
        
        add_another_feedback_btn.click(
            fn=reset_feedback_form,
            outputs=[rating_clinical_multi, rating_age_multi, rating_goals_multi,
                    rating_notes_multi, rating_background_multi,
                    detailed_feedback_multi, feedback_cat_multi, feedback_status_multi]
        )
        
        # === GROUP SESSION PAGE EVENT HANDLERS ===

        # Update member row visibility based on group size
        def update_member_visibility(size):
            return [gr.update(visible=(i < size)) for i in range(4)]

        group_size.change(
            fn=update_member_visibility,
            inputs=group_size,
            outputs=member_rows
        )

        # Toggle custom ID fields for group
        def toggle_custom_group_ids(use_custom):
            return gr.update(visible=use_custom), gr.update(visible=use_custom)

        use_custom_group_ids.change(
            fn=toggle_custom_group_ids,
            inputs=use_custom_group_ids,
            outputs=[custom_group_id_prefix, custom_group_id_start]
        )
        
        # Check compatibility
        def check_compatibility(size, *member_inputs):
            grades = [member_inputs[i*2] for i in range(size)]
            disorders = [member_inputs[i*2+1] for i in range(size)]
            
            grade_compat, grade_msg = check_grade_compatibility(grades)
            disorder_compat, disorder_msg = check_disorder_compatibility(disorders)
            
            status = f"### Compatibility Check Results\n\n"
            status += f"**Grade Levels:** {grade_msg}\n\n"
            status += f"**Disorder Combinations:** {disorder_msg}\n\n"
            
            if grade_compat and disorder_compat:
                status += "✅ **Group configuration is compatible!**"
            else:
                status += "⚠️ **Group configuration needs adjustment**"
            
            return status
        
        all_member_inputs = [item for sublist in member_configs for item in sublist]
        
        check_compat_btn.click(
            fn=check_compatibility,
            inputs=[group_size] + all_member_inputs,
            outputs=compatibility_status
        )
        
        # Generate group session
        def handle_group_generation(size, model, search, use_custom, id_prefix, id_start, *member_inputs):
            grades = [member_inputs[i*2] for i in range(size)]
            disorders = [member_inputs[i*2+1] for i in range(size)]

            for output, filename, btn_state, stop_vis in generate_group_session(
                size, grades, disorders, model, search, use_custom, id_prefix, id_start
            ):
                if filename:
                    full_path = f"{DEFAULT_OUTPUT_PATH}{os.path.basename(filename)}"
                    yield output, gr.update(visible=True, value=full_path), btn_state, stop_vis
                else:
                    yield output, gr.update(), btn_state, stop_vis

        generate_group_btn.click(
            fn=lambda: (gr.update(interactive=False), gr.update(visible=True)),
            outputs=[generate_group_btn, stop_btn_group]
        ).then(
            fn=handle_group_generation,
            inputs=[group_size, model_group, search_existing, use_custom_group_ids, custom_group_id_prefix, custom_group_id_start] + all_member_inputs,
            outputs=[output_group, save_path_group, generate_group_btn, stop_btn_group]
        ).then(
            fn=lambda: gr.update(visible=True),
            outputs=save_section_group
        )
        
        # Stop button handlers
        def stop_generation():
            generation_control["should_stop"] = True
            return gr.update(visible=False), gr.update(interactive=True, variant="primary"), "⛔ Stopping generation..."
        
        stop_btn_single.click(
            fn=stop_generation,
            outputs=[stop_btn_single, generate_btn_single, output_single]
        )
        
        stop_btn_multiple.click(
            fn=stop_generation,
            outputs=[stop_btn_multiple, generate_btn_multiple, output_multiple]
        )
        
        stop_btn_group.click(
            fn=stop_generation,
            outputs=[stop_btn_group, generate_group_btn, output_group]
        )
        
        # Save group session
        def handle_save_group(filepath):
            if not group_session_data["session_id"]:
                return "❌ No group session to save"
            
            try:
                # Read current output and save
                content = f"# Group Session\n**Session ID:** {group_session_data['session_id']}\n"
                # Add members data
                for member in group_session_data["members"]:
                    content += member["output"]
                
                save_case_file(content, filepath)
                
                # Save metadata
                save_case_to_db({
                    "type": "group",
                    "session_id": group_session_data["session_id"],
                    "members": group_session_data["members"],
                    "filepath": filepath
                })
                
                return f"✅ Group session saved!\n**Path:** `{filepath}`\n**Session ID:** {group_session_data['session_id']}"
            except Exception as e:
                return f"❌ Error: {str(e)}"
        
        save_btn_group.click(
            fn=handle_save_group,
            inputs=save_path_group,
            outputs=save_status_group
        )
        
        # Download group session
        def prepare_download_group():
            if group_session_data["session_id"]:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                temp_file = f"temp_group_{timestamp}.md"
                content = f"# Group Session\n**Session ID:** {group_session_data['session_id']}\n"
                for member in group_session_data["members"]:
                    content += member["output"]
                save_case_file(content, temp_file)
                return temp_file
            return None
        
        save_as_btn_group.click(
            fn=prepare_download_group,
            outputs=save_as_btn_group
        )
        
        # Feedback for group session
        def toggle_feedback_group(category):
            is_other = (category == "Other")
            return gr.update(visible=is_other), gr.update(visible=is_other)
        
        feedback_cat_group.change(
            fn=toggle_feedback_group,
            inputs=feedback_cat_group,
            outputs=[detailed_feedback_group, grammar_check_btn_group]
        )
        
        grammar_check_btn_group.click(
            fn=ai_grammar_check,
            inputs=detailed_feedback_group,
            outputs=detailed_feedback_group
        )
        
        # Submit feedback for group
        def handle_feedback_group(r1, r2, r3, r4, r5, cat, detailed):
            if not group_session_data.get("session_id"):
                return ("❌ Save the group session first", gr.update(),
                       gr.update(value=3), gr.update(value=3), gr.update(value=3),
                       gr.update(value=3), gr.update(value=3),
                       gr.update(value=""), gr.update(value="General"))
            
            ratings = {
                "grade_accuracy": r1,
                "disorder_accuracy": r2,
                "goal_accuracy": r3,
                "notes_completeness": r4,
                "setup_feasibility": r5
            }
            
            return submit_feedback(group_session_data["session_id"], ratings, cat, detailed)
        
        submit_feedback_btn_group.click(
            fn=handle_feedback_group,
            inputs=[rating_grade_accuracy, rating_disorder_accuracy, rating_goal_accuracy,
                   rating_notes_completeness, rating_setup_feasibility,
                   feedback_cat_group, detailed_feedback_group],
            outputs=[feedback_status_group, feedback_cat_group, rating_grade_accuracy,
                    rating_disorder_accuracy, rating_goal_accuracy, rating_notes_completeness,
                    rating_setup_feasibility, detailed_feedback_group, feedback_cat_group]
        )
    
    return app

if __name__ == "__main__":
    app = create_ui()
    app.launch()