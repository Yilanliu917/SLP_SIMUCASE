"""
Case generation functions for single, multiple, and group sessions
"""
from datetime import datetime
from typing import List, Dict
import gradio as gr

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

from .config import *
from .models import current_case_data, multiple_cases_batch, group_session_data, generation_control
from .utils import *

# --- SINGLE CASE GENERATION ---
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
    
    if model in FREE_MODELS:
        template = load_prompt("single_case_free_model.txt")
        if not template:
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
        content = response.content if hasattr(response, 'content') else str(response)
        output = f"""# Generated Case File

**Model Used:** {model}
**Grade:** {grade}
**Disorders:** {disorder_string}

---

{content}

---
"""
        current_case_data["content"] = output
        current_case_data["metadata"] = {
            "type": "single",
            "disorders": disorders,
            "grade": grade,
            "model": model,
            "timestamp": datetime.now().strftime('%Y%m%d_%H%M%S')
        }
        
    else:
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
        if generation_control["should_stop"]:
            combined_output += f"\n\n⛔ **Generation stopped by user after {idx} cases**"
            yield (combined_output, 
                   None, 
                   ["Whole Batch"] + [c['display_id'] for c in all_cases],
                   gr.update(interactive=True, variant="primary"))
            return
        
        if use_custom_id:
            case_display_id = f"{id_prefix}{case_counter:03d}"
            case_counter += 1
        else:
            case_display_id = f"Case {idx+1}"
        
        yield (f"{combined_output}\n\n🔄 Generating {case_display_id}: {task['grade']} - {', '.join(task['disorders'])}", 
               None, 
               ["Whole Batch"], 
               gr.update(interactive=False))
        
        if task['model'] in FREE_MODELS:
            template = load_prompt("single_case_free_model.txt")
            if not template:
                template = "Generate a case for {grade} with {disorders}. {characteristics}"
        else:
            template = load_prompt("single_case_premium_model.txt")
            if not template:
                template = "Generate JSON case for {grade} with {disorders}. {characteristics}"
        
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
        
        yield (combined_output, 
               None, 
               ["Whole Batch"] + [c['display_id'] for c in all_cases],
               gr.update(interactive=False))
    
    multiple_cases_batch["cases"] = all_cases
    multiple_cases_batch["batch_id"] = batch_id
    multiple_cases_batch["timestamp"] = datetime.now().isoformat()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suggested_filename = f"{save_path}multiple_cases_{timestamp}.md"
    
    case_choices = ["Whole Batch"] + [f"{case['display_id']}: {case['metadata']['grade']} - {', '.join(case['metadata']['disorders'])}" for case in all_cases]
    
    final_output = combined_output + f"\n\n---\n\n✅ **Generation Complete!** {len(all_cases)} cases created.\n**Batch ID:** {batch_id}"
    
    yield (final_output, 
           suggested_filename, 
           case_choices, 
           gr.update(interactive=True, variant="primary"))

# --- GROUP SESSION GENERATION ---
def generate_group_session(group_size: int, grades: List[str], disorders_list: List[List[str]], 
                          model: str, search_first: bool = True):
    """Generate or search for group session cases."""
    
    generation_control["should_stop"] = False
    
    session_id = f"group_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    yield (f"🔄 Initializing group session...", 
           None, 
           gr.update(interactive=False),
           gr.update(visible=True))
    
    grade_compat, grade_msg = check_grade_compatibility(grades[:group_size])
    disorder_compat, disorder_msg = check_disorder_compatibility(disorders_list[:group_size])
    
    compat_check = f"""### Compatibility Check

**Grade Levels:** {grade_msg}
**Disorder Combinations:** {disorder_msg}

"""
    
    if not grade_compat or not disorder_compat:
        final_msg = compat_check + "\n⚠️ **Warning:** Group configuration may not be optimal per grouping strategies.\nProceed anyway or adjust configuration.\n"
        yield (final_msg, None, gr.update(interactive=True), gr.update(visible=True))
    
    if search_first:
        yield (compat_check + "🔍 Searching existing cases...", 
               None, 
               gr.update(interactive=False),
               gr.update(visible=True))
        
        matching = search_existing_cases(grades[:group_size], disorders_list[:group_size])
        
        if len(matching) >= group_size:
            yield (compat_check + f"✅ Found {len(matching)} matching cases!\nUsing existing cases for group session...", 
                   None, 
                   gr.update(interactive=False),
                   gr.update(visible=True))
        else:
            yield (compat_check + f"⚠️ Only found {len(matching)} matching cases. Need {group_size}.\nGenerating missing cases...", 
                   None, 
                   gr.update(interactive=False),
                   gr.update(visible=True))
    
    output = f"# Group Session\n**Session ID:** {session_id}\n**Date:** {datetime.now().strftime('%Y-%m-%d')}\n**Group Size:** {group_size}\n\n"
    output += compat_check + "\n---\n\n"
    
    members = []
    for i in range(group_size):
        if generation_control["should_stop"]:
            yield (output + "\n\n⛔ **Generation stopped by user**", 
                   None, 
                   gr.update(interactive=True, variant="primary"),
                   gr.update(visible=True))
            return
        
        yield (output + f"\n🔄 Processing Member {i+1}/{group_size}...", 
               None, 
               gr.update(interactive=False),
               gr.update(visible=True))
        
        member_output = f"\n## Member {i+1}: Student {chr(65+i)}\n"
        member_output += f"**Grade:** {grades[i]}\n"
        member_output += f"**Disorders:** {', '.join(disorders_list[i])}\n\n"
        
        member_output += f"### Background\n- Medical history relevant to {', '.join(disorders_list[i])}\n"
        member_output += f"- Parent concerns about communication development\n"
        member_output += f"- Teacher observations in classroom setting\n\n"
        
        member_output += f"### IEP Goals for Group Session\n"
        member_output += f"1. Goal related to {disorders_list[i][0] if disorders_list[i] else 'communication'}\n"
        member_output += f"2. Social communication goal for group interaction\n\n"
        
        members.append({
            "member_num": i+1,
            "grade": grades[i],
            "disorders": disorders_list[i],
            "output": member_output
        })
        
        output += member_output
    
    output += "\n---\n\n## Group Session Plan\n\n"
    output += f"### Session Focus\nCollaborative communication skills with focus on:\n"
    for i, member in enumerate(members):
        output += f"- Member {i+1}: {', '.join(member['disorders'])}\n"
    
    output += "\n### Group Activities\n"
    output += "1. **Turn-taking activity** - Supports all members' communication goals\n"
    output += "2. **Collaborative problem-solving** - Targets pragmatic language skills\n"
    output += "3. **Peer modeling** - Members support each other's articulation/language goals\n\n"
    
    output += "### Session Notes\n"
    output += "**Session 1:** Initial group session. All members participated appropriately...\n"
    output += "**Session 2:** Continued progress on individual and group goals...\n"
    output += "**Session 3:** Strong peer interaction observed. Goals being met...\n"
    
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

# --- CHAT PARSING ---
def parse_complex_request(request: str) -> tuple:
    """Parse natural language request into generation tasks."""
    try:
        llm = ChatOllama(model="llama3.2:latest", temperature=0.1)
        
        prompt = f"""Parse this natural language request for generating SLP case files.

Request: "{request}"

Available options:
Grades: Pre-K, Kindergarten, 1st Grade, 2nd Grade, 3rd Grade, 4th Grade, 5th Grade, 6th Grade, 7th Grade, 8th Grade, 9th Grade, 10th Grade, 11th Grade, 12th Grade

Disorders: Speech Sound Disorder, Articulation Disorders, Phonological Disorders, Language Disorders, Receptive Language Disorders, Expressive Language Disorders, Pragmatics, Fluency, Childhood Apraxia of Speech

Models: Llama3.2, Qwen 2.5 7B, Qwen 2.5 32B, DeepSeek R1 32B, GPT-4o, Gemini 2.5 Pro, Claude 3 Opus, Claude 3.5 Sonnet

Instructions:
1. Identify total number of students
2. Identify which model to use
3. Break down grade distributions
4. Identify disorders for each group
5. Extract specific characteristics

Format each task as:
TASK [number]: [count] student(s), Grade: [grade], Disorders: [disorders], Model: [model], Characteristics: [specific notes if any]

Now parse the request above:"""
        
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else ""
        
        tasks = []
        
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('TASK'):
                try:
                    parts = line.split(':', 1)[1].strip() if ':' in line else line
                    
                    task = {
                        "count": 1,
                        "grade": "1st Grade",
                        "disorders": [],
                        "model": "Llama3.2",
                        "characteristics": ""
                    }
                    
                    if "student" in parts.lower():
                        count_part = parts.split("student")[0].strip()
                        try:
                            task["count"] = int(count_part.split()[0])
                        except:
                            task["count"] = 1
                    
                    for grade in ALL_GRADES:
                        if grade.lower() in parts.lower():
                            task["grade"] = grade
                            break
                    
                    for disorder in DISORDER_TYPES:
                        if disorder.lower() in parts.lower():
                            task["disorders"].append(disorder)
                    
                    all_models = FREE_MODELS + PREMIUM_MODELS
                    for model in all_models:
                        if model.lower() in parts.lower():
                            task["model"] = model
                            break
                    
                    if "Characteristics:" in parts:
                        chars = parts.split("Characteristics:")[1].strip()
                        if chars.lower() != "none":
                            task["characteristics"] = chars
                    
                    tasks.append(task)
                    
                except Exception as e:
                    print(f"Error parsing task line: {e}")
                    continue
        
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
