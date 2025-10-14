"""
Case generation functions for single, multiple, and group sessions
"""
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import gradio as gr
import pandas as pd
import re
import os

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

from .config import *
from .models import current_case_data, multiple_cases_batch, group_session_data, generation_control
from .utils import *
from .name_utils import generate_unique_names

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

    # Precompute unique anonymous names for all cases
    anon_names = generate_unique_names(total_tasks, seed=None)
    
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

        # Get the anonymous name for this case
        anon_name = anon_names[idx]

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
            "characteristics": lambda x: task.get('characteristics', ''),
            "exclude_names_prompt": lambda x: ""
        } | prompt | llm
        
        response = rag_chain.invoke(question_text)
        
        if task['model'] in FREE_MODELS:
            content = response.content if hasattr(response, 'content') else str(response)
            case_output = f"""
---

## {case_display_id}: {anon_name}

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

## {case_display_id}: {anon_name}

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

        # Add anonymous name to metadata
        metadata = task.copy()
        metadata["anonymous_name"] = anon_name

        all_cases.append({
            "id": case_id,
            "display_id": case_display_id,
            "content": case_output,
            "metadata": metadata
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

    grade_compat, grade_msg = check_grade_compatibility(grades[:group_size])
    disorder_compat, disorder_msg = check_disorder_compatibility(disorders_list[:group_size])

    compat_check = f"""### Compatibility Check

**Grade Levels:** {grade_msg}
**Disorder Combinations:** {disorder_msg}

"""

    if not grade_compat or not disorder_compat:
        final_msg = compat_check + "\n⚠️ **Warning:** Group configuration may not be optimal per grouping strategies.\nProceed anyway or adjust configuration.\n"

    # Search for existing cases in folder
    existing_cases = []
    if search_first:
        yield (compat_check + "🔍 Searching existing cases in folder...",
               None,
               gr.update(interactive=False),
               gr.update(visible=True))

        from .utils import search_existing_cases_in_folder
        existing_cases = search_existing_cases_in_folder(grades[:group_size], disorders_list[:group_size])

        if len(existing_cases) >= group_size:
            yield (compat_check + f"✅ Found {len(existing_cases)} matching cases!\nUsing existing cases for group session...",
                   None,
                   gr.update(interactive=False),
                   gr.update(visible=True))
        else:
            yield (compat_check + f"⚠️ Only found {len(existing_cases)} matching cases. Need {group_size}.\n",
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
                   gr.update(visible=False))
            return

        # Generate student ID
        if use_custom_ids:
            student_id = f"{id_prefix}{id_counter:03d}"
            id_counter += 1
        else:
            student_id = f"Member {i+1}"

        yield (output + f"\n🔄 Processing {student_id}/{group_size}...",
               None,
               gr.update(interactive=False),
               gr.update(visible=True))

        # Find matching existing case for this member
        matching_case = None
        for case in existing_cases:
            if case.get("member_index") == i:
                matching_case = case
                existing_cases.remove(case)
                break

        if matching_case:
            # Use existing case - display full profile
            member_output = f"\n## {student_id}: {matching_case['name']}\n\n"
            member_output += f"**Grade:** {matching_case['grade']} | **Age:** {matching_case['age']} | **Gender:** {matching_case['gender']}\n"
            member_output += f"**Disorders:** {matching_case['disorders']}\n"
            if matching_case.get('characteristics'):
                member_output += f"**Special Characteristics:** {matching_case['characteristics']}\n"
            member_output += "\n"

            # Add background
            if matching_case.get('background'):
                member_output += f"### Background\n{matching_case['background']}\n\n"

            # Add annual goals
            if matching_case.get('annual_goals'):
                member_output += f"### Annual IEP Goals\n"
                for idx, goal in enumerate(matching_case['annual_goals'], 1):
                    member_output += f"{idx}. {goal}\n"
                member_output += "\n"

            # Add latest session notes (limit to 3)
            if matching_case.get('session_notes'):
                member_output += f"### Latest Session Notes\n"
                for idx, note in enumerate(matching_case['session_notes'][:3], 1):
                    member_output += f"**Session {idx}:** {note}\n"
                member_output += "\n"


        else:
            # No matching case found - generate new case using single case generation logic
            yield (output + f"\n🔄 Generating new case for {student_id}...",
                   None,
                   gr.update(interactive=False),
                   gr.update(visible=True))

            # Initialize vector database for generation
            embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
            vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

            # Prepare generation template
            disorder_string = ", ".join(disorders_list[i])

            if model in FREE_MODELS:
                template = load_prompt("single_case_free_model.txt")
                if not template:
                    template = "You are an expert SLP. Create a case file for a {grade} student with {disorders}."
            else:
                template = load_prompt("single_case_premium_model.txt")
                if not template:
                    template = "Generate a case file as JSON for {grade} with {disorders}."

            prompt = ChatPromptTemplate.from_template(template)
            llm = get_llm(model)

            rag_chain = {
                "context": retriever,
                "question": RunnablePassthrough(),
                "population_spec": lambda x: "general population",
                "disorders": lambda x: disorder_string,
                "grade": lambda x: grades[i],
                "exclude_names_prompt": lambda x: ""
            } | prompt | llm

            response = rag_chain.invoke(f"Generate case for {grades[i]} with {disorder_string}")

            # Format the generated case
            if model in FREE_MODELS:
                content = response.content if hasattr(response, 'content') else str(response)

                # Extract name from content if possible (look for a name pattern)
                name_match = re.search(r'(?:Name:|Student:|Case:)\s*([A-Z][a-z]+\s+[A-Z][a-z]+)', content)
                student_name = name_match.group(1) if name_match else "Generated Student"

                member_output = f"\n## {student_id}: {student_name}\n\n"
                member_output += f"**Grade:** {grades[i]}\n"
                member_output += f"**Disorders:** {disorder_string}\n"
                member_output += f"**Model:** {model}\n"
                member_output += f"**Source:** Newly Generated\n\n"
                member_output += f"{content}\n\n"

            else:
                case_file = response
                profile = case_file.student_profile

                member_output = f"\n## {student_id}: {profile.name}\n\n"
                member_output += f"**Grade:** {profile.grade_level} | **Age:** {profile.age} | **Gender:** {profile.gender}\n"
                member_output += f"**Disorders:** {disorder_string}\n"
                member_output += f"**Model:** {model}\n"
                member_output += f"**Source:** Newly Generated\n\n"

                # Add background
                member_output += f"### Background\n"
                member_output += f"- **Medical History:** {profile.background.medical_history}\n"
                member_output += f"- **Parent Concerns:** {profile.background.parent_concerns}\n"
                member_output += f"- **Teacher Concerns:** {profile.background.teacher_concerns}\n\n"

                # Add annual goals
                member_output += f"### Annual IEP Goals\n"
                for idx, goal in enumerate(case_file.annual_goals, 1):
                    member_output += f"{idx}. {goal}\n"
                member_output += "\n"

                # Add latest session notes
                member_output += f"### Latest Session Notes\n"
                for idx, note in enumerate(case_file.latest_session_notes, 1):
                    member_output += f"**Session {idx}:** {note}\n"
                member_output += "\n"

        members.append({
            "member_num": i+1,
            "student_id": student_id,
            "grade": grades[i],
            "disorders": disorders_list[i],
            "output": member_output
        })

        output += member_output

    # Generate AI-powered group session summary
    yield (output + "\n\n🔄 Generating group session summary...",
           None,
           gr.update(interactive=False),
           gr.update(visible=True))

    try:
        # Prepare student profiles summary for the AI
        student_profiles_text = ""
        for member in members:
            student_profiles_text += f"\n**{member['student_id']}:**\n"
            student_profiles_text += f"- Grade: {member['grade']}\n"
            student_profiles_text += f"- Disorders: {', '.join(member['disorders'])}\n"
            # Extract annual goals from member output
            if "### Annual IEP Goals" in member['output']:
                goals_section = member['output'].split("### Annual IEP Goals")[1].split("###")[0].strip()
                student_profiles_text += f"- Annual IEP Goals:\n{goals_section}\n"

        # Build prioritized goals placeholder format
        prioritized_goals_format = ""
        for member in members:
            prioritized_goals_format += f"- **{member['student_id']}:** [Select ONE goal from their annual IEP goals]\n"

        # Build data collection table placeholder
        data_collection_table = "| Student | Target Goal | Trial 1 | Trial 2 | Trial 3 | Trial 4 | Trial 5 | Accuracy | Notes |\n"
        data_collection_table += "|---------|-------------|---------|---------|---------|---------|---------|----------|-------|\n"
        for member in members:
            data_collection_table += f"| {member['student_id']} | [Goal] | | | | | | | |\n"

        # Load appropriate template
        if model in FREE_MODELS:
            template = load_prompt("group_session_free_model.txt")
        else:
            template = load_prompt("group_session_premium_model.txt")

        if not template:
            template = "Create a group session plan for:\n{student_profiles}"

        # Format the prompt
        session_prompt = template.format(
            student_profiles=student_profiles_text,
            prioritized_goals_placeholder=prioritized_goals_format,
            data_collection_table=data_collection_table
        )

        # Use text LLM for summary generation
        llm_summary = get_text_llm(model)
        response_summary = llm_summary.invoke(session_prompt)
        session_summary = response_summary.content.strip() if hasattr(response_summary, 'content') else ""

        output += "\n---\n\n## Group Session Plan\n\n"
        output += session_summary + "\n\n"

    except Exception as e:
        print(f"Error generating group session summary: {e}")
        output += "\n---\n\n## Group Session Plan\n\n"
        output += f"### Session Focus\nCollaborative communication skills with focus on:\n"
        for member in members:
            output += f"- {member['student_id']}: {', '.join(member['disorders'])}\n"
        output += "\n### Group Activities\n"
        output += "1. **Turn-taking activity** - Supports all members' communication goals\n"
        output += "2. **Collaborative problem-solving** - Targets pragmatic language skills\n"
        output += "3. **Peer modeling** - Members support each other's articulation/language goals\n\n"

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

# --- TABLE PARSING ---
def parse_table_request(file_path: Optional[str]) -> Tuple[List[Dict], str, Optional[str], Optional[int]]:
    """
    Parse CSV or Excel file containing student case information.

    Expected columns:
    - Student ID (e.g., S-001, PT-012)
    - Grade Level (e.g., Pre-K, 1st Grade)
    - Communication Disorder(s) (e.g., "1. Speech sound disorder & 6. expressive language disorders")

    Args:
        file_path: Path to the uploaded CSV or Excel file

    Returns:
        tuple: (list of task dicts, confirmation markdown string, id_prefix, id_start_number)
    """
    if not file_path or not os.path.exists(file_path):
        return [], "❌ No file uploaded or file not found.", None, None

    try:
        # Determine file type and read accordingly
        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext == '.csv':
            df = pd.read_csv(file_path)
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            return [], f"❌ Unsupported file type: {file_ext}. Please upload CSV or Excel file.", None, None

        # Normalize column headers (case-insensitive, strip whitespace)
        df.columns = df.columns.str.strip().str.lower()

        # Map common variations to standard column names
        column_mapping = {
            'student id': 'student_id',
            'studentid': 'student_id',
            'id': 'student_id',
            'grade level': 'grade',
            'gradelevel': 'grade',
            'grade': 'grade',
            'communication disorder(s)': 'disorders',
            'communication disorders': 'disorders',
            'disorder(s)': 'disorders',
            'disorders': 'disorders',
            'disorder': 'disorders'
        }

        # Rename columns based on mapping
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df.rename(columns={old_name: new_name}, inplace=True)

        # Verify required columns exist
        required_cols = ['student_id', 'grade', 'disorders']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            return [], f"❌ Missing required columns: {', '.join(missing_cols)}. Expected: Student ID, Grade Level, Communication Disorder(s)", None, None

        # Parse each row into a task
        tasks = []

        # Variables to track ID prefix and start number
        detected_prefix = None
        detected_start_number = None

        for idx, row in df.iterrows():
            try:
                # Extract Student ID and parse prefix/number
                student_id = str(row['student_id']).strip()

                # Parse Student ID (e.g., "S-001" -> prefix="S", number=1)
                # Handle formats: S-001, S001, PT-012, etc.
                id_match = re.match(r'^([A-Za-z]+)[-_]?(\d+)$', student_id)
                if id_match:
                    id_prefix = id_match.group(1)
                    id_number = int(id_match.group(2))

                    # Capture the first row's prefix and number for the UI fields
                    if detected_prefix is None:
                        detected_prefix = id_prefix
                        detected_start_number = id_number
                else:
                    # If format doesn't match, use as-is
                    id_prefix = "S"
                    id_number = idx + 1

                # Extract Grade Level
                grade = str(row['grade']).strip()

                # Normalize grade format
                if grade.lower() == 'pre-k' or grade.lower() == 'prek':
                    grade = 'Pre-K'
                elif grade.lower() == 'kindergarten' or grade.lower() == 'k':
                    grade = 'Kindergarten'
                elif re.match(r'^\d+$', grade):
                    # Just a number like "1" -> "1st Grade"
                    num = int(grade)
                    suffix = 'th' if 11 <= num <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(num % 10, 'th')
                    grade = f"{num}{suffix} Grade"

                # Verify grade is in allowed list
                if grade not in ALL_GRADES:
                    # Try to find closest match
                    grade = "1st Grade"  # Default fallback

                # Extract and parse disorders
                disorders_text = str(row['disorders']).strip()

                # Step 1: Extract characteristics from parentheses
                characteristics_list = []
                parentheses_matches = re.findall(r'\(([^)]+)\)', disorders_text)
                for match in parentheses_matches:
                    # Split by commas and clean up
                    chars = [c.strip() for c in match.split(',')]
                    characteristics_list.extend(chars)

                # Remove parentheses and their contents from disorders text
                disorders_text_clean = re.sub(r'\s*\([^)]+\)\s*', ' ', disorders_text)

                # Step 2: Split by common separators: &, and
                # Don't split by commas as they might be within disorder names
                disorder_parts = re.split(r'\s*&\s*|\s+and\s+', disorders_text_clean)

                # Step 3: Remove index numbers from each part
                disorder_parts = [re.sub(r'^\s*\d+\.?\s*', '', part).strip() for part in disorder_parts]

                # Step 4: Match disorders - prefer longer/more specific matches
                parsed_disorders = []

                # Sort DISORDER_TYPES by length (longest first) to match more specific terms first
                sorted_disorder_types = sorted(DISORDER_TYPES, key=len, reverse=True)

                for disorder_text in disorder_parts:
                    disorder_lower = disorder_text.strip().lower()

                    if not disorder_lower:
                        continue

                    # Try to match to known disorder types (bidirectional and fuzzy)
                    matched = False
                    for known_disorder in sorted_disorder_types:
                        known_lower = known_disorder.lower()

                        # Check both directions: disorder text in known, or known in disorder text
                        if known_lower in disorder_lower or disorder_lower in known_lower:
                            parsed_disorders.append(known_disorder)
                            matched = True
                            break

                        # Also check for key word matches for common variations
                        # e.g., "child apraxia" matches "childhood apraxia of speech"
                        disorder_words = set(disorder_lower.split())
                        known_words = set(known_lower.split())

                        # For Apraxia: check if both contain "apraxia"
                        if "apraxia" in disorder_words and "apraxia" in known_words:
                            parsed_disorders.append(known_disorder)
                            matched = True
                            break

                    # If no match, capitalize properly and add anyway
                    if not matched and len(disorder_lower) > 3:
                        disorder_title = ' '.join(word.capitalize() for word in disorder_text.split())
                        parsed_disorders.append(disorder_title)

                # Remove duplicates while preserving order
                parsed_disorders = list(dict.fromkeys(parsed_disorders))

                # If no disorders parsed, skip this row
                if not parsed_disorders:
                    print(f"Warning: Row {idx+1} has no valid disorders, skipping")
                    continue

                # Step 5: Combine characteristics into a single string
                characteristics = ', '.join(characteristics_list) if characteristics_list else ""

                # Determine default model from config
                default_model = FREE_MODELS[0] if FREE_MODELS else "Llama3.2"

                # Create task dict
                task = {
                    "id": student_id,
                    "grade": grade,
                    "disorders": parsed_disorders,
                    "model": default_model,
                    "count": 1,  # Each row is one student
                    "characteristics": characteristics
                }

                tasks.append(task)

            except Exception as row_error:
                print(f"Error parsing row {idx+1}: {row_error}")
                continue

        # Generate confirmation message
        if tasks:
            confirmation = f"### 📊 Parsed Table: {len(tasks)} Student(s)\n\n"
            confirmation += f"**File:** `{os.path.basename(file_path)}`\n\n"

            for i, task in enumerate(tasks, 1):
                confirmation += f"**Student {i} ({task['id']}):**\n"
                confirmation += f"- Grade: {task['grade']}\n"
                confirmation += f"- Disorders: {', '.join(task['disorders'])}\n"
                confirmation += f"- Model: {task['model']}\n"
                if task.get('characteristics'):
                    confirmation += f"- Characteristics: {task['characteristics']}\n"
                confirmation += "\n"

            confirmation += "✓ Review above. Adjust manually in rows below if needed, then click Generate."

            return tasks, confirmation, detected_prefix, detected_start_number
        else:
            return [], "❌ No valid student data found in table. Please check the file format.", None, None

    except Exception as e:
        print(f"Table parse error: {e}")
        return [], f"❌ Error parsing table: {str(e)}", None, None
