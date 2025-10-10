"""
Feedback collection and analysis functions
"""
from datetime import datetime
from typing import Dict, Tuple
import gradio as gr

from .config import FEEDBACK_LOG, FEEDBACK_CATEGORIES
from .utils import load_json, save_json, analyze_feedback_category

def submit_feedback(case_id: str, ratings: Dict[str, int], category: str, 
                   detailed_feedback: str) -> Tuple:
    """Submit feedback for a generated case."""
    
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

def handle_save(filepath: str, case_data: Dict) -> str:
    """Save the current case to specified path."""
    from utils import save_case_file, save_case_to_db
    
    if not case_data.get("content"):
        return "❌ No case to save. Generate a case first."
    
    try:
        save_case_file(case_data["content"], filepath)
        case_id = save_case_to_db({**case_data["metadata"], "filepath": filepath})
        case_data["case_id"] = case_id
        return f"✅ Case saved successfully!\n**Path:** `{filepath}`\n**Case ID:** {case_id}"
    except Exception as e:
        return f"❌ Error saving file: {str(e)}"
