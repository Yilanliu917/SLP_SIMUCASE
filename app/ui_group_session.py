"""
Group session generation UI components and event handlers
"""
import os
from datetime import datetime
import gradio as gr

from .config import *
from .models import group_session_data, generation_control
from .generation import generate_group_session
from .feedback import submit_feedback
from .utils import ai_grammar_check, load_json, save_case_file, save_case_to_db, check_grade_compatibility, check_disorder_compatibility

def create_group_session_ui():
    """Create the group session generation page UI."""
    
    with gr.Column(visible=False) as page:
        with gr.Row():
            gr.Markdown("# Generate Group Session")
            back_btn = gr.Button("← Back", size="sm")
        
        gr.Markdown("### Configure Group Members")
        gr.Markdown("*Create therapy groups following clinical grouping strategies*")
        
        # Group size selection
        group_size = gr.Radio(choices=[2, 3, 4], label="Group Size", value=2)
        
        # Member configurations
        member_configs = []
        member_rows = []
        
        for i in range(4):
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
            stop_btn = gr.Button("⛔ Stop", size="sm", variant="stop", visible=False)
        
        # Output
        gr.Markdown("### Group Session Plan")
        output = gr.Markdown()
        
        # Save section
        with gr.Group(visible=False, elem_classes="save-section") as save_section:
            gr.Markdown("### 💾 Save Group Session")
            
            save_path = gr.Textbox(
                label="Save Path",
                interactive=False,
                value=""
            )
            
            with gr.Row():
                save_btn = gr.Button("💾 Save", variant="primary", size="sm")
                save_as_btn = gr.DownloadButton("📥 Save As", size="sm")
            
            save_status = gr.Markdown("")
        
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
            feedback_cat = gr.Dropdown(
                choices=["General", "Other"] + categories_list,
                label="Category",
                value="General"
            )
            
            detailed_feedback = gr.Textbox(
                label="Detailed Feedback",
                lines=4,
                visible=False,
                placeholder="Provide specific feedback about the group configuration..."
            )
            
            grammar_check_btn = gr.Button("AI Grammar Check", size="sm", visible=False)
            
            submit_feedback_btn = gr.Button("Submit Feedback", variant="secondary")
            feedback_status = gr.Markdown("")
    
    components = {
        "page": page,
        "back_btn": back_btn,
        "group_size": group_size,
        "member_configs": member_configs,
        "member_rows": member_rows,
        "compatibility_status": compatibility_status,
        "model_group": model_group,
        "search_existing": search_existing,
        "check_compat_btn": check_compat_btn,
        "generate_group_btn": generate_group_btn,
        "stop_btn": stop_btn,
        "output": output,
        "save_section": save_section,
        "save_path": save_path,
        "save_btn": save_btn,
        "save_as_btn": save_as_btn,
        "save_status": save_status,
        "rating_grade_accuracy": rating_grade_accuracy,
        "rating_disorder_accuracy": rating_disorder_accuracy,
        "rating_goal_accuracy": rating_goal_accuracy,
        "rating_notes_completeness": rating_notes_completeness,
        "rating_setup_feasibility": rating_setup_feasibility,
        "feedback_cat": feedback_cat,
        "detailed_feedback": detailed_feedback,
        "grammar_check_btn": grammar_check_btn,
        "submit_feedback_btn": submit_feedback_btn,
        "feedback_status": feedback_status
    }
    
    return components

def setup_group_session_events(components):
    """Setup event handlers for group session UI."""
    
    # Update member row visibility based on group size
    def update_member_visibility(size):
        return [gr.update(visible=(i < size)) for i in range(4)]
    
    components["group_size"].change(
        fn=update_member_visibility,
        inputs=components["group_size"],
        outputs=components["member_rows"]
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
    
    all_member_inputs = [item for sublist in components["member_configs"] for item in sublist]
    
    components["check_compat_btn"].click(
        fn=check_compatibility,
        inputs=[components["group_size"]] + all_member_inputs,
        outputs=components["compatibility_status"]
    )
    
    # Generate group session
    def handle_group_generation(size, model, search, *member_inputs):
        grades = [member_inputs[i*2] for i in range(size)]
        disorders = [member_inputs[i*2+1] for i in range(size)]
        
        for output, filename, btn_state, stop_vis in generate_group_session(
            size, grades, disorders, model, search
        ):
            if filename:
                full_path = f"{DEFAULT_OUTPUT_PATH}{os.path.basename(filename)}"
                yield output, gr.update(visible=True, value=full_path), btn_state, stop_vis
            else:
                yield output, gr.update(), btn_state, stop_vis
    
    components["generate_group_btn"].click(
        fn=lambda: (gr.update(interactive=False), gr.update(visible=True)),
        outputs=[components["generate_group_btn"], components["stop_btn"]]
    ).then(
        fn=handle_group_generation,
        inputs=[components["group_size"], components["model_group"], components["search_existing"]] + all_member_inputs,
        outputs=[components["output"], components["save_path"], components["generate_group_btn"], components["stop_btn"]]
    ).then(
        fn=lambda: gr.update(visible=True),
        outputs=components["save_section"]
    )
    
    # Stop button
    def stop_generation():
        generation_control["should_stop"] = True
        return gr.update(visible=False), gr.update(interactive=True, variant="primary"), "⛔ Stopping generation..."
    
    components["stop_btn"].click(
        fn=stop_generation,
        outputs=[components["stop_btn"], components["generate_group_btn"], components["output"]]
    )
    
    # Save group session
    def handle_save_group(filepath):
        if not group_session_data["session_id"]:
            return "❌ No group session to save"
        
        try:
            content = f"# Group Session\n**Session ID:** {group_session_data['session_id']}\n"
            for member in group_session_data["members"]:
                content += member["output"]
            
            save_case_file(content, filepath)
            
            save_case_to_db({
                "type": "group",
                "session_id": group_session_data["session_id"],
                "members": group_session_data["members"],
                "filepath": filepath
            })
            
            return f"✅ Group session saved!\n**Path:** `{filepath}`\n**Session ID:** {group_session_data['session_id']}"
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    components["save_btn"].click(
        fn=handle_save_group,
        inputs=components["save_path"],
        outputs=components["save_status"]
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
    
    components["save_as_btn"].click(
        fn=prepare_download_group,
        outputs=components["save_as_btn"]
    )
    
    # Feedback
    def toggle_feedback_group(category):
        is_other = (category == "Other")
        return gr.update(visible=is_other), gr.update(visible=is_other)
    
    components["feedback_cat"].change(
        fn=toggle_feedback_group,
        inputs=components["feedback_cat"],
        outputs=[components["detailed_feedback"], components["grammar_check_btn"]]
    )
    
    components["grammar_check_btn"].click(
        fn=ai_grammar_check,
        inputs=components["detailed_feedback"],
        outputs=components["detailed_feedback"]
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
    
    components["submit_feedback_btn"].click(
        fn=handle_feedback_group,
        inputs=[components["rating_grade_accuracy"], components["rating_disorder_accuracy"],
                components["rating_goal_accuracy"], components["rating_notes_completeness"],
                components["rating_setup_feasibility"], components["feedback_cat"], components["detailed_feedback"]],
        outputs=[components["feedback_status"], components["feedback_cat"], components["rating_grade_accuracy"],
                components["rating_disorder_accuracy"], components["rating_goal_accuracy"],
                components["rating_notes_completeness"], components["rating_setup_feasibility"],
                components["detailed_feedback"], components["feedback_cat"]]
    )
