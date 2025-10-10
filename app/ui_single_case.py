"""
Single case generation UI components and event handlers
"""
from datetime import datetime
import gradio as gr

from .config import *
from .models import current_case_data, generation_control
from .generation import generate_single_case
from .feedback import submit_feedback, handle_save
from .utils import ai_grammar_check, load_json, save_case_file

def create_single_case_ui(back_btn_handler):
    """Create the single case generation page UI."""
    
    with gr.Column(visible=False) as page:
        with gr.Row():
            gr.Markdown("# Generate Single Case")
            back_btn = gr.Button("← Back", size="sm")
        
        # LEFT AND RIGHT PANELS
        with gr.Row(equal_height=True):
            with gr.Column(scale=1, elem_classes="left-panel"):
                gr.Markdown("### Generation Parameters")
                
                grade = gr.Dropdown(choices=ALL_GRADES, label="Grade Level", value="1st Grade")
                model = gr.Dropdown(choices=FREE_MODELS + PREMIUM_MODELS, label="AI Model", value="Llama3.2")
                disorders = gr.Dropdown(choices=DISORDER_TYPES, label="Disorders", multiselect=True, value=["Articulation Disorders"])
                population_spec = gr.Textbox(label="Population Characteristics", placeholder="e.g., second language learner", lines=2)
                
                generate_btn = gr.Button("Generate", size="sm", variant="primary")
                stop_btn = gr.Button("⛔ Stop", size="sm", variant="stop", visible=False)
            
            with gr.Column(scale=1, elem_classes="right-panel"):
                gr.Markdown("### Advanced Options")
                reference_files = gr.File(label="Upload References", file_count="multiple")
        
        # OUTPUT
        gr.Markdown("### Generated Case File")
        output = gr.Markdown()
        
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
            feedback_cat = gr.Dropdown(choices=["General", "Other"] + categories_list, label="Category", value="General")
            
            detailed_feedback = gr.Textbox(label="Detailed Feedback", lines=4, visible=False, placeholder="Provide specific feedback...")
            grammar_check_btn = gr.Button("AI Grammar Check", size="sm", visible=False)
            
            submit_feedback_btn = gr.Button("Submit Feedback", variant="secondary")
            feedback_status = gr.Markdown("")
    
    # Store components for event handlers
    components = {
        "page": page,
        "back_btn": back_btn,
        "grade": grade,
        "model": model,
        "disorders": disorders,
        "population_spec": population_spec,
        "reference_files": reference_files,
        "generate_btn": generate_btn,
        "stop_btn": stop_btn,
        "output": output,
        "save_section": save_section,
        "save_path_display": save_path_display,
        "save_btn": save_btn,
        "save_as_btn": save_as_btn,
        "save_status": save_status,
        "rating_clinical": rating_clinical,
        "rating_age": rating_age,
        "rating_goals": rating_goals,
        "rating_notes": rating_notes,
        "rating_background": rating_background,
        "feedback_cat": feedback_cat,
        "detailed_feedback": detailed_feedback,
        "grammar_check_btn": grammar_check_btn,
        "submit_feedback_btn": submit_feedback_btn,
        "feedback_status": feedback_status
    }
    
    return components

def setup_single_case_events(components, generated_filename):
    """Setup event handlers for single case UI."""
    
    # Back button
    components["back_btn"].click(
        fn=lambda: gr.update(visible=False),
        outputs=components["page"]
    )
    
    # Generation with streaming
    def handle_generation(grade, disorders, model, pop_spec, refs):
        if not disorders:
            yield "⚠️ Please select at least one disorder", None, gr.update(), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
            return
        
        for output, filename, btn_state, path_display, stop_vis, save_vis in generate_single_case(grade, disorders, model, pop_spec, refs):
            yield output, filename, btn_state, path_display, stop_vis, save_vis
    
    components["generate_btn"].click(
        fn=handle_generation,
        inputs=[components["grade"], components["disorders"], components["model"], components["population_spec"], components["reference_files"]],
        outputs=[components["output"], generated_filename, components["generate_btn"], components["save_path_display"], components["stop_btn"], components["save_section"]]
    )
    
    # Stop button
    def stop_generation():
        generation_control["should_stop"] = True
        return gr.update(visible=False), gr.update(interactive=True, variant="primary"), "⛔ Stopping generation..."
    
    components["stop_btn"].click(
        fn=stop_generation,
        outputs=[components["stop_btn"], components["generate_btn"], components["output"]]
    )
    
    # Save functionality
    components["save_btn"].click(
        fn=lambda fp: handle_save(fp, current_case_data),
        inputs=components["save_path_display"],
        outputs=components["save_status"]
    )
    
    # Download functionality
    def prepare_download():
        if current_case_data["content"]:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_file = f"temp_case_{timestamp}.md"
            save_case_file(current_case_data["content"], temp_file)
            return temp_file
        return None
    
    components["save_as_btn"].click(fn=prepare_download, outputs=components["save_as_btn"])
    
    # Show/hide detailed feedback
    def toggle_feedback_fields(category):
        is_other = (category == "Other")
        return gr.update(visible=is_other), gr.update(visible=is_other)
    
    components["feedback_cat"].change(
        fn=toggle_feedback_fields,
        inputs=components["feedback_cat"],
        outputs=[components["detailed_feedback"], components["grammar_check_btn"]]
    )
    
    # Grammar check
    components["grammar_check_btn"].click(
        fn=ai_grammar_check,
        inputs=components["detailed_feedback"],
        outputs=components["detailed_feedback"]
    )
    
    # Submit feedback
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
    
    components["submit_feedback_btn"].click(
        fn=handle_feedback,
        inputs=[components["rating_clinical"], components["rating_age"], components["rating_goals"],
                components["rating_notes"], components["rating_background"], components["feedback_cat"],
                components["detailed_feedback"]],
        outputs=[components["feedback_status"], components["feedback_cat"], components["rating_clinical"],
                components["rating_age"], components["rating_goals"], components["rating_notes"],
                components["rating_background"], components["detailed_feedback"], components["feedback_cat"]]
    )
