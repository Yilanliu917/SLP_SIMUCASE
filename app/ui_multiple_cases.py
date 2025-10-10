"""
Multiple cases generation UI components and event handlers
"""
from datetime import datetime
import gradio as gr

from .config import *
from .models import multiple_cases_batch, generation_control
from .generation import generate_multiple_cases, parse_complex_request
from .feedback import submit_feedback
from .utils import ai_grammar_check, load_json, save_case_file, save_case_to_db

def create_multiple_cases_ui():
    """Create the multiple cases generation page UI."""
    
    with gr.Column(visible=False) as page:
        with gr.Row():
            gr.Markdown("# Generate Multiple Cases")
            back_btn = gr.Button("← Back", size="sm")
        
        # CHAT BOX
        with gr.Group(elem_classes="left-panel"):
            gr.Markdown("### 💬 Natural Language Request")
            chat_request = gr.Textbox(
                label="Describe what you want to generate",
                placeholder='Example: "generate 20 students with articulation disorders, using GPT-4o, 5 are pre-k, rest kindergarten"',
                lines=4
            )
            with gr.Row():
                parse_btn = gr.Button("🔍 Parse Request", size="sm", variant="secondary")
                clear_chat_btn = gr.Button("Clear", size="sm")
            
            parsed_output = gr.Markdown(label="Parsed Tasks", visible=False)
        
        # MANUAL CONFIGURATION
        with gr.Accordion("Manual Configuration (Alternative)", open=True):
            gr.Markdown("### Condition Rows")
            
            # Custom ID options
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
                        placeholder="e.g., has final consonant deletion",
                        lines=1,
                        scale=2
                    )
                    
                    condition_rows.append([grade_multi, disorders_multi, num_students_multi, model_multi, characteristics_multi])
                    condition_row_components.append(row)
            
            with gr.Row():
                add_row_btn = gr.Button("➕ Add Row", size="sm")
                remove_row_btn = gr.Button("➖ Remove Row", size="sm")
        
        generate_btn = gr.Button("🚀 Generate All Cases", variant="primary", size="lg")
        stop_btn = gr.Button("⛔ Stop Generation", size="sm", variant="stop", visible=False)
        
        # OUTPUT
        gr.Markdown("### Generated Cases")
        output = gr.Markdown()
        
        # SAVE SECTION
        with gr.Group(visible=False, elem_classes="save-section") as save_section:
            gr.Markdown("### 💾 Save Batch")
            
            save_path = gr.Textbox(
                label="Save Path",
                interactive=False,
                value=""
            )
            
            with gr.Row():
                save_btn = gr.Button("💾 Save", variant="primary", size="sm")
                save_as_btn = gr.DownloadButton("📥 Save As", size="sm")
            
            save_status = gr.Markdown("")
        
        # FEEDBACK SECTION
        with gr.Accordion("Provide Feedback", open=False):
            gr.Markdown("### Evaluate Generated Cases")
            
            case_selector = gr.Dropdown(
                choices=["Whole Batch"],
                label="Select Case to Evaluate",
                value="Whole Batch"
            )
            
            with gr.Row():
                rating_clinical = gr.Slider(1, 5, value=3, step=1, label="Clinical Accuracy")
                rating_age = gr.Slider(1, 5, value=3, step=1, label="Age Appropriate")
                rating_goals = gr.Slider(1, 5, value=3, step=1, label="Goal Quality")
            
            with gr.Row():
                rating_notes = gr.Slider(1, 5, value=3, step=1, label="Session Notes")
                rating_background = gr.Slider(1, 5, value=3, step=1, label="Background")
            
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
                placeholder="Provide specific feedback..."
            )
            
            grammar_check_btn = gr.Button("AI Grammar Check", size="sm", visible=False)
            
            with gr.Row():
                submit_feedback_btn = gr.Button("Submit Feedback", variant="secondary")
                add_another_feedback_btn = gr.Button("Add Another Evaluation", variant="secondary", size="sm")
            
            feedback_status = gr.Markdown("")
    
    components = {
        "page": page,
        "back_btn": back_btn,
        "chat_request": chat_request,
        "parse_btn": parse_btn,
        "clear_chat_btn": clear_chat_btn,
        "parsed_output": parsed_output,
        "use_custom_ids": use_custom_ids,
        "custom_id_prefix": custom_id_prefix,
        "custom_id_start": custom_id_start,
        "visible_row_count": visible_row_count,
        "condition_rows": condition_rows,
        "condition_row_components": condition_row_components,
        "add_row_btn": add_row_btn,
        "remove_row_btn": remove_row_btn,
        "generate_btn": generate_btn,
        "stop_btn": stop_btn,
        "output": output,
        "save_section": save_section,
        "save_path": save_path,
        "save_btn": save_btn,
        "save_as_btn": save_as_btn,
        "save_status": save_status,
        "case_selector": case_selector,
        "rating_clinical": rating_clinical,
        "rating_age": rating_age,
        "rating_goals": rating_goals,
        "rating_notes": rating_notes,
        "rating_background": rating_background,
        "feedback_cat": feedback_cat,
        "detailed_feedback": detailed_feedback,
        "grammar_check_btn": grammar_check_btn,
        "submit_feedback_btn": submit_feedback_btn,
        "add_another_feedback_btn": add_another_feedback_btn,
        "feedback_status": feedback_status
    }
    
    return components

def setup_multiple_cases_events(components):
    """Setup event handlers for multiple cases UI."""
    
    # Row management
    def update_row_visibility(count):
        return [gr.update(visible=(i < count)) for i in range(MAX_CONDITION_ROWS)]
    
    def add_row(count):
        return min(count + 1, MAX_CONDITION_ROWS)
    
    def remove_row(count):
        return max(count - 1, 1)
    
    components["add_row_btn"].click(
        fn=add_row,
        inputs=components["visible_row_count"],
        outputs=components["visible_row_count"]
    ).then(
        fn=update_row_visibility,
        inputs=components["visible_row_count"],
        outputs=components["condition_row_components"]
    )
    
    components["remove_row_btn"].click(
        fn=remove_row,
        inputs=components["visible_row_count"],
        outputs=components["visible_row_count"]
    ).then(
        fn=update_row_visibility,
        inputs=components["visible_row_count"],
        outputs=components["condition_row_components"]
    )
    
    # Parse natural language
    def handle_parse_request(request):
        if not request.strip():
            return gr.update(visible=False, value="")
        
        tasks, breakdown = parse_complex_request(request)
        
        if tasks:
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
                    updates.extend([gr.update() for _ in range(5)])
            
            visible_count = min(len(tasks), MAX_CONDITION_ROWS)
            
            return [gr.update(visible=True, value=breakdown), visible_count] + updates
        else:
            return [gr.update(visible=True, value=breakdown), gr.update()] + [gr.update() for _ in range(MAX_CONDITION_ROWS * 5)]
    
    all_row_inputs = [item for sublist in components["condition_rows"] for item in sublist]
    
    components["parse_btn"].click(
        fn=handle_parse_request,
        inputs=components["chat_request"],
        outputs=[components["parsed_output"], components["visible_row_count"]] + all_row_inputs
    )
    
    components["clear_chat_btn"].click(
        fn=lambda: ("", gr.update(visible=False)),
        outputs=[components["chat_request"], components["parsed_output"]]
    )
    
    # Toggle custom IDs
    def toggle_custom_ids(use_custom):
        return gr.update(visible=use_custom), gr.update(visible=use_custom)
    
    components["use_custom_ids"].change(
        fn=toggle_custom_ids,
        inputs=components["use_custom_ids"],
        outputs=[components["custom_id_prefix"], components["custom_id_start"]]
    )
    
    # Generate multiple cases
    def handle_multiple_generation(row_count, use_custom, id_prefix, id_start, *row_inputs):
        tasks = []
        
        for i in range(row_count):
            grade = row_inputs[i * 5]
            disorders = row_inputs[i * 5 + 1]
            num = row_inputs[i * 5 + 2]
            model = row_inputs[i * 5 + 3]
            characteristics = row_inputs[i * 5 + 4]
            
            if disorders:
                tasks.append({
                    "grade": grade,
                    "disorders": disorders,
                    "count": int(num),
                    "model": model,
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
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            full_path = f"{save_path}multiple_cases_{timestamp}.md"
            yield (output,
                   gr.update(visible=True, value=full_path),
                   gr.update(choices=case_choices),
                   btn_state)
    
    components["generate_btn"].click(
        fn=handle_multiple_generation,
        inputs=[components["visible_row_count"], components["use_custom_ids"],
                components["custom_id_prefix"], components["custom_id_start"]] + all_row_inputs,
        outputs=[components["output"], components["save_path"], components["case_selector"], components["generate_btn"]]
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
        outputs=[components["stop_btn"], components["generate_btn"], components["output"]]
    )
    
    # Save functionality
    def handle_save_multiple(filepath):
        if not multiple_cases_batch["cases"]:
            return "❌ No cases to save"
        
        try:
            combined_content = f"# Multiple Cases Batch\n**Batch ID:** {multiple_cases_batch['batch_id']}\n\n"
            combined_content += "\n\n".join([case["content"] for case in multiple_cases_batch["cases"]])
            
            save_case_file(combined_content, filepath)
            
            for case in multiple_cases_batch["cases"]:
                save_case_to_db({
                    **case["metadata"],
                    "batch_id": multiple_cases_batch["batch_id"],
                    "filepath": filepath
                })
            
            return f"✅ Batch saved successfully!\n**Path:** `{filepath}`\n**Batch ID:** {multiple_cases_batch['batch_id']}"
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    components["save_btn"].click(
        fn=handle_save_multiple,
        inputs=components["save_path"],
        outputs=components["save_status"]
    )
    
    # Download
    def prepare_download_multiple():
        if multiple_cases_batch["cases"]:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_file = f"temp_batch_{timestamp}.md"
            combined = "\n\n".join([case["content"] for case in multiple_cases_batch["cases"]])
            save_case_file(combined, temp_file)
            return temp_file
        return None
    
    components["save_as_btn"].click(
        fn=prepare_download_multiple,
        outputs=components["save_as_btn"]
    )
    
    # Feedback
    def toggle_feedback_multi(category):
        is_other = (category == "Other")
        return gr.update(visible=is_other), gr.update(visible=is_other)
    
    components["feedback_cat"].change(
        fn=toggle_feedback_multi,
        inputs=components["feedback_cat"],
        outputs=[components["detailed_feedback"], components["grammar_check_btn"]]
    )
    
    components["grammar_check_btn"].click(
        fn=ai_grammar_check,
        inputs=components["detailed_feedback"],
        outputs=components["detailed_feedback"]
    )
    
    # Submit feedback
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
        
        if selected_case == "Whole Batch":
            target_id = multiple_cases_batch["batch_id"]
        else:
            case_num = int(selected_case.split(":")[0].split()[1]) - 1
            target_id = multiple_cases_batch["cases"][case_num]["id"]
        
        return submit_feedback(target_id, ratings, cat, detailed)
    
    components["submit_feedback_btn"].click(
        fn=handle_feedback_multiple,
        inputs=[components["case_selector"], components["rating_clinical"], components["rating_age"],
                components["rating_goals"], components["rating_notes"], components["rating_background"],
                components["feedback_cat"], components["detailed_feedback"]],
        outputs=[components["feedback_status"], components["feedback_cat"], components["rating_clinical"],
                components["rating_age"], components["rating_goals"], components["rating_notes"],
                components["rating_background"], components["detailed_feedback"], components["feedback_cat"]]
    )
    
    # Reset feedback form
    def reset_feedback_form():
        return (gr.update(value=3), gr.update(value=3), gr.update(value=3),
                gr.update(value=3), gr.update(value=3),
                gr.update(value=""), gr.update(value="General"), "")
    
    components["add_another_feedback_btn"].click(
        fn=reset_feedback_form,
        outputs=[components["rating_clinical"], components["rating_age"], components["rating_goals"],
                components["rating_notes"], components["rating_background"],
                components["detailed_feedback"], components["feedback_cat"], components["feedback_status"]]
    )
