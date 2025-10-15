"""
Main application file for SLP SimuCase Generator
Run this file to start the application: python main.py
"""
import gradio as gr
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import UI modules from app package
from app.ui_single_case import create_single_case_ui, setup_single_case_events
from app.ui_multiple_cases import create_multiple_cases_ui, setup_multiple_cases_events
from app.ui_group_session import create_group_session_ui, setup_group_session_events

def create_app():
    """Create the complete Gradio application."""
    
    with gr.Blocks(theme=gr.themes.Soft(), title="SLP SimuCase Generator", css="""
        .title {text-align: center; margin: 20px 0;}
        .cover-buttons {max-width: 600px; margin: 0 auto;}
        .left-panel {padding: 20px; background: #f8f9fa; border-radius: 10px;}
        .right-panel {padding: 20px; background: #f0f0f0; border-radius: 10px;}
        .save-section {padding: 15px; background: #e7f3ff; border-radius: 8px; margin: 15px 0;}
        .scrollable-textbox textarea {
            max-height: 500px !important;
            overflow-y: auto !important;
        }
    """) as app:
        
        # Global state for generated filename
        generated_filename = gr.State(None)
        
        # === COVER PAGE ===
        with gr.Column(visible=True, elem_classes="cover-buttons") as cover_page:
            gr.Markdown("# SLP SimuCase Generator", elem_classes="title")
            gr.Markdown("### Professional Case File Generation System", elem_classes="title")
            
            gr.Markdown("")
            btn_single = gr.Button("Generate Single Case", size="lg", variant="primary")
            btn_multiple = gr.Button("Generate Multiple Cases", size="lg", variant="primary")
            btn_group = gr.Button("Generate Group Session", size="lg", variant="primary")
        
        # === CREATE ALL PAGE UIs ===
        single_case_components = create_single_case_ui(lambda: gr.update(visible=False))
        multiple_cases_components = create_multiple_cases_ui()
        group_session_components = create_group_session_ui()
        
        # === NAVIGATION FUNCTIONS ===
        def show_single():
            return {
                cover_page: gr.update(visible=False),
                single_case_components["page"]: gr.update(visible=True),
                multiple_cases_components["page"]: gr.update(visible=False),
                group_session_components["page"]: gr.update(visible=False)
            }
        
        def show_multiple():
            return {
                cover_page: gr.update(visible=False),
                single_case_components["page"]: gr.update(visible=False),
                multiple_cases_components["page"]: gr.update(visible=True),
                group_session_components["page"]: gr.update(visible=False)
            }
        
        def show_group():
            return {
                cover_page: gr.update(visible=False),
                single_case_components["page"]: gr.update(visible=False),
                multiple_cases_components["page"]: gr.update(visible=False),
                group_session_components["page"]: gr.update(visible=True)
            }
        
        def show_cover():
            return {
                cover_page: gr.update(visible=True),
                single_case_components["page"]: gr.update(visible=False),
                multiple_cases_components["page"]: gr.update(visible=False),
                group_session_components["page"]: gr.update(visible=False)
            }
        
        # === COVER PAGE NAVIGATION ===
        btn_single.click(
            show_single,
            outputs=[cover_page, single_case_components["page"],
                    multiple_cases_components["page"], group_session_components["page"]]
        )
        
        btn_multiple.click(
            show_multiple,
            outputs=[cover_page, single_case_components["page"],
                    multiple_cases_components["page"], group_session_components["page"]]
        )
        
        btn_group.click(
            show_group,
            outputs=[cover_page, single_case_components["page"],
                    multiple_cases_components["page"], group_session_components["page"]]
        )
        
        # === BACK BUTTON HANDLERS ===
        single_case_components["back_btn"].click(
            show_cover,
            outputs=[cover_page, single_case_components["page"],
                    multiple_cases_components["page"], group_session_components["page"]]
        )

        # Home button handler for multiple cases page
        multiple_cases_components["home_btn"].click(
            show_cover,
            outputs=[cover_page, single_case_components["page"],
                    multiple_cases_components["page"], group_session_components["page"]]
        )

        group_session_components["back_btn"].click(
            show_cover,
            outputs=[cover_page, single_case_components["page"],
                    multiple_cases_components["page"], group_session_components["page"]]
        )
        
        # === SETUP EVENT HANDLERS FOR EACH PAGE ===
        setup_single_case_events(single_case_components, generated_filename)
        setup_multiple_cases_events(multiple_cases_components)
        setup_group_session_events(group_session_components)
    
    return app

if __name__ == "__main__":
    app = create_app()

    # Local development: Use 127.0.0.1 and auto-open browser
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True
    )