"""
Single case generation UI components with ASR support
"""
from datetime import datetime
import gradio as gr

from .config import *
from .models import current_case_data, generation_control
from .generation import generate_single_case
from .feedback import submit_feedback, handle_save
from .utils import ai_grammar_check, load_json, save_case_file, save_case_to_db, markdown_to_pdf
from .asr_processor import get_speech_analyzer

def create_single_case_ui(back_btn_handler):
    """Create the single case generation page UI with ASR."""
    
    with gr.Column(visible=False) as page:
        with gr.Row():
            gr.Markdown("# Generate Single Case")
            back_btn = gr.Button("🏠 Home", size="sm", variant="secondary")
        
        # LEFT AND RIGHT PANELS
        with gr.Row(equal_height=True):
            with gr.Column(scale=1, elem_classes="left-panel"):
                gr.Markdown("### Generation Parameters")
                
                # ASR SECTION - NEW!
                with gr.Accordion("🎤 Audio Analysis (Optional)", open=False):
                    gr.Markdown("""
                    Upload a speech sample to automatically analyze disorders and generate a matching case.
                    
                    **Supported formats:** WAV, MP3, M4A, FLAC
                    """)
                    
                    audio_upload = gr.Audio(
                        label="Upload Speech Sample",
                        type="filepath",
                        sources=["upload"]
                    )
                    
                    analyze_audio_btn = gr.Button("🎤 Analyze Audio", variant="secondary")
                    
                    audio_status = gr.Markdown("")
                    
                    with gr.Accordion("📝 Transcript & Analysis", open=False) as transcript_section:
                        transcript_display = gr.Textbox(
                            label="Deidentified Transcript",
                            lines=5,
                            interactive=False
                        )

                        patterns_display = gr.Textbox(
                            label="Detected Patterns",
                            lines=5,
                            interactive=False
                        )

                        ai_analysis_display = gr.Textbox(
                            label="AI Clinical Analysis",
                            lines=20,
                            interactive=False,
                            show_copy_button=True,
                            elem_classes="scrollable-textbox"
                        )
                
                gr.Markdown("---")
                gr.Markdown("### Manual Parameters")
                
                grade = gr.Dropdown(choices=ALL_GRADES, label="Grade Level", value="1st Grade")
                model = gr.Dropdown(choices=FREE_MODELS + PREMIUM_MODELS, label="AI Model", value=DEFAULT_MODEL if 'DEFAULT_MODEL' in dir() else "Llama3.2")
                disorders = gr.Dropdown(choices=DISORDER_TYPES, label="Disorders", multiselect=True, value=["Articulation Disorders"])
                population_spec = gr.Textbox(label="Population Characteristics", placeholder="e.g., second language learner", lines=2)

                with gr.Row():
                    generate_btn = gr.Button("Generate", variant="primary", scale=3)
                    stop_btn = gr.Button("⛔ Stop", variant="stop", visible=False, scale=1)
            
            with gr.Column(scale=1, elem_classes="right-panel"):
                gr.Markdown("### Advanced Options")
                reference_files = gr.File(label="Upload References", file_count="multiple")
        
        # OUTPUT
        gr.Markdown("### Generated Case File")
        output = gr.Markdown()
        
        # SAVE SECTION (existing code...)
        with gr.Group(visible=False, elem_classes="save-section") as save_section:
            gr.Markdown("### 💾 Save Case File")
            
            save_path_display = gr.Textbox(
                label="Save Path",
                interactive=False,
                value=""
            )
            
            with gr.Row():
                save_btn = gr.Button("💾 Save", variant="primary", size="sm")
                download_pdf_btn = gr.DownloadButton("📄 Download PDF", size="sm")
            
            save_status = gr.Markdown("")
        
        # FEEDBACK SECTION (existing code...)
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

        # BACK TO TOP BUTTON
        gr.Markdown("---")
        back_to_top_btn = gr.Button("⬆️ Back to Top", size="lg", variant="secondary")

    # Store components
    components = {
        "page": page,
        "back_btn": back_btn,
        "audio_upload": audio_upload,
        "analyze_audio_btn": analyze_audio_btn,
        "audio_status": audio_status,
        "transcript_display": transcript_display,
        "patterns_display": patterns_display,
        "ai_analysis_display": ai_analysis_display,
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
        "download_pdf_btn": download_pdf_btn,
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
        "feedback_status": feedback_status,
        "back_to_top_btn": back_to_top_btn
    }

    return components

def setup_single_case_events(components, generated_filename):
    """Setup event handlers for single case UI with ASR."""
    
    # Existing event handlers...
    # (keep all your existing code)
    
    # NEW: Audio analysis event handler
    def handle_audio_analysis(audio_path, analysis_model):
        """Analyze uploaded audio sample."""
        if not audio_path:
            return (
                "❌ Please upload an audio file first",
                "",
                "",
                "",
                gr.update(),
                gr.update(),
                gr.update()
            )
        
        try:
            analyzer = get_speech_analyzer()
            
            # Step 1: Transcribe
            yield (
                "🎤 Transcribing audio...",
                "",
                "",
                "",
                gr.update(),
                gr.update(),
                gr.update()
            )
            
            transcription = analyzer.transcribe_audio(audio_path)
            
            if not transcription["success"]:
                yield (
                    f"❌ Transcription failed: {transcription.get('error', 'Unknown error')}",
                    "",
                    "",
                    "",
                    gr.update(),
                    gr.update(),
                    gr.update()
                )
                return
            
            # Step 2: Deidentify
            yield (
                "🔒 Deidentifying transcript...",
                transcription["text"],
                "",
                "",
                gr.update(),
                gr.update(),
                gr.update()
            )
            
            deidentified = analyzer.deidentify_transcript(transcription["text"])
            
            # Step 3: Analyze patterns
            yield (
                "🔍 Analyzing speech patterns...",
                deidentified,
                "",
                "",
                gr.update(),
                gr.update(),
                gr.update()
            )
            
            patterns = analyzer.analyze_speech_patterns(
                deidentified,
                transcription["segments"]
            )
            
            patterns_text = f"""**Articulation Errors:**
{chr(10).join(f'- {e}' for e in patterns['articulation_errors']) if patterns['articulation_errors'] else 'None detected'}

**Phonological Patterns:**
{chr(10).join(f'- {p}' for p in patterns['phonological_patterns']) if patterns['phonological_patterns'] else 'None detected'}

**Fluency Issues:**
{chr(10).join(f'- {f}' for f in patterns['fluency_issues']) if patterns['fluency_issues'] else 'None detected'}

**Language Patterns:**
{chr(10).join(f'- {l}' for l in patterns['language_patterns']) if patterns['language_patterns'] else 'None detected'}

**Characteristics:**
{chr(10).join(f'- {c}' for c in patterns['characteristics'])}
"""
            
            # Step 4: AI Analysis
            yield (
                "🤖 Running AI analysis...",
                deidentified,
                patterns_text,
                "",
                gr.update(),
                gr.update(),
                gr.update()
            )
            
            ai_result = analyzer.identify_disorders_ai(
                deidentified,
                patterns,
                analysis_model
            )
            
            if ai_result["success"]:
                ai_analysis_text = f"""🎯 AI CLINICAL ANALYSIS
{'='*60}

{ai_result['analysis']}

{'='*60}
Analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            else:
                ai_analysis_text = f"❌ AI analysis failed: {ai_result['analysis']}"

            # Auto-fill form fields based on analysis - IMPROVED LOGIC
            suggested_disorders = []
            analysis_lower = ai_result['analysis'].lower()

            # Check for fluency first (prioritize based on detected disfluencies)
            has_fluency_disfluencies = patterns.get('fluency_metrics', {}).get('core_disfluencies', 0) > 0
            fluency_mentioned = any(term in analysis_lower for term in ['fluency', 'stuttering', 'stutter', 'disfluenc', 'repetition', 'prolongation', 'block'])

            if has_fluency_disfluencies or fluency_mentioned:
                suggested_disorders.append("Fluency")

            # Check for articulation (only if explicitly mentioned as a primary concern)
            if any(term in analysis_lower for term in ['articulation disorder', 'speech sound disorder', 'articulation error']):
                if 'no articulation' not in analysis_lower and 'no significant articulation' not in analysis_lower:
                    suggested_disorders.append("Articulation Disorders")

            # Check for phonological
            if 'phonological' in analysis_lower:
                suggested_disorders.append("Phonological Disorders")

            # Check for language
            if any(term in analysis_lower for term in ['language disorder', 'expressive language', 'receptive language', 'language delay']):
                suggested_disorders.append("Language Disorders")

            # If no disorders detected, use fluency if we have disfluencies, otherwise default to articulation
            if not suggested_disorders:
                if has_fluency_disfluencies:
                    suggested_disorders = ["Fluency"]
                else:
                    suggested_disorders = ["Articulation Disorders"]
            
            # Build population spec from detected patterns
            pop_spec_parts = []
            if patterns.get('fluency_metrics', {}).get('stuttering_frequency', 0) > 0:
                severity = patterns['fluency_metrics']['severity_rating']
                pop_spec_parts.append(f"{severity} stuttering")
            if patterns.get('language_patterns'):
                pop_spec_parts.append(patterns['language_patterns'][0])

            population_spec_text = f"Based on audio analysis: {'; '.join(pop_spec_parts)}" if pop_spec_parts else ""

            yield (
                "✅ Analysis complete! Review results and generate case below.",
                deidentified,
                patterns_text,
                ai_analysis_text,
                gr.update(value=suggested_disorders),
                gr.update(value=population_spec_text),
                gr.update()
            )
            
        except Exception as e:
            yield (
                f"❌ Error during analysis: {str(e)}",
                "",
                "",
                "",
                gr.update(),
                gr.update(),
                gr.update()
            )
    
    components["analyze_audio_btn"].click(
        fn=handle_audio_analysis,
        inputs=[components["audio_upload"], components["model"]],
        outputs=[
            components["audio_status"],
            components["transcript_display"],
            components["patterns_display"],
            components["ai_analysis_display"],
            components["disorders"],
            components["population_spec"],
            components["grade"]
        ]
    ).then(
        fn=lambda: None,
        js="""() => {
            // Scroll to Manual Parameters section and highlight auto-filled fields
            setTimeout(() => {
                // Find and scroll to Manual Parameters section
                const manualSection = Array.from(document.querySelectorAll('h3')).find(el =>
                    el.textContent.includes('Manual Parameters')
                );
                if (manualSection) {
                    manualSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

                    // Find the Disorders dropdown and Population Characteristics field
                    const labels = document.querySelectorAll('label');
                    const disordersLabel = Array.from(labels).find(el => el.textContent.includes('Disorders'));
                    const popLabel = Array.from(labels).find(el => el.textContent.includes('Population Characteristics'));

                    // Highlight the auto-filled fields
                    [disordersLabel, popLabel].forEach(label => {
                        if (label) {
                            const container = label.closest('.block');
                            if (container) {
                                container.style.transition = 'all 0.3s ease';
                                container.style.backgroundColor = '#e7f3ff';
                                container.style.border = '2px solid #4CAF50';
                                container.style.borderRadius = '8px';
                                container.style.padding = '8px';

                                // Remove highlight after 3 seconds
                                setTimeout(() => {
                                    container.style.backgroundColor = '';
                                    container.style.border = '';
                                    container.style.padding = '';
                                }, 3000);
                            }
                        }
                    });
                }
            }, 800);
        }"""
    )

    # Generate single case
    components["generate_btn"].click(
        fn=lambda: (gr.update(visible=True), gr.update(interactive=False)),
        outputs=[components["stop_btn"], components["generate_btn"]]
    ).then(
        fn=generate_single_case,
        inputs=[components["grade"], components["disorders"], components["model"],
                components["population_spec"], components["reference_files"]],
        outputs=[components["output"], generated_filename, components["generate_btn"],
                components["save_path_display"], components["stop_btn"], components["save_section"]]
    ).then(
        fn=lambda: gr.update(visible=False),
        outputs=components["stop_btn"]
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
    def handle_save_single(filepath):
        if not current_case_data["content"]:
            return "❌ No case to save"
        try:
            save_case_file(current_case_data["content"], filepath)
            save_case_to_db(current_case_data["metadata"])
            return f"✅ Case saved successfully!\n**Path:** `{filepath}`"
        except Exception as e:
            return f"❌ Error: {str(e)}"

    components["save_btn"].click(
        fn=handle_save_single,
        inputs=components["save_path_display"],
        outputs=components["save_status"]
    )

    # Download PDF button
    def prepare_download_single():
        if current_case_data["content"]:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_pdf = f"temp_case_{timestamp}.pdf"
            markdown_to_pdf(current_case_data["content"], temp_pdf)
            return temp_pdf
        return None

    components["download_pdf_btn"].click(
        fn=prepare_download_single,
        outputs=components["download_pdf_btn"]
    )

    # Feedback
    def toggle_feedback_single(category):
        is_other = (category == "Other")
        return gr.update(visible=is_other), gr.update(visible=is_other)

    components["feedback_cat"].change(
        fn=toggle_feedback_single,
        inputs=components["feedback_cat"],
        outputs=[components["detailed_feedback"], components["grammar_check_btn"]]
    )

    components["grammar_check_btn"].click(
        fn=ai_grammar_check,
        inputs=components["detailed_feedback"],
        outputs=components["detailed_feedback"]
    )

    # Submit feedback
    def handle_feedback_single(r1, r2, r3, r4, r5, cat, detailed):
        if not current_case_data["metadata"]:
            return "❌ Generate a case first", gr.update(), gr.update(value=3), gr.update(value=3), gr.update(value=3), gr.update(value=3), gr.update(value=3), gr.update(value=""), gr.update(value="General")

        ratings = {
            "clinical_accuracy": r1,
            "age_appropriateness": r2,
            "goal_quality": r3,
            "session_notes": r4,
            "background": r5
        }

        case_id = current_case_data["metadata"].get("timestamp", "unknown")
        return submit_feedback(case_id, ratings, cat, detailed)

    components["submit_feedback_btn"].click(
        fn=handle_feedback_single,
        inputs=[components["rating_clinical"], components["rating_age"], components["rating_goals"],
                components["rating_notes"], components["rating_background"],
                components["feedback_cat"], components["detailed_feedback"]],
        outputs=[components["feedback_status"], components["feedback_cat"],
                components["rating_clinical"], components["rating_age"], components["rating_goals"],
                components["rating_notes"], components["rating_background"],
                components["detailed_feedback"], components["feedback_cat"]]
    )

    # Back to top button
    components["back_to_top_btn"].click(
        fn=lambda: None,
        js="() => { window.scrollTo({ top: 0, behavior: 'smooth' }); }"
    )