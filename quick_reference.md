# Quick Reference Guide

## 🚀 Getting Started

```bash
# Start application
python app.py

# Install dependencies
pip install -r requirements.txt

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

---

## 📁 File Purpose (One-Line Summary)

| File | Purpose |
|------|---------|
| `app.py` | Main entry point, navigation, launches app |
| `config.py` | All constants and configuration |
| `models.py` | Data structures and global state |
| `utils.py` | Helper functions used across modules |
| `generation.py` | Case generation algorithms |
| `feedback.py` | Feedback collection system |
| `ui_single_case.py` | Single case page UI and events |
| `ui_multiple_cases.py` | Multiple cases page UI and events |
| `ui_group_session.py` | Group session page UI and events |

---

## 🔍 Where to Find Things

### "Where is the X?"

| What You're Looking For | File | Function/Variable |
|------------------------|------|------------------|
| Disorder types list | `config.py` | `DISORDER_TYPES` |
| Grade levels list | `config.py` | `ALL_GRADES` |
| Available models | `config.py` | `FREE_MODELS`, `PREMIUM_MODELS` |
| Model ID mapping | `config.py` | `MODEL_MAP` |
| Default save path | `config.py` | `DEFAULT_OUTPUT_PATH` |
| Data models | `models.py` | `SimuCaseFile`, `StudentProfile`, etc. |
| Current case state | `models.py` | `current_case_data` |
| LLM initialization | `utils.py` | `get_llm()` |
| File save/load | `utils.py` | `save_case_file()`, `load_json()` |
| Single case generation | `generation.py` | `generate_single_case()` |
| Multiple cases generation | `generation.py` | `generate_multiple_cases()` |
| Group session generation | `generation.py` | `generate_group_session()` |
| NLP parsing | `generation.py` | `parse_complex_request()` |
| Feedback submission | `feedback.py` | `submit_feedback()` |
| Grammar checking | `utils.py` | `ai_grammar_check()` |
| Grade compatibility | `utils.py` | `check_grade_compatibility()` |
| Disorder compatibility | `utils.py` | `check_disorder_compatibility()` |

---

## ⚡ Common Tasks

### Add a New Model

```python
# 1. Edit config.py
FREE_MODELS.append("New Model")
MODEL_MAP["New Model"] = "model-id"

# 2. Edit utils.py -> get_llm()
elif model_name == "New Model":
    return ChatOllama(model="model-id", temperature=0.7)
```

### Add a New Disorder

```python
# Edit config.py
DISORDER_TYPES.append("New Disorder Name")
```

### Add a New Grade Level

```python
# Edit config.py
ALL_GRADES.append("New Grade")
```

### Change Default Save Path

```python
# Edit config.py
DEFAULT_OUTPUT_PATH = "new/path/"
```

### Modify Generation Prompt

```python
# Edit files in prompts/ directory
# OR
# Edit generation.py -> load_prompt() usage
```

### Change UI Layout

```python
# Single case: ui_single_case.py -> create_single_case_ui()
# Multiple cases: ui_multiple_cases.py -> create_multiple_cases_ui()
# Group session: ui_group_session.py -> create_group_session_ui()
```

### Add New Event Handler

```python
# 1. Add button/component in create_*_ui()
new_button = gr.Button("New Feature")

# 2. Add to components dictionary
components["new_button"] = new_button

# 3. Add handler in setup_*_events()
def handle_new_feature():
    # Your code here
    pass

components["new_button"].click(
    fn=handle_new_feature,
    inputs=...,
    outputs=...
)
```

---

## 🐛 Debugging Checklist

### Application Won't Start

- [ ] Virtual environment activated?
- [ ] Dependencies installed? (`pip install -r requirements.txt`)
- [ ] All files in same directory?
- [ ] `.env` file present with API keys?

### Import Errors

```python
# Check imports at top of each file match:
from config import *
from models import current_case_data, ...
from utils import load_json, ...
```

### Generation Fails

- [ ] Check `generation.py` for errors
- [ ] Verify LLM initialization in `utils.py`
- [ ] Check vector database exists (`data/slp_vector_db/`)
- [ ] Verify API keys in `.env`

### UI Not Showing/Updating

- [ ] Check `visible=True/False` in UI files
- [ ] Verify `gr.update()` calls in event handlers
- [ ] Check component dictionary includes all elements

### Save Not Working

- [ ] Check `feedback.py` -> `handle_save()`
- [ ] Verify path in `config.py` -> `DEFAULT_OUTPUT_PATH`
- [ ] Check directory exists or can be created
- [ ] Verify state not empty (`current_case_data`)

---

## 📝 Code Templates

### Add New UI Component

```python
# In create_*_ui()
new_component = gr.Textbox(label="New Field", ...)

# Add to components dict
components["new_component"] = new_component

# Return it
return components
```

### Add New Event Handler

```python
# In setup_*_events()
def handle_new_event(input_value):
    # Process input
    result = process(input_value)
    return result

components["trigger_btn"].click(
    fn=handle_new_event,
    inputs=components["input_field"],
    outputs=components["output_field"]
)
```

### Add New Generation Function

```python
# In generation.py
def generate_new_type(param1, param2):
    """Generate new type of case."""
    
    generation_control["should_stop"] = False
    
    # Your generation logic
    
    if generation_control["should_stop"]:
        yield "Stopped", None
        return
    
    # More logic
    
    yield final_output, filename
```

### Add New Utility Function

```python
# In utils.py
def my_new_utility(param1, param2) -> ReturnType:
    """
    Description of what this does.
    
    Args:
        param1: Description
        param2: Description
    
    Returns:
        Description of return value
    """
    # Your logic here
    return result
```

---

## 🎨 Styling Quick Reference

### CSS Classes Available

```python
# In Gradio components
elem_classes="left-panel"    # Light gray panel
elem_classes="right-panel"   # Darker gray panel  
elem_classes="save-section"  # Blue save section
elem_classes="cover-buttons" # Centered cover buttons
elem_classes="title"         # Centered title
```

### Common Gradio Patterns

```python
# Button with variant
gr.Button("Text", variant="primary")  # Blue
gr.Button("Text", variant="secondary") # Gray
gr.Button("Text", variant="stop")     # Red

# Size options
gr.Button("Text", size="sm")   # Small
gr.Button("Text", size="lg")   # Large

# Visibility control
gr.Column(visible=False)  # Hidden initially
gr.update(visible=True)   # Show it later

# Interactive control
gr.Button("Text", interactive=False)  # Disabled
gr.update(interactive=True)           # Enable later
```

---

## 🔧 Configuration Quick Changes

### Model Temperature

```python
# In utils.py -> get_llm()
ChatOllama(model=model_id, temperature=0.7)  # Change 0.7 to desired
```

### Vector Database Settings

```python
# In generation.py
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
# Change k value for more/fewer context documents
```

### Output File Naming

```python
# In generation functions
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
suggested_filename = f"case_{timestamp}.md"
# Modify format as needed
```

---

## 📊 State Management

### Access Global State

```python
from models import current_case_data, multiple_cases_batch, group_session_data, generation_control

# Read
content = current_case_data["content"]

# Write
current_case_data["case_id"] = new_id

# Stop generation
generation_control["should_stop"] = True
```

---

## 🧪 Testing Tips

### Test Individual Module

```python
# Create test file: test_utils.py
import unittest
from utils import check_grade_compatibility

class TestUtils(unittest.TestCase):
    def test_grades(self):
        result, msg = check_grade_compatibility(["1st Grade", "2nd Grade"])
        self.assertTrue(result)

if __name__ == "__main__":
    unittest.main()
```

### Test UI Without Full App

```python
# Test just one page
from ui_single_case import create_single_case_ui, setup_single_case_events
import gradio as gr

with gr.Blocks() as test_app:
    components = create_single_case_ui(lambda: None)
    setup_single_case_events(components, gr.State())

test_app.launch()
```

---

## 💾 Backup Strategy

```bash
# Before major changes
cp -r . ../backup_$(date +%Y%m%d)

# Or use git
git init
git add .
git commit -m "Working version"
```

---

## 🚀 Performance Tips

1. **Use free models for development**: Llama3.2 for quick iteration
2. **Use premium models for production**: GPT-4o/Claude for quality
3. **Reduce search results**: Lower `k` value in retriever if slow
4. **Batch generations**: Multiple cases in one run more efficient

---

## 📞 Quick Help

| Problem | Solution File | Function to Check |
|---------|--------------|------------------|
| App crashes on start | `app.py` | `create_app()` |
| Generation hangs | `generation.py` | Specific generate function |
| UI doesn't update | `ui_*.py` | Event handler setup |
| Save fails | `feedback.py`, `utils.py` | `handle_save()` |
| Model not found | `utils.py` | `get_llm()` |
| Import error | All files | Top import statements |

---

## 🎯 Remember

- **One feature = One file** (mostly)
- **UI files** = What you see
- **Generation files** = What happens
- **Utils files** = How it works
- **Config files** = What's available

---

**Keep this file open while coding!** 📌
