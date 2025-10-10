# SLP SimuCase Generator - Refactored Structure

## 📁 New File Structure

```
project_root/
│
├── app.py                      # Main entry point - run this file
│
├── config.py                   # Configuration & constants
├── models.py                   # Pydantic models & global state
├── utils.py                    # Utility functions (file I/O, LLM init, parsing)
│
├── generation.py               # Case generation logic
├── feedback.py                 # Feedback collection & analysis
│
├── ui_single_case.py          # Single case UI components
├── ui_multiple_cases.py       # Multiple cases UI components
├── ui_group_session.py        # Group session UI components
│
├── .env                        # Environment variables (API keys)
├── requirements.txt            # Python dependencies
│
├── data/
│   └── slp_vector_db/         # Vector database
│
├── prompts/                    # Prompt template files
│   ├── grammar_check.txt
│   ├── feedback_category.txt
│   ├── single_case_free_model.txt
│   └── single_case_premium_model.txt
│
└── generated_case_files/       # Output directory
```

---

## 🚀 How to Run

**Start the application:**

```bash
python app.py
```

That's it! The app will launch at `http://localhost:7860`

---

## 🎯 Benefits of This Structure

### 1. **Separation of Concerns**
Each file has a clear, single responsibility:
- `config.py` → All configuration in one place
- `generation.py` → All generation logic
- `ui_single_case.py` → Only single case UI

### 2. **Easy Maintenance**
Want to modify the multiple cases feature? Just edit:
- `ui_multiple_cases.py` (UI changes)
- `generation.py` (logic changes if needed)

### 3. **No More Giant File**
- Original: 1 file with 1,800+ lines ❌
- New: 9 files with ~200-500 lines each ✅

### 4. **Reusability**
Functions are now importable and reusable across modules.

---

## 📝 How to Modify Each Function

### **Modifying Single Case Generation**

**UI Changes (buttons, layout, inputs):**
```python
# Edit: ui_single_case.py
# Function: create_single_case_ui()
```

**Logic Changes (how cases are generated):**
```python
# Edit: generation.py
# Function: generate_single_case()
```

**Event Handlers (what happens when buttons clicked):**
```python
# Edit: ui_single_case.py
# Function: setup_single_case_events()
```

---

### **Modifying Multiple Cases Generation**

**UI Changes:**
```python
# Edit: ui_multiple_cases.py
# Function: create_multiple_cases_ui()
```

**Generation Logic:**
```python
# Edit: generation.py
# Function: generate_multiple_cases()
```

**Natural Language Parsing:**
```python
# Edit: generation.py
# Function: parse_complex_request()
```

---

### **Modifying Group Session**

**UI Changes:**
```python
# Edit: ui_group_session.py
# Function: create_group_session_ui()
```

**Generation Logic:**
```python
# Edit: generation.py
# Function: generate_group_session()
```

**Compatibility Checking:**
```python
# Edit: utils.py
# Functions: check_grade_compatibility(), check_disorder_compatibility()
```

---

### **Adding a New Model**

```python
# Edit: config.py
# Add to FREE_MODELS or PREMIUM_MODELS
FREE_MODELS.append("New Model Name")

# Add to MODEL_MAP
MODEL_MAP["New Model Name"] = "model-id-string"

# Edit: utils.py
# Add handling in get_llm() function
```

---

### **Changing Feedback System**

```python
# Edit: feedback.py
# All feedback logic is here
```

---

### **Adding New Disorder Types**

```python
# Edit: config.py
DISORDER_TYPES.append("New Disorder Type")
```

---

## 🔧 Common Modifications

### **Change Default Output Path**
```python
# Edit: config.py
DEFAULT_OUTPUT_PATH = "your/new/path/"
```

### **Add New Grade Level**
```python
# Edit: config.py
ALL_GRADES.append("13th Grade")
```

### **Modify Prompt Templates**
```python
# Edit files in: prompts/
# No code changes needed!
```

### **Change AI Temperature**
```python
# Edit: utils.py
# Function: get_llm()
# Change: temperature=0.7 to desired value
```

---

## 🐛 Debugging Tips

### **Import Errors?**
Make sure all files are in the same directory, or adjust Python path:
```python
import sys
sys.path.append('/path/to/your/modules')
```

### **Function Not Found?**
Check the import statements at the top of each file. Example:
```python
# In app.py
from ui_single_case import create_single_case_ui, setup_single_case_events
```

### **State Issues?**
All global state is in `models.py`:
- `current_case_data`
- `multiple_cases_batch`
- `group_session_data`
- `generation_control`

---

## 📊 Module Dependencies

```
app.py
  ├── ui_single_case.py
  │     ├── config.py
  │     ├── models.py
  │     ├── generation.py
  │     ├── feedback.py
  │     └── utils.py
  │
  ├── ui_multiple_cases.py
  │     ├── config.py
  │     ├── models.py
  │     ├── generation.py
  │     ├── feedback.py
  │     └── utils.py
  │
  └── ui_group_session.py
        ├── config.py
        ├── models.py
        ├── generation.py
        ├── feedback.py
        └── utils.py
```

---

## 🎨 Example: Adding a New Feature

**Want to add a "Bulk Export" feature to Multiple Cases?**

1. **Add UI button:**
```python
# Edit: ui_multiple_cases.py
# In create_multiple_cases_ui(), add:
bulk_export_btn = gr.Button("📦 Bulk Export to CSV")
```

2. **Create handler function:**
```python
# In ui_multiple_cases.py
def handle_bulk_export():
    # Your export logic here
    pass
```

3. **Connect event:**
```python
# In setup_multiple_cases_events()
components["bulk_export_btn"].click(
    fn=handle_bulk_export,
    outputs=...
)
```

No need to touch other files! ✨

---

## ✅ Migration Checklist

- [ ] Create all 9 new files
- [ ] Copy your `.env` file
- [ ] Copy your `prompts/` directory
- [ ] Copy your `data/` directory
- [ ] Test each function:
  - [ ] Single case generation
  - [ ] Multiple cases generation
  - [ ] Group session generation
  - [ ] Feedback submission
  - [ ] File saving
- [ ] Delete old monolithic file (backup first!)

---

## 💡 Pro Tips

1. **Work on one feature at a time** - Only open the relevant files
2. **Use version control** - Git makes it easy to track changes per module
3. **Test after each change** - Smaller files = easier debugging
4. **Share modules** - Other projects can now reuse your utilities!

---

## 🆘 Need Help?

**Common Issues:**

| Issue | File to Check | Solution |
|-------|---------------|----------|
| UI not showing | `ui_*.py` | Check `visible=True/False` |
| Generation fails | `generation.py` | Check LLM initialization |
| Save not working | `feedback.py`, `utils.py` | Check file paths |
| Import errors | All files | Ensure all in same directory |

---

**Happy Coding! 🎉**

With this modular structure, you can now:
- ✅ Modify one feature without touching others
- ✅ Add new features easily
- ✅ Collaborate with others (each person can work on different files)
- ✅ Reuse code in other projects
- ✅ Debug faster with smaller, focused files
