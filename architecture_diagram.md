# System Architecture

## 🏗️ High-Level Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         app.py                               │
│                    (Main Entry Point)                        │
│                                                              │
│  • Launches Gradio application                              │
│  • Creates cover page with navigation                       │
│  • Orchestrates all UI modules                              │
│  • Sets up page switching                                   │
└───────────────────────┬─────────────────────────────────────┘
                        │
         ┌──────────────┼──────────────┐
         │              │              │
         ▼              ▼              ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ui_single    │  │ui_multiple  │  │ui_group     │
│_case.py     │  │_cases.py    │  │_session.py  │
│             │  │             │  │             │
│• UI Layout  │  │• UI Layout  │  │• UI Layout  │
│• Events     │  │• Events     │  │• Events     │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │
       └────────────────┼────────────────┘
                        │
       ┌────────────────┴────────────────┐
       │                                  │
       ▼                                  ▼
┌─────────────┐                    ┌─────────────┐
│generation.py│                    │feedback.py  │
│             │                    │             │
│• Single     │                    │• Submit     │
│• Multiple   │                    │• Save       │
│• Group      │                    │• Analyze    │
│• Parsing    │                    └──────┬──────┘
└──────┬──────┘                           │
       │                                  │
       └──────────────┬───────────────────┘
                      │
                      ▼
              ┌───────────────┐
              │   utils.py    │
              │               │
              │• File I/O     │
              │• LLM Init     │
              │• Parsers      │
              │• Validators   │
              │• AI Helpers   │
              └───────┬───────┘
                      │
       ┌──────────────┼──────────────┐
       │              │              │
       ▼              ▼              ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ config.py   │  │ models.py   │  │External APIs│
│             │  │             │  │             │
│• Constants  │  │• Pydantic   │  │• OpenAI     │
│• Settings   │  │• State      │  │• Google     │
│• Options    │  │• Data       │  │• Anthropic  │
└─────────────┘  └─────────────┘  │• Ollama     │
                                   └─────────────┘
```

---

## 🔄 Data Flow

### Single Case Generation Flow

```
User Interaction (ui_single_case.py)
    │
    ├─> Select Parameters (grade, disorders, model)
    │
    ├─> Click "Generate" Button
    │
    ▼
Event Handler (ui_single_case.py)
    │
    ├─> Validates inputs
    │
    ├─> Calls generation function
    │
    ▼
Generation Logic (generation.py)
    │
    ├─> Initialize vector database (utils.py)
    │
    ├─> Load prompt template (utils.py)
    │
    ├─> Get LLM instance (utils.py)
    │
    ├─> Build RAG chain
    │
    ├─> Generate content
    │
    ▼
State Update (models.py)
    │
    ├─> Store in current_case_data
    │
    ▼
UI Update (ui_single_case.py)
    │
    ├─> Display generated case
    │
    ├─> Show save section
    │
    ▼
Save Operation (feedback.py)
    │
    ├─> Write to file (utils.py)
    │
    ├─> Update database (utils.py)
    │
    ▼
Complete
```

---

## 📦 Module Dependencies

### Import Hierarchy

```
app.py
  │
  ├── ui_single_case.py
  │     ├── config.py
  │     ├── models.py
  │     ├── generation.py
  │     │     ├── config.py
  │     │     ├── models.py
  │     │     └── utils.py
  │     │           ├── config.py
  │     │           └── models.py
  │     ├── feedback.py
  │     │     └── utils.py
  │     └── utils.py
  │
  ├── ui_multiple_cases.py
  │     └── [same dependencies as single case]
  │
  └── ui_group_session.py
        └── [same dependencies as single case]
```

### Dependency Rules

1. **config.py** and **models.py** have NO dependencies (base layer)
2. **utils.py** depends only on config.py and models.py
3. **generation.py** and **feedback.py** depend on utils, config, models
4. **UI modules** depend on everything
5. **app.py** depends only on UI modules

---

## 🎯 Responsibility Matrix

| Module | Create UI | Handle Events | Generate Content | Manage State | Config | I/O |
|--------|-----------|---------------|------------------|--------------|--------|-----|
| app.py | ✓ Cover | ✓ Navigation | ✗ | ✗ | ✗ | ✗ |
| ui_single_case.py | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |
| ui_multiple_cases.py | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |
| ui_group_session.py | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |
| generation.py | ✗ | ✗ | ✓ | ✓ Write | ✗ | ✗ |
| feedback.py | ✗ | ✗ | ✗ | ✓ Write | ✗ | ✓ |
| utils.py | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| models.py | ✗ | ✗ | ✗ | ✓ Define | ✗ | ✗ |
| config.py | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ |

---

## 🔐 State Management Architecture

```
┌────────────────────────────────────────────────────────┐
│                    models.py                           │
│               (Central State Store)                    │
│                                                        │
│  current_case_data = {                                │
│    "content": str,                                    │
│    "case_id": str,                                    │
│    "metadata": dict                                   │
│  }                                                    │
│                                                        │
│  multiple_cases_batch = {                             │
│    "cases": list,                                     │
│    "batch_id": str,                                   │
│    "timestamp": str                                   │
│  }                                                    │
│                                                        │
│  group_session_data = {                               │
│    "session_id": str,                                 │
│    "members": list,                                   │
│    "timestamp": str                                   │
│  }                                                    │
│                                                        │
│  generation_control = {                               │
│    "should_stop": bool                                │
│  }                                                    │
└────────────────────────────────────────────────────────┘
         ▲                    ▲                    ▲
         │                    │                    │
         │                    │                    │
    ┌────┴────┐          ┌───┴────┐          ┌───┴────┐
    │ Single  │          │Multiple│          │ Group  │
    │  Case   │          │ Cases  │          │Session │
    │   UI    │          │   UI   │          │   UI   │
    └─────────┘          └────────┘          └────────┘
```

---

## 🔌 External Integration Points

```
┌─────────────────────────────────────────────────────────┐
│                  SLP SimuCase App                        │
└─────────────────────────────────────────────────────────┘
              │                           │
              │                           │
    ┌─────────▼─────────┐       ┌────────▼─────────┐
    │  Vector Database  │       │  LLM Providers   │
    │  (ChromaDB)       │       │                  │
    │                   │       │  • OpenAI API    │
    │  • Embeddings     │       │  • Google API    │
    │  • Knowledge Base │       │  • Anthropic API │
    │  • Search         │       │  • Ollama Local  │
    └───────────────────┘       └──────────────────┘
              │
              │
    ┌─────────▼─────────┐
    │  File System      │
    │                   │
    │  • .env          │
    │  • prompts/      │
    │  • data/         │
    │  • generated/    │
    │  • logs/         │
    └───────────────────┘
```

---

## 🚦 Request Flow Example

### User wants to generate 5 cases

```
1. User: Opens "Multiple Cases" page
   ↓ [ui_multiple_cases.py]

2. UI: Displays chat box and manual config
   ↓ [User types natural language request]

3. User: "Generate 5 cases with articulation, use GPT-4o"
   ↓ [Clicks "Parse Request"]

4. Event Handler: Calls parse_complex_request()
   ↓ [generation.py]

5. Parser: Uses Llama3.2 to parse request
   ↓ [Returns structured task list]

6. UI: Auto-fills configuration rows
   ↓ [User clicks "Generate All Cases"]

7. Event Handler: Collects all parameters
   ↓ [Calls generate_multiple_cases()]

8. Generator: For each case:
   ├─> Initialize embeddings [utils.py]
   ├─> Load prompts [utils.py]
   ├─> Get LLM instance [utils.py]
   ├─> Build RAG chain
   ├─> Generate content
   └─> Yield progress update
   ↓

9. State: Updates multiple_cases_batch
   ↓ [models.py]

10. UI: Streams output to display
    ↓ [Real-time updates]

11. Complete: Shows save button
    ↓ [User saves]

12. Feedback: Writes to file system
    └─> [feedback.py → utils.py]
```

---

## 🎨 UI Component Architecture

```
┌────────────────────────────────────────────┐
│            Gradio Blocks App               │
├────────────────────────────────────────────┤
│                                            │
│  ┌──────────────────────────────────────┐ │
│  │         Cover Page (app.py)          │ │
│  │                                      │ │
│  │  [Generate Single Case]    ────────┐│ │
│  │  [Generate Multiple Cases] ────────┤│ │
│  │  [Generate Group Session]  ────────┤│ │
│  └──────────────────────────────────────┘ │
│                ││ navigation              │
│                ││                         │
│  ┌──────────────▼▼────────────────────┐  │
│  │                                     │  │
│  │  Single Case Page                  │  │
│  │  (ui_single_case.py)              │  │
│  │                                     │  │
│  │  ┌───────────────────────────────┐ │  │
│  │  │ Left Panel: Parameters        │ │  │
│  │  │ • Grade, Model, Disorders     │ │  │
│  │  │ • Generate Button             │ │  │
│  │  └───────────────────────────────┘ │  │
│  │                                     │  │
│  │  ┌───────────────────────────────┐ │  │
│  │  │ Right Panel: Options          │ │  │
│  │  │ • Reference Upload            │ │  │
│  │  └───────────────────────────────┘ │  │
│  │                                     │  │
│  │  ┌───────────────────────────────┐ │  │
│  │  │ Output Display                │ │  │
│  │  └───────────────────────────────┘ │  │
│  │                                     │  │
│  │  ┌───────────────────────────────┐ │  │
│  │  │ Save Section                  │ │  │
│  │  └───────────────────────────────┘ │  │
│  │                                     │  │
│  │  ┌───────────────────────────────┐ │  │
│  │  │ Feedback Accordion            │ │  │
│  │  └───────────────────────────────┘ │  │
│  └─────────────────────────────────────┘ │
│                                            │
│  [Similar structure for Multiple Cases    │
│   and Group Session pages...]             │
│                                            │
└────────────────────────────────────────────┘
```

---

## 🔧 Configuration Flow

```
Environment Variables (.env)
        │
        ▼
┌───────────────────┐
│   config.py       │
│                   │
│ • Loads from env  │
│ • Sets constants  │
│ • Defines options │
└────────┬──────────┘
         │
         ├─────────────────────┐
         │                     │
         ▼                     ▼
   ┌─────────┐          ┌──────────┐
   │utils.py │          │All UI    │
   │         │          │modules   │
   │• Uses   │          │          │
   │  MODEL  │          │• Display │
   │  _MAP   │          │  options │
   │• Uses   │          │  from    │
   │  paths  │          │  config  │
   └─────────┘          └──────────┘
```

---

## 💡 Design Patterns Used

### 1. **Module Pattern**
Each UI module exports:
- `create_*_ui()` - Factory function
- `setup_*_events()` - Event binder

### 2. **Dependency Injection**
```python
# app.py injects dependencies
components = create_single_case_ui(back_handler)
setup_single_case_events(components, state)
```

### 3. **Observer Pattern**
```python
# UI observes state changes
generation_control["should_stop"] = True  # Signal
# Generator checks flag and stops
```

### 4. **Factory Pattern**
```python
# utils.py creates appropriate LLM
llm = get_llm(model_name)  # Returns correct instance
```

### 5. **Component Pattern**
```python
# UI components returned as dictionary
components = {
    "button": button,
    "output": output
}
```

---

## 📊 Performance Considerations

```
Bottlenecks:
1. LLM API Calls      → Use streaming for feedback
2. Vector DB Search   → Cache frequently used queries
3. File I/O          → Async operations where possible
4. UI Updates        → Use gr.update() efficiently

Optimizations Applied:
✓ Streaming output for long generations
✓ Separate threads for generation (Gradio handles)
✓ Lazy loading of large models
✓ Efficient state management
```

---

## 🔒 Security Architecture

```
┌─────────────────────────────────────┐
│     User Environment (.env)         │
│  • API keys stored locally          │
│  • Not committed to version control │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│      Application Layer              │
│  • Loads keys at runtime            │
│  • Never logs sensitive data        │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│     External APIs                   │
│  • HTTPS connections only           │
│  • Provider's security measures     │
└─────────────────────────────────────┘
```

---

## 🎯 Key Architectural Decisions

1. **Modular by Feature**: Each feature gets its own UI module
2. **Centralized State**: All state in models.py
3. **Utility Layer**: Shared functions in utils.py
4. **Configuration Separate**: Easy to modify without code changes
5. **UI/Logic Separation**: UI modules don't contain business logic
6. **Streaming Generators**: For responsive UI during long operations

---

## 📈 Scalability Path

```
Current Architecture
    │
    ├─> Add New Page? → New ui_*.py module
    │
    ├─> Add New Model? → Update config.py + utils.py
    │
    ├─> Add New Feature? → Update relevant module only
    │
    └─> Add Database? → New module (e.g., database.py)
```

---

This architecture enables:
- ✅ Easy maintenance
- ✅ Independent feature development
- ✅ Clear separation of concerns
- ✅ Scalable growth
- ✅ Testable components
