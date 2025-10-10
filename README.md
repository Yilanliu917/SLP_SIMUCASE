# SLP SimuCase Generator (Modular Architecture)

Professional AI-powered Speech-Language Pathology case file generator with modular architecture.

## 🌟 Features

- Modular architecture for easy maintenance
- Support for multiple LLM providers (OpenAI, Google, Anthropic, Ollama)
- Single case, multiple cases, and group session generation
- RAG-powered context-aware generation
- Built-in feedback system

## 🚀 Quick Start

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create `.env` file with your API keys
4. Set up vector database in `data/slp_vector_db/`
5. Run: `python main.py`

## 📁 Project Structure
SLP_SIMUCASE/
├── main.py              # Entry point
├── app/                 # Application modules
│   ├── config.py
│   ├── models.py
│   ├── utils.py
│   ├── generation.py
│   ├── feedback.py
│   └── ui_*.py
└── prompts/             # Prompt templates
## 🔧 Configuration

Create a `.env` file:

OPENAI_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here

## 📝 License
