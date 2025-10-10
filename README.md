---
title: SLP SimuCase Generator
emoji: 🗣️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.44.0"
app_file: main.py
pinned: false
license: mit
---

# SLP SimuCase Generator 🗣️

AI-powered Speech-Language Pathology case file generator with modular architecture.

## ✨ Features

- 🎯 **Single Case Generation** - Create individual student case files
- 📦 **Batch Generation** - Generate multiple cases with natural language
- 👥 **Group Sessions** - Plan therapy groups following clinical strategies  
- 🤖 **Multi-Model** - GPT-4o, Gemini, Claude, or local Ollama
- 📚 **RAG-Powered** - Context-aware with vector database
- 💬 **Feedback System** - Built-in evaluation

## 🚀 Usage

1. Select generation mode (Single/Multiple/Group)
2. Choose grade level and disorders
3. Select AI model
4. Generate!

## ⚙️ Setup for Full Functionality

This Space requires API keys. Add them in **Settings → Repository secrets**:

- `OPENAI_API_KEY` - Required for embeddings and GPT-4o
- `ANTHROPIC_API_KEY` - Optional for Claude models
- `GOOGLE_API_KEY` - Optional for Gemini models

## 📝 Note

This demo shows the UI and architecture. Full knowledge base not included due to size limits.

## 🔗 Links

- [GitHub Repository](https://github.com/Yilanliu917/SLP_SIMUCASE)
- [Documentation](https://github.com/Yilanliu917/SLP_SIMUCASE#readme)

## 📄 License

MIT License