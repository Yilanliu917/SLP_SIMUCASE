cd C:\Users\yilan\Documents\SLP_SIMUCASE

# Create README.md with Hugging Face configuration
@"
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

# SLP SimuCase Generator

Professional AI-powered Speech-Language Pathology case file generator with modular architecture.

## 🌟 Features

- **Single Case Generation**: Create individual student case files
- **Multiple Cases Generation**: Batch generate cases with natural language parsing
- **Group Session Planning**: Generate therapy group sessions following clinical strategies
- **Multi-Model Support**: OpenAI GPT-4o, Google Gemini, Anthropic Claude, and local Ollama models
- **RAG-Powered**: Context-aware generation using vector database
- **Feedback System**: Built-in evaluation and feedback collection

## 🚀 Quick Start

This Space demonstrates the modular architecture and UI. For full functionality with knowledge base:

1. Clone this repository
2. Set up your API keys in Hugging Face Secrets
3. Add your SLP knowledge base to the vector database

## ⚙️ Configuration

### Required Secrets (Add in Space Settings)

- ``OPENAI_API_KEY``: For GPT-4o and embeddings
- ``ANTHROPIC_API_KEY``: For Claude models (optional)
- ``GOOGLE_API_KEY``: For Gemini models (optional)

### Add Secrets:
1. Go to Space Settings → Repository secrets
2. Add your API keys

## 📁 Architecture

Built with modular architecture for easy maintenance:
- ``main.py``: Entry point
- ``app/``: Core application modules
  - ``config.py``: Configuration
  - ``models.py``: Data models
  - ``generation.py``: Generation logic
  - ``ui_*.py``: UI components

## 🔒 Note

This is a demonstration of the application architecture. The full knowledge base and vector database are not included due to size limitations.

## 📚 Documentation

For full documentation, visit the [GitHub repository](https://github.com/Yilanliu917/SLP_SIMUCASE).

## 📄 License

MIT License - See LICENSE file for details
"@ | Out-File -FilePath README.md -Encoding utf8

# Add and commit
git add README.md
git commit -m "Add Hugging Face Space configuration"

# Push to HuggingFace
git push hf main --force

Write-Host "`n✅ README created and pushed!" -ForegroundColor Green
Write-Host "Your Space should start building now at:" -ForegroundColor Cyan
Write-Host "https://huggingface.co/spaces/Yilanliu917/SLP-SimuCase-Generator2" -ForegroundColor Yellow