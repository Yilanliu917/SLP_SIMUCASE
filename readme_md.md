# SLP SimuCase Generator

A professional AI-powered system for generating Speech-Language Pathology (SLP) case files using RAG (Retrieval Augmented Generation) and multiple LLM providers.

## 🌟 Features

- **Single Case Generation**: Create individual student case files with customizable parameters
- **Multiple Cases Generation**: Batch generate cases with natural language parsing
- **Group Session Planning**: Generate therapy group sessions following clinical grouping strategies
- **Multi-Model Support**: Choose from free (Llama, Qwen, DeepSeek) and premium (GPT-4o, Gemini, Claude) models
- **RAG Integration**: Uses vector database for context-aware generation
- **Feedback System**: Built-in evaluation and feedback collection with AI-powered categorization

## 📋 Prerequisites

- Python 3.9+
- API Keys for premium models (optional):
  - OpenAI API key (for GPT-4o and embeddings)
  - Google API key (for Gemini)
  - Anthropic API key (for Claude)
- Ollama installed (for free models)

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project
cd slp-simucase-generator

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file in the root directory:

```env
# Required for embeddings and premium models
OPENAI_API_KEY=your_openai_key_here

# Optional: Only if using premium models
GOOGLE_API_KEY=your_google_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# Ollama is used for free models - install from https://ollama.ai
```

### 3. Setup Vector Database

Ensure you have your SLP knowledge base in the vector database:

```
data/
  └── slp_vector_db/
      └── [your chroma database files]
```

### 4. Run the Application

```bash
python app.py
```

The application will start at `http://localhost:7860`

## 📁 Project Structure

```
slp-simucase-generator/
│
├── app.py                      # Main application entry point
│
├── config.py                   # Configuration and constants
├── models.py                   # Pydantic data models
├── utils.py                    # Utility functions
│
├── generation.py               # Case generation logic
├── feedback.py                 # Feedback system
│
├── ui_single_case.py          # Single case UI
├── ui_multiple_cases.py       # Multiple cases UI
├── ui_group_session.py        # Group session UI
│
├── prompts/                    # Prompt templates
│   ├── grammar_check.txt
│   ├── feedback_category.txt
│   ├── single_case_free_model.txt
│   └── single_case_premium_model.txt
│
├── data/
│   └── slp_vector_db/         # Vector database
│
├── generated_case_files/       # Output directory
│
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (create this)
└── README.md                   # This file
```

## 🎯 Usage

### Generate Single Case

1. Click "Generate Single Case" on the cover page
2. Select grade level, disorders, and AI model
3. (Optional) Add population characteristics
4. Click "Generate"
5. Save the generated case file

### Generate Multiple Cases

1. Click "Generate Multiple Cases"
2. **Option A**: Use natural language
   - Describe what you want in the chat box
   - Click "Parse Request"
   - Review and adjust
3. **Option B**: Manual configuration
   - Configure each row with grade, disorders, count, model
   - Add/remove rows as needed
4. Click "Generate All Cases"
5. Save the batch

### Generate Group Session

1. Click "Generate Group Session"
2. Select group size (2-4 students)
3. Configure each member's grade and disorders
4. Click "Check Compatibility" to verify grouping
5. Click "Generate Group Session"
6. Save the session plan

## 🔧 Configuration Options

### Available Models

**Free Models** (requires Ollama):
- Llama3.2
- Qwen 2.5 7B
- Qwen 2.5 32B
- DeepSeek R1 32B

**Premium Models** (requires API keys):
- GPT-4o (OpenAI)
- Gemini 2.5 Pro (Google)
- Claude 3 Opus (Anthropic)
- Claude 3.5 Sonnet (Anthropic)

### Supported Disorders

- Speech Sound Disorder
- Articulation Disorders
- Phonological Disorders
- Language Disorders
- Receptive Language Disorders
- Expressive Language Disorders
- Pragmatics
- Fluency
- Childhood Apraxia of Speech

### Grade Levels

Pre-K through 12th Grade

## 📝 Customization

### Adding a New Model

Edit `config.py`:

```python
FREE_MODELS.append("New Model Name")
MODEL_MAP["New Model Name"] = "model-id"
```

Edit `utils.py` in `get_llm()` function to handle the new model.

### Modifying Prompts

Edit files in the `prompts/` directory. No code changes needed!

### Adding New Disorders

Edit `config.py`:

```python
DISORDER_TYPES.append("New Disorder Type")
```

## 🐛 Troubleshooting

### "Module not found" errors
- Ensure all files are in the same directory
- Verify virtual environment is activated
- Run `pip install -r requirements.txt`

### "API key not found" errors
- Check `.env` file exists and contains valid keys
- Ensure `python-dotenv` is installed
- Restart the application after adding keys

### Vector database errors
- Verify `data/slp_vector_db/` directory exists
- Check database was created with same embedding model
- Ensure OpenAI API key is set (used for embeddings)

### Ollama connection errors
- Install Ollama from https://ollama.ai
- Pull required models: `ollama pull llama3.2`
- Ensure Ollama service is running

## 📊 Feedback System

The application includes a comprehensive feedback system:

1. Rate cases on 5 metrics (1-5 scale)
2. Categorize feedback automatically with AI
3. Provide detailed written feedback
4. Track improvements over time

Feedback is saved to `feedback_log.json` and can be analyzed for model performance.

## 🔒 Data Privacy

- All generated cases are stored locally
- API calls to LLM providers follow their respective privacy policies
- No case data is stored on external servers except during API calls
- Vector database is stored locally

## 🤝 Contributing

Contributions are welcome! The modular structure makes it easy:

1. Choose the module you want to improve
2. Make your changes
3. Test thoroughly
4. Submit a pull request

## 📄 License

[Add your license here]

## 👥 Authors

[Add your name/team here]

## 🙏 Acknowledgments

- LangChain for the RAG framework
- Gradio for the UI framework
- OpenAI, Google, Anthropic, and Ollama for LLM access

## 📞 Support

For issues and questions:
- Check the REFACTORING_GUIDE.md for common solutions
- Open an issue on GitHub
- Contact [your contact info]

---

**Version**: 2.0 (Modular Architecture)  
**Last Updated**: 2025

## ⚡ Performance Tips

1. **Use free models for testing**: Iterate quickly with Llama/Qwen
2. **Use premium models for production**: Better quality with GPT-4o/Claude
3. **Batch processing**: Generate multiple cases in one run for efficiency
4. **Vector database optimization**: Keep your knowledge base updated and well-organized

## 🎓 Best Practices

1. **Single Cases**: Use for quick iterations and testing
2. **Multiple Cases**: Ideal for creating training datasets
3. **Group Sessions**: Perfect for planning real therapy sessions
4. **Feedback**: Always provide feedback to track model performance
5. **Prompts**: Customize prompts in `prompts/` directory for your specific needs

## 🔮 Future Roadmap

- [ ] Export to PDF format
- [ ] Integration with IEP software
- [ ] Real-time collaboration features
- [ ] Advanced analytics dashboard
- [ ] Mobile application
- [ ] Custom prompt templates UI
- [ ] Model performance comparison tools
