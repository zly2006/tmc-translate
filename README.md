# TMC Translate - RAG Translation System

A Chinese-English terminology translation system based on LangChain, using RAG technology to ensure accurate translation of professional terms.

## Features

- 🌐 **Bidirectional Translation**: Supports mutual translation between Chinese and English
- 📚 **Terminology Management**: Load terminology database from Excel files to ensure accurate professional term translation
- 🤖 **Multi-Model Support**: Supports both Ollama local models and Google Gemini cloud models
- 🔍 **RAG Technology**: Uses vector database for semantic search to provide relevant terminology context
- 💬 **Interactive Interface**: Provides user-friendly command-line interactive interface
- 📊 **Excel Support**: Supports reading terminology database from Excel files

## Project Structure

```
src/
├── tmc_translate/
│   ├── __init__.py          # Package initialization
│   ├── main.py              # Main program and interactive interface
│   ├── models.py            # Data model definitions
│   ├── rag_translator.py    # RAG translator core
│   └── terminology_manager.py # Terminology manager
```

## Installation

The project uses uv for dependency management:

```bash
# Clone the project
git clone <repository_url>
cd tmc-translate

# Install dependencies using uv
uv sync
```

## Terminology Database Format

The terminology database should be an Excel file (.xlsx or .xls) containing the following columns:

| Column Name | Description | Example |
|-------------|-------------|---------|
| english_name | English term name | Machine Learning |
| chinese_name | Chinese term name | 机器学习 |
| english_description | English description | A method of data analysis that automates analytical model building |
| chinese_description | Chinese description | 一种自动化分析模型构建的数据分析方法 |

## Usage

### 1. Direct Execution

```bash
# Run using uv
uv run tmc-translate

# Or activate virtual environment and run
uv shell
python -m tmc_translate.main
```

### 2. Environment Configuration

Create a `.env` file (optional):

```env
# Google API Key for Gemini (optional, can also be entered at runtime)
GOOGLE_API_KEY=your_google_api_key_here

# Ollama configuration (optional)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2
```

### 3. Prepare Terminology Database

- The program will create a sample terminology file `sample_terminology.xlsx` on first run
- You can also prepare your own Excel terminology file, ensuring it contains the required column names

### 4. Model Selection

The program supports two types of models:

#### Ollama (Local)
- Requires Ollama service to be installed and running
- Recommended models: llama2, qwen, baichuan and other Chinese-supporting models
- Pros: Local execution, data security, no API key required
- Cons: Requires local computing resources

#### Google Gemini (Cloud)
- Requires API key from Google AI Studio
- Recommended models: gemini-pro, gemini-1.5-pro
- Pros: Powerful performance, fast response
- Cons: Requires network connection and API key

## Main Functions

### 1. Text Translation
- Automatic source language detection (Chinese/English)
- Intelligent matching of relevant terminology
- Provides terminology reference
- Generates high-quality translation results

### 2. Terminology Management
- View all terminology
- Reload terminology database
- Supports real-time updates

### 3. Model Switching
- Switch between different models at runtime
- Supports parameter adjustments

## Technical Architecture

- **LangChain**: Core framework providing LLM integration and chain operations
- **ChromaDB**: Vector database for terminology semantic search
- **Pandas**: Data processing and Excel file reading
- **dataclass**: Type-safe data models
- **dotenv**: Environment variable management

## Development

### Adding New Model Providers

Inherit from the `ModelProvider` abstract base class:

```python
from .rag_translator import ModelProvider

class YourModelProvider(ModelProvider):
    def get_llm(self):
        # Return your LLM instance
        pass
    
    def get_embeddings(self):
        # Return your Embeddings instance
        pass
```

### Extending Terminology Database Format

Modify the `load_from_excel` method in `terminology_manager.py` to support new column formats.

## FAQ

### Q: What to do if Ollama connection fails?
A: Make sure Ollama service is running:
```bash
ollama serve
ollama pull llama2  # Download model
```

### Q: Gemini API call failed?
A: Check:
- Whether the API Key is correct
- Whether the network connection is normal
- Whether there is sufficient API quota

### Q: Terminology database loading failed?
A: Check the Excel file:
- Whether the file format is correct (.xlsx or .xls)
- Whether it contains required column names
- Whether there are null values in the data

## License

[CC-BY-SA-NC 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) - for [terminology.xlsx](terminology.xlsx).
[AGPL-3.0](LICENSE_AGPL) - for the rest of the code.

## Contributing

Issues and Pull Requests are welcome!
