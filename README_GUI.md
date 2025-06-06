# AI Testbed GUI

A graphical user interface for testing multiple AI models concurrently with automatic context preparation from codebases.

## Features

- **Context Preparation**: Automatically scan C# codebases and prepare context for AI prompts
- **Multi-Model Testing**: Test prompts against OpenAI, Claude, Gemini, and Ollama models simultaneously
- **Async Processing**: Each model runs in its own thread with real-time status updates
- **Tabbed Output**: Clean, organized display of responses from each model
- **Context Management**: Copy context to clipboard, attach to prompts automatically
- **Prompt Management**: Load/save prompts from files, edit in integrated text editor

## Screenshot/Layout

The GUI is organized into several sections:
- **Context Preparation**: Browse directories, prepare context from C# files, preview context
- **Model Selection**: Enable/disable models, configure model names and parameters
- **Prompt Section**: Edit prompts, load/save from files, toggle context attachment
- **Control Section**: Submit to models, clear outputs, stop all processes
- **Output Section**: Tabbed display showing each model's response with status indicators

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements_gui.txt
```

Required packages:
- `openai>=1.0.0` - OpenAI API client
- `anthropic>=0.25.0` - Anthropic Claude API client  
- `google-generativeai>=0.3.0` - Google Gemini API client
- `tiktoken>=0.5.0` - Token counting for OpenAI
- `requests>=2.28.0` - HTTP requests for Ollama
- `colorama>=0.4.0` - Enhanced console output (optional)

### 2. Setup API Keys

1. Copy the template file:
   ```bash
   cp apikeys_template.json apikeys.json
   ```

2. Edit `apikeys.json` with your actual API keys:
   ```json
   {
     "openai": {
       "api_key": "sk-your-actual-openai-key"
     },
     "anthropic": {
       "api_key": "sk-ant-your-actual-anthropic-key"
     },
     "google": {
       "api_key": "your-actual-gemini-key"
     }
   }
   ```

3. **Never commit `apikeys.json` to version control!**

### 3. Setup Ollama (Optional)

If you want to use local Ollama models:

1. Install Ollama from https://ollama.ai/
2. Pull desired models:
   ```bash
   ollama pull llama3.1
   ollama pull codellama
   ```

## Usage

### Starting the GUI

```bash
python ai_testbed_gui.py
```

### Basic Workflow

1. **Prepare Context**:
   - Click "Browse" to select your codebase directory
   - Click "Prepare Context" to scan C# files
   - Preview shows first 2000 characters of the prepared context
   - Context is automatically limited to first 20 files to avoid token limits

2. **Select Models**:
   - Check the models you want to test
   - Configure model names (e.g., `o3-mini`, `claude-3-5-sonnet-latest`)
   - For OpenAI, set reasoning effort (`auto`, `high`, `low`)

3. **Create Prompt**:
   - Type your prompt in the text area, or
   - Load from file using "Load from File" button
   - Toggle "Attach context to prompt" to include prepared context

4. **Submit and Review**:
   - Click "Submit to Selected Models"
   - Watch status indicators for each model
   - Review responses in separate tabs
   - Use "Clear All Outputs" to reset for next test

### Advanced Features

- **Context Clipboard**: Use "Copy Context to Clipboard" to manually paste context elsewhere
- **Prompt Management**: Save commonly used prompts to files for reuse
- **Stop All**: Emergency stop for all running model queries
- **Status Indicators**: Color-coded status (blue=processing, green=complete, red=error)

## Model-Specific Notes

### OpenAI
- Supports reasoning effort parameter for o3 models
- Uses tiktoken for accurate token counting
- Requires valid OpenAI API key

### Claude
- Uses Anthropic's latest API
- Supports "thinking" mode for complex reasoning
- Requires valid Anthropic API key

### Gemini
- Uses Google's Generative AI API
- Good for fast responses and multimodal tasks
- Requires valid Google API key

### Ollama
- Runs locally, no API key required
- Requires Ollama installation and pulled models
- Good for privacy-sensitive prompts

## Troubleshooting

### Common Issues

**"API keys file not found"**
- Create `apikeys.json` from the template
- Ensure it's in the same directory as the GUI script

**"No .cs files found"**  
- Make sure the selected directory contains C# files
- Check that files aren't in excluded directories (obj, bin, .git, .vs)

**Model timeouts or errors**
- Check internet connection for cloud APIs
- Verify API keys are valid and have sufficient credits
- For Ollama, ensure the service is running and model is pulled

**UI freezing**
- All model calls run in separate threads, but very large contexts may cause delays
- Use the "Stop All" button if needed

### Performance Tips

- Context is automatically limited to 20 files to prevent token limit issues
- Use smaller, focused directories for faster context preparation
- Test with shorter prompts first to verify API connectivity
- Ollama performance depends on your local hardware

## File Structure

```
SimpleAITestbed/
├── ai_testbed_gui.py          # Main GUI application
├── aitestbed.py               # Core AI model interfaces
├── prep_context.py            # Context preparation utilities
├── requirements_gui.txt       # Python dependencies
├── apikeys_template.json      # API keys template
├── apikeys.json              # Your actual API keys (create from template)
└── README_GUI.md             # This file
```

## Security Notes

- API keys are stored locally in `apikeys.json`
- Never commit API keys to version control
- Consider using environment variables for production deployments
- Ollama runs locally and doesn't send data to external services

## Contributing

To extend the GUI:
1. Model interfaces are in `aitestbed.py`
2. Context preparation logic is in `prep_context.py` 
3. GUI layout and threading are in `ai_testbed_gui.py`
4. Follow the existing pattern for adding new AI models

## License

This is part of the SimpleAITestbed project. See main project for license details.