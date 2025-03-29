# AI Testbed

A flexible toolkit for testing and comparing responses from multiple AI models (OpenAI, Claude, Gemini, and Ollama).

## Screenshot
![image](https://github.com/user-attachments/assets/bb30ee12-8925-4637-a684-7e0bd295b9fc)


## Overview

AI Testbed provides two primary workflows:

1. **Sequential Testing** - Run a prompt through multiple AI models one after another with `aitestbed.py`
2. **Concurrent Testing** - Test multiple AI models in parallel using separate windows with `concurrent_ai_test.py`

This toolkit is designed for researchers, developers, and enthusiasts who want to:
- Compare model outputs side-by-side
- Test the same prompt across different models
- Experiment with various model parameters
- Benchmark performance and response quality

## Setup Requirements

### Dependencies

Install required packages:
```
pip install openai anthropic google-generativeai halo requests
```

### API Keys

Create an `apikeys.json` file in the same directory as the scripts with the following structure:

```json
{
  "openai": "sk-your-openai-api-key",
  "anthropic": "sk-ant-your-anthropic-api-key",
  "gemini": "your-gemini-api-key"
}
```

Notes:
- For Ollama, no API key is required (it runs locally)
- Place the `apikeys.json` file in the same directory as the scripts
- API keys for services you don't plan to use can be omitted

### Prompt File

Create a `prompt.txt` file with your question or instructions. For example:

```
Explain quantum computing in simple terms that a 10-year-old could understand.
```

This can be one line or hundreds of lines long. It all depends on the context length of the models you've using. I've successfully provided multiple files from a codebase or entire chapters of writings.

It's usually helpful to denote files in a larger query like so:

```
# Task: Refactor the webpage mypage.html below to introduce charts from chartjs. 
# Criteria 1: Only use ChartJS
# Criteria 2: Don't add new text content but feel free to re-arrange items.
# Criteria 3: Don't use the color BLUE, I hate it.

# mypage.html
<lots of content

# another_reference_page.html
<content>

# mycustom.js
<content> 

```

You can also create a `system_prompt.txt` file for Ollama models.

## Usage

### Method 1: Sequential Testing with aitestbed.py

This mode runs the same prompt through multiple models sequentially in the same terminal window.

```bash
python aitestbed.py
```

This will:
1. Load prompt from `prompt.txt` (or prompt for input if not found)
2. Run the prompt through OpenAI, Claude, Gemini, and Ollama models
3. Display responses one after another

### Method 2: Concurrent Testing with concurrent_ai_test.py

This mode launches separate windows/tabs for each model, allowing you to see responses develop in parallel.

```bash
python concurrent_ai_test.py --prompt-file "prompt.txt" --openai-model "gpt-4o" --claude-model "claude-3-7-sonnet-latest" --ollama-model "llama3.1" --gemini-model "gemini-2.0-flash"
```

Arguments:
- `--prompt-file`: Path to the prompt file (required)
- `--system-prompt-file`: Path to system prompt file for Ollama (optional)
- `--openai-model`: OpenAI model to use (default: "o3-mini")
- `--claude-model`: Claude model to use (default: "claude-3-7-sonnet-latest")
- `--ollama-model`: Ollama model to use (default: "llama3.1")
- `--gemini-model`: Gemini model to use (default: "gemini-2.0-flash")
- `--reasoning-effort`: Reasoning effort for OpenAI (optional, "high", "medium", "low")

### Using Individual AI Runners

You can also run queries against specific models using `ai_runner.py`:

```bash
python ai_runner.py --model openai --model-name "gpt-4o" --prompt-file "prompt.txt" --wait
```

Arguments:
- `--model`: Which AI model to use (required, choices: openai, claude, ollama, gemini)
- `--prompt-file`: Path to the prompt file (optional)
- `--system-prompt-file`: Path to system prompt file for Ollama (optional)
- `--model-name`: Specific model name to use (e.g., "gpt-4o", "llama3.1")
- `--wait`: Wait for user input before closing (optional)
- `--reasoning-effort`: Reasoning effort for OpenAI (optional, "high", "auto", "off")

## Supported Models

Any model can be added with a simple json config entry in `aitestbed.py` like so:
```json
    "gemini-2.0-flash": {
        "max_tokens": 8192,
    }
```
Models without configs fallback to 4096 context size but otherwise work fine.

### Preconfigured Models
#### OpenAI
- gpt-4o
- o3-mini (supports reasoning_effort)
- o1 (supports reasoning_effort)

#### Claude
- claude-3-7-sonnet-latest (supports thinking mode)
- claude-3-5-haiku

#### Gemini
- gemini-2.5-pro-exp-03-25
- gemini-2.0-pro
- gemini-2.0-flash
- gemini-2.0-flash-lite


### Ollama (Local Models)
- llama3.1
- gemma3
- (and any other models you have installed locally)

## Advanced Usage
### Model Configuration
The system includes built-in configurations for various models (max tokens, reasoning support, etc.) in `aitestbed.py`. If you're using a model not in the configuration, it will use sensible defaults.

### Colored Output
Responses from different models are color-coded for easy identification:
- OpenAI: Magenta
- Claude: Cyan
- Gemini: Yellow
- Ollama: Green

### Windows Terminal Integration

On Windows, the concurrent tester will attempt to use Windows Terminal to launch separate tabs. If not available, it will fall back to using command prompt windows.

## Troubleshooting

### Common Issues

1. **API Key Errors**: Double-check your `apikeys.json` file is in the correct location and properly formatted.

2. **Model Not Found**: Ensure you're using the correct model identifiers for each service.

3. **Ollama Connection Errors**: Make sure Ollama is running locally (`ollama serve`). If you've used a different/custom url or port then update `run_ollama_query()` and pass a `base_url`

4. **Parameter Errors**: Some models don't support certain parameters (e.g., reasoning_effort). The tool will attempt to filter these out automatically.

## Notes

- Claude models with thinking enabled will show their reasoning process before providing the final answer
- The system automatically handles appropriate token limits for different models
- For best results with local models, ensure Ollama is running before starting tests (kick it if it's asleep via Ollama list on a console window)
