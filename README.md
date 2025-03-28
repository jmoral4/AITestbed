# AI Model Conversation Tool

A simple yet powerful Python utility for interacting with multiple AI language models through a consistent interface.

## Features

- **Multi-model support**: Talk to Claude, OpenAI, and Ollama models with the same code structure
- **Conversation history tracking**: Maintain context within each model conversation
- **Streaming responses**: See responses as they're generated
- **Thinking mode support**: See Claude's thinking process (Claude 3.7+ only)
- **Class-based architecture**: Consistent, maintainable code structure

## Requirements

- Python 3.7+
- Required packages: `openai`, `anthropic`, `requests`, `halo`

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install openai anthropic requests halo
```
3. Create an `apikeys.json` file with your API keys:
```json
{
  "anthropic": "your-anthropic-key",
  "openai": "your-openai-key"
}
```
4. For Ollama, ensure you have it running locally at the default URL.

## Usage

### Basic Usage

```python
# Initialize key manager
key_manager = APIKeyManager("apikeys.json")
    
claude = ClaudeConversation(key_manager.get_key("anthropic")) 
openai_chat = OpenAIConversation(key_manager.get_key("openai"))
ollama = OllamaConversation(model="llama3.1")

# Ask questions
claude.ask_with_thinking("What are three interesting facts about quantum computing?")
openai_chat.ask("Write a short poem about programming.")
ollama.ask("Explain how transformers work in machine learning.", system_prompt="Be concise and clear.")
```

### Working with Files

To use prompt files instead of direct input:

1. Create a `prompt.txt` file with your question
2. Create a `system_prompt.txt` file with your system prompt (for Ollama and OpenAI)
3. Run the script.

## Class Overview

- `APIKeyManager`: Handles loading, storage, and retrieval of API keys
- `ClaudeConversation`: Interface for Claude models with thinking mode support
- `OpenAIConversation`: Interface for OpenAI GPT models
- `OllamaConversation`: Interface for locally-run Ollama models

## Customization

- Change the default model by modifying the model field in each class
- Adjust output colors by changing the ANSI color constants
- Modify context sizes for different performance characteristics

## License

MIT

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.