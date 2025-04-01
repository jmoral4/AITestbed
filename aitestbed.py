import openai
import anthropic
import os
import requests
from halo import Halo
import json
from pathlib import Path
import google.generativeai as genai
import datetime
import re
import tiktoken

# ANSI escape codes for some colors
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
RESET = "\033[0m"  # Resets the color to default

# Model configurations
MODEL_CONFIGS = {
    # OpenAI models
    "gpt-4o": {
        "max_tokens": 16384,
        "supports_reasoning": False,
    },
    "o3-mini": {
        "max_tokens": 100000,
        "supports_reasoning": True,
    },
    "o1": {
        "max_tokens": 100000,
        "supports_reasoning": True,
    },
    "claude-3-7-sonnet-latest": {
        "max_tokens": 30720,
        "thinking_enabled": True,
        "thinking_budget": 32000,
        "max_tokens_with_thinking": 128000,
    },
    "claude-3-5-haiku-latest": {
        "max_tokens": 100000,
        "thinking_enabled": False,
    },
    "gemini-2.5-pro-exp-03-25": {
        "max_tokens": 65636,
    },
    "gemini-2.0-flash": {
        "max_tokens": 8192,
    },
    "gemini-2.0-flash-lite": {
        "max_tokens": 8192,
    },
    "gemini-2.0-pro": {
        "max_tokens": 16384,
    },
    "llama3.1": {
        "max_tokens": 4096,
    },
    "gemma3": {
        "max_tokens": 4096,
    },
}

# Default configuration to use when model isn't found
DEFAULT_CONFIG = {
    "max_tokens": 4096,
    "supports_reasoning": False,
    "thinking_enabled": False,
}


def get_model_config(model_name):
    """Get the configuration for a specific model, with fallback to defaults"""
    return MODEL_CONFIGS.get(model_name, DEFAULT_CONFIG)


def count_tokens(text, model="claude-3-7-sonnet-latest"):
    """
    Estimate token count for a given text using tiktoken.

    Args:
        text (str): The text to count tokens for
        model (str): The model to use for token counting

    Returns:
        int: Estimated token count
    """
    try:


        # Map model names to encoding types
        # This is a simplified mapping; add more as needed
        model_to_encoding = {
            # OpenAI models generally use cl100k_base for newer models
            "gpt-4o": "cl100k_base",
            "o3-mini": "cl100k_base",
            "o1": "cl100k_base",

            # Claude models - we'll use cl100k as approximation
            "claude-3-7-sonnet-latest": "cl100k_base",
            "claude-3-5-haiku-latest": "cl100k_base",

            # Gemini models - use cl100k as approximation
            "gemini-2.5-pro-exp-03-25": "cl100k_base",
            "gemini-2.0-flash": "cl100k_base",

            # Default for other models
            "default": "cl100k_base"
        }

        # Get the encoding type based on model
        encoding_name = model_to_encoding.get(model, model_to_encoding["default"])
        encoding = tiktoken.get_encoding(encoding_name)

        # Count tokens
        tokens = encoding.encode(text)
        return len(tokens)

    except ImportError:
        # If tiktoken is not available, provide a rough estimate
        # This is very approximate (assuming ~4 chars per token)
        print("Warning: tiktoken not installed. Using rough estimate (~4 chars/token).")
        return len(text) // 4


class ResponseSaver:
    """
    A reusable class for saving AI responses to files with standardized naming.
    """

    def __init__(self, output_dir="responses"):
        """
        Initialize the ResponseSaver.

        Args:
            output_dir (str, optional): Directory to save responses. Defaults to "responses".
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def save_response(self, prompt, response, model):
        """
        Save an AI response to a markdown file with the format:
        <timestamp:HHMMSS>.<model>.<first 50 chars of prompt>.md

        Args:
            prompt (str): The user's prompt
            response (str): The AI's response
            model (str): The model name

        Returns:
            str: Path to the saved file
        """
        # Generate timestamp (HHMMSS)
        timestamp = datetime.datetime.now().strftime("%H%M%S")

        # Get first 30 chars of prompt and strip non-alphanumeric characters
        prompt_part = re.sub(r'[^a-zA-Z0-9]', '', prompt[:50])

        # Clean model name (remove non-alphanumeric characters)
        model_clean = re.sub(r'[^a-zA-Z0-9]', '', model)

        # Create filename
        filename = f"{timestamp}.{model_clean}.{prompt_part}.md"

        # Full path to file
        file_path = os.path.join(self.output_dir, filename)

        # Write response to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"# Prompt: {prompt}\n\n")
            f.write(f"## Model: {model}\n\n")
            f.write(response)

        print(f"Response saved to: {file_path}")
        return file_path


# Create a global response saver instance
response_saver = ResponseSaver()


class APIKeyManager:
    """Manages API keys for different providers"""

    def __init__(self, key_file=None):
        """
        Initialize the API key manager

        Args:
            key_file (str, optional): Path to a JSON file containing API keys
        """
        self.keys = {}
        if key_file:
            self.load_keys_from_file(key_file)

    def load_keys_from_file(self, file_path):
        """
        Load API keys from a JSON file

        Args:
            file_path (str): Path to the JSON file

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            file_path = Path(file_path).expanduser().resolve()
            if not file_path.exists():
                print(f"Key file not found: {file_path}")
                return False

            with open(file_path, 'r') as f:
                self.keys = json.load(f)
            return True
        except Exception as e:
            print(f"Error loading API keys: {e}")
            return False

    def get_key(self, provider):
        """
        Get an API key for the specified provider

        Args:
            provider (str): The API provider (e.g., 'anthropic', 'openai')

        Returns:
            str: The API key, or None if not found
        """
        return self.keys.get(provider)

    def set_key(self, provider, key):
        """
        Set an API key for the specified provider

        Args:
            provider (str): The API provider
            key (str): The API key
        """
        self.keys[provider] = key

    def save_keys_to_file(self, file_path):
        """
        Save API keys to a JSON file

        Args:
            file_path (str): Path to the JSON file

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            file_path = Path(file_path).expanduser().resolve()
            os.makedirs(file_path.parent, exist_ok=True)

            with open(file_path, 'w') as f:
                json.dump(self.keys, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving API keys: {e}")
            return False


class ClaudeConversation:
    def __init__(self, api_key, color=None):
        """Initialize a Claude conversation with the provided API key"""
        self.client = anthropic.Anthropic(api_key=api_key)
        self.conversation_history = []
        self.model = "claude-3-7-sonnet-latest"
        self.color = color

    def ask_with_thinking(self, prompt, model=None, max_tokens=None, thinking_budget=None):
        """
        Send a message to Claude with thinking enabled, stream the response, and update conversation history.

        Args:
            prompt (str): The prompt to send to Claude
            model (str, optional): Model to use. Defaults to the instance's model.
            max_tokens (int, optional): Maximum tokens in the response. Defaults to model's config.
            thinking_budget (int, optional): Budget for thinking. Defaults to model's config.

        Returns:
            dict: The complete response from Claude
        """
        if model is None:
            model = self.model

        # Get model configuration
        config = get_model_config(model)

        # Use provided values or fall back to config
        if max_tokens is None:
            if config.get("thinking_enabled", False):
                max_tokens = config.get("max_tokens_with_thinking", 30720)
            else:
                max_tokens = config.get("max_tokens", 30720)

        if thinking_budget is None:
            thinking_budget = config.get("thinking_budget", 32000)

        # Create messages array with conversation history plus new prompt
        messages = self.conversation_history + [
            {"role": "user", "content": prompt}
        ]

        # Track the current block type
        current_block = None
        thinking_started = False
        response_started = False
        full_response = ""

        # Only enable thinking if the model supports it
        thinking_params = {}
        if config.get("thinking_enabled", False):
            thinking_params = {
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": thinking_budget
                },
                "betas": ["output-128k-2025-02-19"]
            }

        with self.client.beta.messages.stream(
                model=model,
                max_tokens=max_tokens,
                messages=messages,
                **thinking_params
        ) as stream:
            for event in stream:
                if event.type == "content_block_start":
                    current_block = event.content_block.type

                    # Print the thinking tag when thinking block starts
                    if current_block == "thinking" and not thinking_started:
                        print("<thinking>")
                        thinking_started = True

                elif event.type == "content_block_delta":
                    if event.delta.type == "thinking_delta":
                        # Stream thinking content directly (not saved to conversation)
                        print(event.delta.thinking, end="", flush=True)

                    elif event.delta.type == "text_delta":
                        # If we're transitioning from thinking to response
                        if thinking_started and not response_started:
                            print("</thinking>\n")
                            response_started = True

                        # Stream response content directly
                        print_colored(event.delta.text, self.color)
                        # Also accumulate for conversation history
                        full_response += event.delta.text

                elif event.type == "content_block_stop":
                    if current_block == "thinking" and not response_started:
                        print("</thinking>\n")

                    current_block = None

                elif event.type == "message_delta":
                    # This captures other message information like stop reason
                    pass

                elif event.type == "message_stop":
                    # Final event when message is complete
                    pass

        # Update conversation history with the new user prompt and assistant response
        self.conversation_history.append({"role": "user", "content": prompt})
        self.conversation_history.append({"role": "assistant", "content": full_response})

        # Save response to file
        response_saver.save_response(prompt, full_response, model)

        return full_response

    def reset_conversation(self):
        """Clear the conversation history"""
        self.conversation_history = []
        print("Conversation history has been reset.")

    def get_conversation_history(self):
        """Return the current conversation history"""
        return self.conversation_history


class OpenAIConversation:
    def __init__(self, api_key=None, model="o3-mini", reasoning_effort=None, color=None):
        """
        Initialize an OpenAI conversation

        Args:
            api_key (str, required): OpenAI API key
            model (str, optional): Default model to use
            reasoning_effort (str, optional): Reasoning effort setting
            color (str, optional): Color for output
        """

        # Initialize client if we have a key
        if api_key:
            self.client = openai.OpenAI(api_key=api_key)
        else:
            self.client = None
            print("ERROR: No OpenAI API key provided. Please set a key before making requests.")

        self.conversation_history = []
        self.model = model
        self.color = color
        self.reasoning_effort = reasoning_effort

    def ask(self, prompt, model=None, max_tokens=None):
        """
        Send a message to OpenAI and update conversation history

        Args:
            prompt (str): The prompt to send to OpenAI
            model (str, optional): Model to use. Defaults to the instance's model.
            max_tokens (int, optional): Maximum tokens in the response. If None, use model config.

        Returns:
            str: The response from OpenAI
        """
        if self.client is None:
            raise ValueError("OpenAI client not initialized. Please provide an API key.")

        if model is None:
            model = self.model

        # Get model configuration
        config = get_model_config(model)

        # Use provided max_tokens or fall back to config
        if max_tokens is None:
            max_tokens = config.get("max_tokens", 4096)

        # Create messages array with conversation history plus new prompt
        messages = [{"role": m["role"], "content": m["content"]} for m in self.conversation_history]
        messages.append({"role": "user", "content": prompt})

        # Prepare API call parameters
        params = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_tokens
        }

        # Only add reasoning_effort if the model supports it and it's provided
        if self.reasoning_effort is not None and config.get("supports_reasoning", False):
            params["reasoning_effort"] = self.reasoning_effort

        # Make the API call
        response = self.client.chat.completions.create(**params)

        # Get the response content
        full_response = response.choices[0].message.content

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": prompt})
        self.conversation_history.append({"role": "assistant", "content": full_response})

        print_colored(full_response, self.color)

        # Save response to file
        response_saver.save_response(prompt, full_response, model)

        return full_response

    def reset_conversation(self):
        """Clear the conversation history"""
        self.conversation_history = []
        print("Conversation history has been reset.")

    def get_conversation_history(self):
        """Return the current conversation history"""
        return self.conversation_history


class OllamaConversation:
    """Manages conversations with Ollama models"""

    def __init__(self, model="llama3.1", base_url="http://localhost:11435/v1", api_key="ollama", color=None):
        """
        Initialize an Ollama conversation

        Args:
            model (str, optional): The default model to use. Defaults to "llama3.1".
            base_url (str, optional): The Ollama API base URL. Defaults to "http://localhost:11434/v1".
            api_key (str, optional): API key (Ollama doesn't require a real key). Defaults to "ollama".
        """
        self.model = model
        self.client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.conversation_history = []
        self.context_size = self.get_model_context_size(model)
        self.color = color

    def get_model_context_size(self, model):
        """
        Fetch the model's max context size from Ollama.

        Args:
            model (str): The model name

        Returns:
            int: The context size or -1 if unknown
        """
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            response.raise_for_status()
            data = response.json()

            # Find candidates that match the base model name
            candidates = []
            for item in data.get("models", []):
                name = item.get("name", "")
                if name == model or name.startswith(model + ":"):
                    candidates.append(item)

            if not candidates:
                print_colored(f"Warning: Model '{model}' not found in Ollama response. Returning unknown (-1).", YELLOW)
                return -1

            # Prefer models that do not end with ':latest'
            non_latest_candidates = [c for c in candidates if not c.get("name", "").endswith(":latest")]
            chosen = non_latest_candidates[0] if non_latest_candidates else candidates[0]

            # Print key model stats
            print("Model Stats:")
            print(f"  Name: {chosen.get('name')}  Model: {chosen.get('model')}")
            print("  Details:")
            details = chosen.get('details', {})
            for key, value in details.items():
                print(f"    {key}: {value}")

            return chosen.get("context_size", 4096)

        except requests.RequestException:
            print_colored("Warning: Could not retrieve context size from Ollama. Returning unknown (-1).", YELLOW)
            return -1

    def ask(self, prompt, model=None, system_prompt=None):
        """
        Send a message to Ollama, stream the response, and update conversation history

        Args:
            prompt (str): The prompt to send to Ollama
            model (str, optional): Model to use. Defaults to the instance's model.
            system_prompt (str, optional): System prompt to guide the model. Defaults to None.
            color (str, optional): ANSI color for output. Defaults to RED.

        Returns:
            str: The complete response from Ollama
        """
        if model is None:
            model = self.model

        # If model changed, update context size
        if model != self.model:
            self.model = model
            self.context_size = self.get_model_context_size(model)

        try:
            # Display context size info
            display_context_size = self.context_size if self.context_size != -1 else "UNKNOWN"

            spinner = Halo(
                text=f'Waiting for response from {model} (context: {display_context_size})...',
                spinner='dots',
                color='yellow'
            )
            spinner.start()

            # Build the message list, including system prompt if available
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            # Add conversation history
            for msg in self.conversation_history:
                messages.append({"role": msg["role"], "content": msg["content"]})

            # Add the new prompt
            messages.append({"role": "user", "content": prompt})

            # Make the API call with streaming
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                timeout=60,
            )

            full_response = ""
            first_token_received = False

            for chunk in response:
                if not first_token_received and chunk.choices and chunk.choices[0].delta.content:
                    spinner.stop()
                    first_token_received = True
                    print_colored("", f"{self.color}")

                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    print_colored(content, self.color)
                    full_response += content

            # Update conversation history
            self.conversation_history.append({"role": "user", "content": prompt})
            self.conversation_history.append({"role": "assistant", "content": full_response})

            # Save response to file
            response_saver.save_response(prompt, full_response, model)

            return full_response

        except Exception as e:
            if 'spinner' in locals():
                spinner.stop()
            print(f"{self.color}Error: {str(e)}{RESET}")
            return f"Error: {str(e)}"

    def reset_conversation(self):
        """Clear the conversation history"""
        self.conversation_history = []
        print("Conversation history has been reset.")

    def get_conversation_history(self):
        """Return the current conversation history"""
        return self.conversation_history

    def print_model_stats(self):
        """Print the current model's statistics"""
        if self.context_size != -1:
            print(f"Model: {self.model}, Context Size: {self.context_size}")
        else:
            print(f"Model: {self.model}, Context Size: UNKNOWN")


class GeminiConversation:
    def __init__(self, api_key, model="gemini-2.0-flash", color=None):
        """
        Initialize a Gemini conversation with the provided API key

        Args:
            api_key (str): Gemini API key
            model (str, optional): The model to use. Defaults to "gemini-1.5-pro".
            color (str, optional): ANSI color for output. Defaults to None.
        """
        try:

            self.genai = genai

            self.genai.configure(api_key=api_key)
            self.model = model
            self.color = color
            self.model_instance = self.genai.GenerativeModel(model)
            self.chat_session = self.model_instance.start_chat(history=[])
            self.conversation_history = []
        except ImportError:
            print_colored(
                "Error: google-generativeai package not installed. Please install it with 'pip install google-generativeai'",
                RED)
            raise
        except Exception as e:
            print_colored(f"Error initializing Gemini: {str(e)}", RED)
            raise

    def ask(self, prompt, model=None, max_tokens=None):
        """
        Send a message to Gemini, stream the response, and update conversation history

        Args:
            prompt (str): The prompt to send to Gemini
            model (str, optional): Model to use. Defaults to the instance's model.
            max_tokens (int, optional): Maximum tokens in the response. If None, use model config.

        Returns:
            str: The complete response from Gemini
        """
        try:
            if model and model != self.model:
                self.model = model
                self.model_instance = self.genai.GenerativeModel(model)
                # Create a new chat session for the new model
                self.chat_session = self.model_instance.start_chat(history=[])
                # Note: This loses conversation history when changing models

            # Get model configuration
            config = get_model_config(model or self.model)

            # Use provided max_tokens or fall back to config
            if max_tokens is None:
                max_tokens = config.get("max_tokens", 8192)

            # Add the new prompt to conversation history for tracking
            self.conversation_history.append({"role": "user", "content": prompt})

            # Send the message and get the response
            response = self.chat_session.send_message(
                prompt,
                stream=True,
                generation_config={
                    "max_output_tokens": max_tokens,
                }
            )

            full_response = ""
            for chunk in response:
                # Different versions of the API might return different objects
                if hasattr(chunk, "text"):
                    text = chunk.text
                elif hasattr(chunk, "parts") and len(chunk.parts) > 0:
                    text = str(chunk.parts[0])
                else:
                    text = str(chunk)

                print_colored(text, self.color)
                full_response += text

            # Add the response to conversation history for tracking
            self.conversation_history.append({"role": "assistant", "content": full_response})

            # Save response to file
            response_saver.save_response(prompt, full_response, model or self.model)

            return full_response
        except Exception as e:
            error_msg = f"Gemini API error: {str(e)}"
            print_colored(error_msg, RED)
            return error_msg

    def reset_conversation(self):
        """Clear the conversation history by starting a new chat session"""
        self.chat_session = self.model_instance.start_chat(history=[])
        self.conversation_history = []
        print("Conversation history has been reset.")

    def get_conversation_history(self):
        """Return the current conversation history in a format compatible with other models"""
        return self.conversation_history


# Standalone for quick one-shots without conversation history
def ask_claude_thinking_streaming(prompt):
    client = anthropic.Anthropic()
    apikeys = APIKeyManager("apikeys.json")
    client.api_key = apikeys.get_key("anthropic")
    model = "claude-3-7-sonnet-latest"

    # Track the current block type
    current_block = None
    thinking_started = False
    response_started = False
    full_response = ""

    with client.beta.messages.stream(
            model=model,
            max_tokens=60000,
            thinking={
                "type": "enabled",
                "budget_tokens": 32000
            },
            messages=[{
                "role": "user",
                "content": prompt
            }],
            betas=["output-128k-2025-02-19"],
    ) as stream:
        for event in stream:
            if event.type == "content_block_start":
                current_block = event.content_block.type

                # Print the thinking tag when thinking block starts
                if current_block == "thinking" and not thinking_started:
                    print("<thinking>")
                    thinking_started = True

            elif event.type == "content_block_delta":
                if event.delta.type == "thinking_delta":
                    # Stream thinking content directly
                    print(event.delta.thinking, end="", flush=True)

                elif event.delta.type == "text_delta":
                    # If we're transitioning from thinking to response
                    if thinking_started and not response_started:
                        print("</thinking>\n")
                        response_started = True

                    # Stream response content directly
                    print(event.delta.text, end="", flush=True)
                    full_response += event.delta.text

            elif event.type == "content_block_stop":
                if current_block == "thinking" and not response_started:
                    print("</thinking>\n")

                current_block = None

    # Save response to file
    response_saver.save_response(prompt, full_response, model)

    return full_response


def print_colored(text, color):
    print(f"{color}{text}{RESET}", end="", flush=True)


def load_prompt_from_file(filename, model="claude-3-7-sonnet-latest"):
    """Load a prompt from a text file with token counting."""
    try:
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as file:
                prompt = file.read().strip()

            # Count tokens
            token_count = count_tokens(prompt, model)

            print_colored(f"Loaded {filename} ({token_count} tokens)\n", YELLOW)
            return prompt
    except Exception as e:
        print_colored(f"Error loading {filename}: {str(e)}\n", RED)
    return None


def run_openai_query(prompt, api_key=None, model="o3-mini", key_file="apikeys.json", reasoning_effort=None):
    """Run a query against OpenAI models"""
    if not api_key:
        key_manager = APIKeyManager(key_file)
        api_key = key_manager.get_key("openai")
        if not api_key:
            print_colored("Error: No OpenAI API key found\n", RED)
            return

    # Get the model configuration
    config = get_model_config(model)
    max_tokens = config.get("max_tokens", 4096)

    # Only pass reasoning_effort if the model supports it
    if reasoning_effort is not None and not config.get("supports_reasoning", False):
        print_colored(f"Note: {model} does not support reasoning_effort. This parameter will be ignored.\n", YELLOW)
        reasoning_effort = None

    openai_chat = OpenAIConversation(api_key, model=model, color=YELLOW, reasoning_effort=reasoning_effort)
    return openai_chat.ask(prompt, max_tokens=max_tokens)


def run_claude_query(prompt, api_key=None, model="claude-3-7-sonnet-latest", key_file="apikeys.json"):
    """Run a query against Claude models with thinking enabled"""
    if not api_key:
        key_manager = APIKeyManager(key_file)
        api_key = key_manager.get_key("anthropic")
        if not api_key:
            print_colored("Error: No Anthropic API key found\n", RED)
            return

    # Get the model configuration
    config = get_model_config(model)

    # Log a note if thinking is not enabled but we're using the thinking function
    if not config.get("thinking_enabled", False):
        print_colored(f"Note: {model} does not support thinking. Using standard API call.\n", YELLOW)

    claude = ClaudeConversation(api_key, CYAN)
    return claude.ask_with_thinking(prompt, model=model)


def run_gemini_query(prompt, api_key=None, model="gemini-2.0-flash", key_file="apikeys.json"):
    """Run a query against Google Gemini models"""
    if not api_key:
        key_manager = APIKeyManager(key_file)
        api_key = key_manager.get_key("gemini")
        if not api_key:
            print_colored("Error: No Gemini API key found\n", RED)
            return

    # Get the model configuration
    config = get_model_config(model)
    max_tokens = config.get("max_tokens", 8192)

    try:
        gemini = GeminiConversation(api_key, model=model, color=YELLOW)
        return gemini.ask(prompt, max_tokens=max_tokens)
    except ImportError:
        print_colored("Skipping Gemini (google-generativeai package not installed)\n", YELLOW)
        return None
    except Exception as e:
        print_colored(f"Error initializing Gemini: {str(e)}\n", RED)
        return None


def run_ollama_query(prompt, model="llama3.1", system_prompt=None):
    """Run a query against Ollama models"""
    ollama = OllamaConversation(model=model, color=GREEN)
    return ollama.ask(prompt, system_prompt=system_prompt)


# Main execution for when the script is run directly
def main():
    """Main function for direct execution of the script"""
    # Initialize key manager
    key_manager = APIKeyManager("apikeys.json")

    # Default model for Ollama
    ollama_model = 'llama3.1'

    # Load prompts
    user_prompt = load_prompt_from_file("prompt.txt")
    system_prompt = load_prompt_from_file("system_prompt.txt")

    # Prepare the prompt
    if user_prompt:
        q = user_prompt
        print_colored("Using prompt from file:", GREEN)
        print(q)
    else:
        q = input("Enter your question: ")

    # Display the question
    print(f"\nQUESTION: {q}\n")

    # Ask OpenAI
    if key_manager.get_key("openai"):
        print_colored("\n=== OPENAI RESPONSE ===\n", MAGENTA)
        run_openai_query(q)

    # Ask Claude
    if key_manager.get_key("anthropic"):
        print_colored("\n=== CLAUDE RESPONSE ===\n", CYAN)
        run_claude_query(q)

    # Ask Gemini
    if key_manager.get_key("gemini"):
        print_colored("\n=== GEMINI RESPONSE ===\n", YELLOW)
        run_gemini_query(q)

    # Ask Ollama
    print_colored("\n=== OLLAMA RESPONSE ===\n", GREEN)
    run_ollama_query(q, system_prompt=system_prompt)


if __name__ == "__main__":
    main()