import openai
import anthropic
import os
import requests
from halo import Halo
import json
from pathlib import Path

# ANSI escape codes for some colors
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
RESET = "\033[0m"  # Resets the color to default


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
    def __init__(self, api_key):
        """Initialize a Claude conversation with the provided API key"""
        self.client = anthropic.Anthropic(api_key=api_key)
        self.conversation_history = []
        self.model = "claude-3-7-sonnet-latest"

    def ask_with_thinking(self, prompt, model=None, max_tokens=60000, thinking_budget=32000):
        """
        Send a message to Claude with thinking enabled, stream the response, and update conversation history.

        Args:
            prompt (str): The prompt to send to Claude
            model (str, optional): Model to use. Defaults to the instance's model.
            max_tokens (int, optional): Maximum tokens in the response. Defaults to 60000.
            thinking_budget (int, optional): Budget for thinking. Defaults to 32000.

        Returns:
            dict: The complete response from Claude
        """
        if model is None:
            model = self.model

        # Create messages array with conversation history plus new prompt
        messages = self.conversation_history + [
            {"role": "user", "content": prompt}
        ]

        # Track the current block type
        current_block = None
        thinking_started = False
        response_started = False
        full_response = ""

        with self.client.beta.messages.stream(
                model=model,
                max_tokens=max_tokens,
                thinking={
                    "type": "enabled",
                    "budget_tokens": thinking_budget
                },
                messages=messages,
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
                        # Stream thinking content directly (not saved to conversation)
                        print(event.delta.thinking, end="", flush=True)

                    elif event.delta.type == "text_delta":
                        # If we're transitioning from thinking to response
                        if thinking_started and not response_started:
                            print("</thinking>\n")
                            response_started = True

                        # Stream response content directly
                        print(event.delta.text, end="", flush=True)
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

        return full_response

    def reset_conversation(self):
        """Clear the conversation history"""
        self.conversation_history = []
        print("Conversation history has been reset.")

    def get_conversation_history(self):
        """Return the current conversation history"""
        return self.conversation_history

class OpenAIConversation:
    def __init__(self, api_key=None, key_file=None):
        """
        Initialize an OpenAI conversation

        Args:
            api_key (str, optional): OpenAI API key
            key_file (str, optional): Path to a JSON file containing API keys
        """
        self.key_manager = APIKeyManager(key_file)

        # Set API key from parameter or file
        if api_key:
            self.key_manager.set_key('openai', api_key)

        # Initialize client if we have a key
        openai_key = self.key_manager.get_key('openai')
        if openai_key:
            self.client = openai.OpenAI(api_key=openai_key)
        else:
            self.client = None
            print("Warning: No OpenAI API key provided. Please set a key before making requests.")

        self.conversation_history = []
        self.model = "gpt-4-turbo"

    def ask(self, prompt, model=None, max_tokens=4000):
        """
        Send a message to OpenAI and update conversation history

        Args:
            prompt (str): The prompt to send to OpenAI
            model (str, optional): Model to use. Defaults to the instance's model.
            max_tokens (int, optional): Maximum tokens in the response. Defaults to 4000.

        Returns:
            str: The response from OpenAI
        """
        if self.client is None:
            raise ValueError("OpenAI client not initialized. Please provide an API key.")

        if model is None:
            model = self.model

        # Create messages array with conversation history plus new prompt
        messages = [{"role": m["role"], "content": m["content"]} for m in self.conversation_history]
        messages.append({"role": "user", "content": prompt})

        # Make the API call
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens
        )

        # Get the response content
        full_response = response.choices[0].message.content

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": prompt})
        self.conversation_history.append({"role": "assistant", "content": full_response})

        print(full_response)
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

    def __init__(self, model="llama3.1", base_url="http://localhost:11434/v1", api_key="ollama"):
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

    def ask(self, prompt, model=None, system_prompt=None, color=RED):
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
                    print(f"{color}", end="", flush=True)

                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    full_response += content

            print(f"{RESET}")  # Reset color

            # Update conversation history
            self.conversation_history.append({"role": "user", "content": prompt})
            self.conversation_history.append({"role": "assistant", "content": full_response})

            return full_response

        except Exception as e:
            if 'spinner' in locals():
                spinner.stop()
            print(f"{color}Error: {str(e)}{RESET}")
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


# Standalone for quick one-shots without conversation history
def ask_claude_thinking_streaming(prompt):
    client = anthropic.Anthropic()
    apikeys = APIKeyManager("apikeys.json")
    client.api_key = apikeys.get_key("anthropic")

    # Track the current block type
    current_block = None
    thinking_started = False
    response_started = False

    with client.beta.messages.stream(
            model="claude-3-7-sonnet-latest",
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

            elif event.type == "content_block_stop":
                if current_block == "thinking" and not response_started:
                    print("</thinking>\n")

                current_block = None

def print_colored(text, color):
    print(f"{color}{text}{RESET}")



def load_prompt_from_file(filename):
    """Load a prompt from a text file."""
    try:
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as file:
                prompt = file.read().strip()
            print_colored(f"Loaded {filename}", GREEN)
            return prompt
    except Exception as e:
        print_colored(f"Error loading {filename}: {str(e)}", RED)
    return None


# Main execution
if __name__ == "__main__":
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

    # Example: Ask Claude with thinking enabled
    if key_manager.get_key("anthropic"):
        print_colored("\n=== CLAUDE RESPONSE ===", BLUE)
        claude = ClaudeConversation(key_manager.get_key("anthropic"))
        claude.ask_with_thinking(q)

    # Example: Ask Ollama
    print_colored("\n=== OLLAMA RESPONSE ===", GREEN)
    ollama = OllamaConversation(model=ollama_model)
    ollama.ask(q, system_prompt=system_prompt, color=RED)

    # Example: Ask OpenAI
    if key_manager.get_key("openai"):
        print_colored("\n=== OPENAI RESPONSE ===", MAGENTA)
        openai_chat = OpenAIConversation(key_manager.get_key("openai"))
        openai_chat.ask(q)
