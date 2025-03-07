import openai
import os
import json
import requests
from halo import Halo

# ANSI escape codes for some colors
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
RESET = "\033[0m"  # Resets the color to default


def print_colored(text, color):
    print(f"{color}{text}{RESET}")


def get_model_context_size(model):
    """
    Attempt to fetch the model's max context size from Ollama.
    It finds the best match for the given model base name (e.g. "llama3.1")
    by considering models that start with that base name. Models with a specific
    configuration (like "llama3.1:32b") are preferred over the default ":latest" version.
    Also prints out all key stats from the model's return.
    If the model isn't found, returns -1.
    """
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status()
        data = response.json()

        # Find candidates that match the base model name.
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

        # Print key model stats.
        print("Model Stats:")
        print(f"  Name: {chosen.get('name')}")
        print(f"  Model: {chosen.get('model')}")
        print(f"  Modified At: {chosen.get('modified_at')}")                
        print("  Details:")
        details = chosen.get('details', {})
        for key, value in details.items():
            print(f"    {key}: {value}")

        # Return the model's context size, defaulting to 4096 if not provided.
        return chosen.get("context_size", 4096)

    except requests.RequestException:
        print_colored("Warning: Could not retrieve context size from Ollama. Returning unknown (-1).", YELLOW)
        return -1



def talk_to_ollama(prompt, model, system_prompt=None, color=RED, context_size_override=None):
    """
    Sends a request to the Ollama model with an optional system prompt.

    Note: The Ollama OpenAI compatibility does not support a 'context_size' parameter.
    If you need a custom context size, create a new model using a Modelfile (see docs),
    and then use that model name here.
    """
    client = openai.OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # Ollama doesn't require an actual API key
    )

    try:
        # Use the override for display if provided; otherwise, fetch from Ollama.
        if context_size_override is not None:
            display_context_size = context_size_override
        else:
            display_context_size = get_model_context_size(model)

        context_str = str(display_context_size) if display_context_size != -1 else "UNKNOWN"
        spinner = Halo(text=f'Waiting for response from {model} (context: {context_str})...', spinner='dots',
                       color='yellow')
        spinner.start()

        # Build the message list, including system prompt if available.
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Since context size cannot be set at runtime via the API, no extra parameter is passed.
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,  # Enable streaming
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
        return full_response

    except Exception as e:
        if 'spinner' in locals():
            spinner.stop()
        print(f"{color}Error: {str(e)}{RESET}")
        return f"Error: {str(e)}"


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
    # Set this variable to an integer (e.g., 8192) if you have created a custom model with that context size.
    # Otherwise, leave it as None to use the model's default context size.
    OVERRIDE_CONTEXT_SIZE = None

    user_prompt = load_prompt_from_file("prompt.txt")
    system_prompt = load_prompt_from_file("system_prompt.txt")

    if user_prompt:
        q = user_prompt
        print_colored("Using prompt from file:", GREEN)
        print(q)
    else:
        q = input("Enter your question: ")

    print(f"\nQUESTION: {q}")

    try:
        import halo
    except ImportError:
        print_colored("\nNOTE: This script requires the 'halo' package. Install it with:", YELLOW)
        print_colored("pip install halo", CYAN)

    # If you want Ollama to use a custom context size, ensure you've created a model with that context size.
    # The OVERRIDE_CONTEXT_SIZE is used here only for display purposes.
    r1 = talk_to_ollama(q, 'llama3.1', system_prompt, RED, context_size_override=OVERRIDE_CONTEXT_SIZE)
