import openai
import os
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


def talk_to_ollama(prompt, model, color=RED):
    # Configure the OpenAI client to point to the local Ollama instance
    client = openai.OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # Ollama doesn't require an actual API key
    )

    try:
        # Create a spinner with Halo
        spinner = Halo(text=f'Waiting for response from {model}...', spinner='dots', color='yellow')
        spinner.start()

        # Send the request to the Ollama model with streaming enabled
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,  # Enable streaming
            timeout=60,  # Add a timeout to avoid hanging forever
        )

        # Process the streaming response
        full_response = ""
        first_token_received = False

        for chunk in response:
            if not first_token_received and chunk.choices and chunk.choices[0].delta.content:
                # Stop the spinner when we get the first token
                spinner.stop()
                first_token_received = True
                print(f"{color}", end="", flush=True)  # Set color at the beginning

            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)  # Print immediately without newline
                full_response += content

        print(f"{RESET}")  # Reset color at the end
        return full_response

    except Exception as e:
        # Make sure to stop the spinner if an error occurs
        if 'spinner' in locals():
            spinner.stop()
        print(f"{color}Error: {str(e)}{RESET}")
        return f"Error: {str(e)}"

    except Exception as e:
        print(f"{color}Error: {str(e)}{RESET}")
        return f"Error: {str(e)}"


def load_prompt_from_file(filename="prompt.txt"):
    """Load a prompt from a text file."""
    try:
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as file:
                prompt = file.read().strip()
            print_colored(f"Prompt loaded from {filename}", GREEN)
            return prompt
        else:
            print_colored(f"File {filename} not found. Please enter your prompt manually.", YELLOW)
            return None
    except Exception as e:
        print_colored(f"Error loading prompt from file: {str(e)}", RED)
        return None


# Main execution
if __name__ == "__main__":
    # First, try to load the prompt from the file
    file_prompt = load_prompt_from_file()

    # If the file exists and was loaded successfully, use it; otherwise, ask for input
    if file_prompt:
        q = file_prompt
        print_colored("Using prompt from file:", GREEN)
        print(q)
    else:
        q = input("Enter your question: ")

    print(f"\nQUESTION: {q}")

    # Note about required dependency
    try:
        import halo
    except ImportError:
        print_colored("\nNOTE: This script requires the 'halo' package. Please install it with:", YELLOW)
        print_colored("pip install halo", CYAN)
        print_colored("Then run this script again.\n", YELLOW)

    # Use the correct model name format for Ollama
    r1 = talk_to_ollama(q, 'llama3.1', RED)