# ai_runner.py - Run a specific AI model with given prompt

import argparse
import aitestbed
import os


def main():
    parser = argparse.ArgumentParser(description='Run a single AI model query')
    parser.add_argument('--model', required=True, choices=['openai', 'claude', 'ollama', 'gemini'],
                        help='Which AI model to use')
    parser.add_argument('--prompt-file', help='File containing the prompt')
    parser.add_argument('--system-prompt-file', help='File containing the system prompt (for Ollama)')
    parser.add_argument('--model-name', help='Specific model name to use (e.g., gpt-4, llama3.1)')
    parser.add_argument('--wait', action='store_true', help='Wait for user input before closing')
    parser.add_argument('--reasoning-effort', help='Reasoning effort for OpenAI (high, auto, off)')

    args = parser.parse_args()

    # Initialize prompt to a default value
    prompt = "No prompt provided"

    # Only try to load the prompt if the file is specified
    if args.prompt_file:
        try:
            if os.path.exists(args.prompt_file):
                prompt = aitestbed.load_prompt_from_file(args.prompt_file)
            else:
                print(f"Warning: Prompt file '{args.prompt_file}' does not exist.")
        except Exception as e:
            print(f"Error loading prompt file: {e}")

    # Print the question for reference
    print(f"QUESTION:\n{prompt}\n")

    # Get the system prompt for Ollama if specified
    system_prompt = None
    if args.system_prompt_file:
        try:
            if os.path.exists(args.system_prompt_file):
                system_prompt = aitestbed.load_prompt_from_file(args.system_prompt_file)
            else:
                print(f"Warning: System prompt file '{args.system_prompt_file}' does not exist.")
        except Exception as e:
            print(f"Error loading system prompt file: {e}")

    # Check for API keys
    try:
        if args.model == 'openai' or args.model == 'claude' or args.model == 'gemini':
            # Look for apikeys.json in the script directory first
            script_dir = os.path.dirname(os.path.abspath(__file__))
            api_keys_path = os.path.join(script_dir, 'apikeys.json')

            # If not in script directory, try current directory
            if not os.path.exists(api_keys_path):
                api_keys_path = 'apikeys.json'

            if not os.path.exists(api_keys_path):
                print(f"Warning: 'apikeys.json' not found in script directory or current directory.")
                print(f"         This file is required for {args.model.upper()} API access.")
                print(f"         Looked in: {script_dir} and {os.getcwd()}")
            else:
                # Set an environment variable for the aitestbed module to use
                os.environ['AITESTBED_API_KEYS_PATH'] = api_keys_path
                print(f"Using API keys from: {api_keys_path}")
    except Exception as e:
        print(f"Error checking for API keys: {e}")

    # Run the appropriate model
    try:
        if args.model == 'openai':
            model_name = args.model_name if args.model_name else "o3-mini"
            print(f"=== OPENAI RESPONSE ({model_name}) ===")
            aitestbed.run_openai_query(prompt, model=model_name, reasoning_effort=args.reasoning_effort)

        elif args.model == 'claude':
            model_name = args.model_name if args.model_name else "claude-3-7-sonnet-latest"
            print(f"=== CLAUDE RESPONSE ({model_name}) ===")
            aitestbed.run_claude_query(prompt, model=model_name)

        elif args.model == 'ollama':
            model_name = args.model_name if args.model_name else "llama3.1"
            print(f"=== OLLAMA RESPONSE ({model_name}) ===")
            aitestbed.run_ollama_query(prompt, model=model_name, system_prompt=system_prompt)

        elif args.model == 'gemini':
            model_name = args.model_name if args.model_name else "gemini-2.0-flash"
            print(f"=== GEMINI RESPONSE ({model_name}) ===")
            aitestbed.run_gemini_query(prompt, model=model_name)

    except Exception as e:
        print(f"Error running {args.model} query: {e}")

    # Wait for user input if requested
    if args.wait:
        input("\nPress Enter to close...")


if __name__ == "__main__":
    main()