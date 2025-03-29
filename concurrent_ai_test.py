# concurrent_ai_test.py - Spawns separate windows for each AI model

import subprocess
import sys
import os
import argparse
import aitestbed


def check_windows_terminal():
    """Check if Windows Terminal is available"""
    try:
        # Try to run 'wt --version' to see if Windows Terminal is installed
        result = subprocess.run(['wt', '--version'],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        return result.returncode == 0
    except FileNotFoundError:
        return False
    except Exception:
        return False


def run_in_windows_terminal(title, command, working_dir):
    """Run a command in a new Windows Terminal tab with specified working directory"""
    try:
        # Try the new syntax first with working directory
        subprocess.Popen(['wt', '-w', '0', 'nt', '--title', title, '-d', working_dir, 'cmd', '/c', command])
    except Exception as e:
        print(f"First Windows Terminal method failed: {e}")
        try:
            # Fall back to older syntax if the first one fails
            subprocess.Popen(['wt', 'new-tab', '--title', title, '-d', working_dir, '--', 'cmd', '/c', command])
        except Exception as e:
            print(f"Second Windows Terminal method failed: {e}")
            # If all else fails, use CMD as a fallback
            run_in_cmd(title, command, working_dir)


def run_in_cmd(title, command, working_dir):
    """Run a command in a new CMD window with specified working directory"""
    # Use /D flag to set working directory
    subprocess.Popen(f'start "{title}" cmd /c "cd /D "{working_dir}" && {command}"', shell=True)


def main(args=None):
    parser = argparse.ArgumentParser(description='Run AI models concurrently in separate windows')
    parser.add_argument('--prompt-file', required=True, help='File containing the prompt')
    parser.add_argument('--system-prompt-file', help='File containing the system prompt for Ollama')
    parser.add_argument('--openai-model', default='o3-mini', help='OpenAI model to use')
    parser.add_argument('--claude-model', default='claude-3-7-sonnet-latest', help='Claude model to use')
    parser.add_argument('--ollama-model', default='llama3.1', help='Ollama model to use')
    parser.add_argument('--gemini-model', default='gemini-2.0-flash', help='gemini model to use')
    parser.add_argument('--reasoning-effort', help='Reasoning effort for OpenAI (high, medium, low)')

    # If args is provided, parse them, otherwise use sys.argv
    parsed_args = parser.parse_args(args)

    # Check if prompt file exists
    if not os.path.exists(parsed_args.prompt_file):
        print(f"Error: Prompt file '{parsed_args.prompt_file}' does not exist.")
        return

    # Check if system prompt file exists (if specified)
    if parsed_args.system_prompt_file and not os.path.exists(parsed_args.system_prompt_file):
        print(f"Warning: System prompt file '{parsed_args.system_prompt_file}' does not exist.")

    # Get script directory - this is where apikeys.json should be
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Check if apikeys.json exists for OpenAI and Claude
    api_keys_path = os.path.join(script_dir, 'apikeys.json')
    if not os.path.exists(api_keys_path):
        print(f"Warning: 'apikeys.json' not found in {script_dir}")
        print("OpenAI and Claude runners may fail.")
    else:
        print(f"Found API keys file: {api_keys_path}")

    # Get the absolute path to the prompt file
    prompt_file = os.path.abspath(parsed_args.prompt_file)
    prompt_arg = f'--prompt-file "{prompt_file}"'

    # System prompt argument
    system_prompt_arg = ''
    if parsed_args.system_prompt_file:
        system_prompt_file = os.path.abspath(parsed_args.system_prompt_file)
        system_prompt_arg = f'--system-prompt-file "{system_prompt_file}"'

    # Reasoning effort argument for OpenAI
    reasoning_arg = f'--reasoning-effort {parsed_args.reasoning_effort}' if parsed_args.reasoning_effort else ""

    # Get the path to the ai_runner.py script
    # Assuming it's in the same directory as this script
    runner_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ai_runner.py')

    # Get Python executable
    python_exe = sys.executable

    # Check if Windows Terminal is available
    use_windows_terminal = True

    # Use double quotes around the entire command to handle paths with spaces
    openai_cmd = f'{python_exe} "{runner_script}" --model openai --model-name {parsed_args.openai_model} {prompt_arg} {reasoning_arg} --wait'
    claude_cmd = f'{python_exe} "{runner_script}" --model claude --model-name {parsed_args.claude_model} {prompt_arg} --wait'
    ollama_cmd = f'{python_exe} "{runner_script}" --model ollama --model-name {parsed_args.ollama_model} {prompt_arg} {system_prompt_arg} --wait'
    gemini_cmd = f'{python_exe} "{runner_script}" --model gemini --model-name {parsed_args.gemini_model} {prompt_arg} --wait'

    print(f"Using prompt file: {prompt_file}")

    # Run the commands in separate windows
    # Pass the script directory as the working directory to ensure API keys are found
    if use_windows_terminal:
        print("Using Windows Terminal to spawn separate tabs...")
        run_in_windows_terminal(f"OpenAI ({parsed_args.openai_model})", openai_cmd, script_dir)
        run_in_windows_terminal(f"Claude ({parsed_args.claude_model})", claude_cmd, script_dir)
        run_in_windows_terminal(f"Ollama ({parsed_args.ollama_model})", ollama_cmd, script_dir)
        run_in_windows_terminal(f"Gemini ({parsed_args.gemini_model})", gemini_cmd, script_dir)
    else:
        print("Using CMD to spawn separate windows...")
        run_in_cmd(f"OpenAI ({parsed_args.openai_model})", openai_cmd, script_dir)
        run_in_cmd(f"Claude ({parsed_args.claude_model})", claude_cmd, script_dir)
        run_in_cmd(f"Ollama ({parsed_args.ollama_model})", ollama_cmd, script_dir)
        run_in_cmd(f"Gemini ({parsed_args.gemini_model})", gemini_cmd, script_dir)

    print("All AI model processes have been started in separate windows.")
    print("If any model fails, check that apikeys.json exists and contains valid API keys.")


if __name__ == "__main__":
    # Call main with command line arguments as a list
    main(["--prompt-file", "prompt.txt",
          "--openai-model", "gpt-4o",
          "--claude-model", "claude-3-7-sonnet-latest",
          "--ollama-model", "gemma3:27b",
          "--gemini-model", "gemini-2.5-pro-exp-03-25"])

            # Other Options
            # gemini-2.5-pro-exp-03-25
            # gemini-2.0-flash-lite (fast and cheap)
            # o3-mini
            # gpt-4o
            # o1
            # claude-3-5-haiku-latest (small and fast)