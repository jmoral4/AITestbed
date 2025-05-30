# concurrent_ai_test.py  •  spawn separate windows for multiple AI models
import subprocess
import sys
import os
import argparse
import time

# ──────────────────────────────────────────────────────────────────── helpers

def check_windows_terminal() -> bool:
    """Return True if Windows Terminal is installed."""
    try:
        res = subprocess.run(['wt', '--version'],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             text=True,
                             creationflags=subprocess.CREATE_NO_WINDOW)
        return res.returncode == 0
    except FileNotFoundError:
        return False


def run_in_windows_terminal(title: str, cmd: str, cwd: str, first: bool = False):
    """Open *cmd* in a new Windows-Terminal tab (new window for the first tab)."""
    try:
        if first:
            subprocess.Popen(['wt', 'nt', '--title', title, '-d', cwd,
                              'cmd', '/c', cmd])
        else:
            subprocess.Popen(['wt', '-w', '0', 'nt', '--title', title, '-d', cwd,
                              'cmd', '/c', cmd])
            time.sleep(0.5)          # give the tab time to appear
    except Exception as e1:
        print(f"[WT] primary launch failed → {e1}")
        try:  # older syntax
            subprocess.Popen(['wt', 'new-tab', '--title', title, '-d', cwd,
                              '--', 'cmd', '/c', cmd])
        except Exception as e2:
            print(f"[WT] fallback failed → {e2}")
            run_in_cmd(title, cmd, cwd)


def run_in_cmd(title: str, cmd: str, cwd: str):
    subprocess.Popen(
        f'start "{title}" cmd /c "cd /D \"{cwd}\" && {cmd}"',
        shell=True)

# ──────────────────────────────────────────────────────────────────── CLI

def parse_openai_specs(tokens):
    """
    Yield (model_name, effort|None) tuples from a list of spec strings.
    Accepts comma-separated or space-separated tokens:  ["o3:high", "o4-mini:high"]
    """
    for token in tokens:
        for raw in token.split(','):
            raw = raw.strip()
            if not raw:
                continue
            if ':' in raw:
                name, effort = raw.split(':', 1)
                yield name, effort.lower()
            else:
                yield raw, None


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Run many AI models concurrently in separate Windows-Terminal tabs")

    p.add_argument('--prompt-file', required=True,
                   help='File containing the prompt')
    p.add_argument('--system-prompt-file',
                   help='System prompt file for Ollama (optional)')

    # ⇢ new
    p.add_argument('--openai-models', nargs='*',
                   help='Space/comma list of OpenAI model specs, '
                        'e.g.  o3:low o3:high o4-mini:high')

    # ⇢ legacy (kept for compatibility)
    p.add_argument('--openai-model', default='o3-mini',
                   help='Single OpenAI model (legacy)')
    p.add_argument('--reasoning-effort',
                   help='Reasoning effort for the legacy single model')

    # Other vendors (unchanged)
    p.add_argument('--claude-model', default='claude-3-7-sonnet-latest')
    p.add_argument('--ollama-model', default='llama3.1')
    p.add_argument('--gemini-model', default='gemini-2.0-flash')

    args = p.parse_args(argv)

    # ───────────────────────────── pre-flight

    if not os.path.exists(args.prompt_file):
        sys.exit(f"❌ Prompt file '{args.prompt_file}' not found.")

    if args.system_prompt_file and not os.path.exists(args.system_prompt_file):
        print(f"⚠️  System prompt file '{args.system_prompt_file}' not found; "
              "Ollama may ignore it.")

    script_dir   = os.path.dirname(os.path.abspath(__file__))
    runner_py    = os.path.join(script_dir, 'ai_runner.py')
    python_exe   = sys.executable
    prompt_path  = os.path.abspath(args.prompt_file)
    prompt_arg   = f'--prompt-file "{prompt_path}"'

    # ───────────────────────────── assemble OpenAI commands

    if args.openai_models:
        openai_specs = list(parse_openai_specs(args.openai_models))
    else:  # backwards compatible single-model path
        openai_specs = [(args.openai_model, args.reasoning_effort)]

    openai_cmds = []
    for model_name, effort in openai_specs:
        effort_arg = f'--reasoning-effort {effort}' if effort else ''
        cmd = (f'{python_exe} "{runner_py}" --model openai '
               f'--model-name {model_name} {prompt_arg} {effort_arg} --wait')
        openai_cmds.append((model_name, cmd))

    # ───────────────────────────── assemble other vendor commands

    claude_cmd = (f'{python_exe} "{runner_py}" --model claude '
                  f'--model-name {args.claude_model} {prompt_arg} --wait')

    gemini_cmd = (f'{python_exe} "{runner_py}" --model gemini '
                  f'--model-name {args.gemini_model} {prompt_arg} --wait')

    # ollama_cmd = (f'{python_exe} "{runner_py}" --model ollama '
    #               f'--model-name {args.ollama_model} {prompt_arg} '
    #               f'{("--system-prompt-file " + os.path.abspath(args.system_prompt_file)) if args.system_prompt_file else ""} '
    #               f'--wait')

    # ───────────────────────────── spawn windows

    print(f"🔹 Using prompt: {prompt_path}")
    first = True
    for mdl, cmd in openai_cmds:
        run_in_windows_terminal(f"OpenAI ({mdl})", cmd, script_dir, first)
        first = False

    run_in_windows_terminal(f"Claude ({args.claude_model})", claude_cmd, script_dir, first)
    run_in_windows_terminal(f"Gemini ({args.gemini_model})", gemini_cmd, script_dir)
    # Uncomment if you want Ollama as well
    # run_in_windows_terminal(f"Ollama ({args.ollama_model})", ollama_cmd, script_dir)

    print("🚀 All AI processes launched. Check each tab for output.")


if __name__ == "__main__":
    # Example call: three OpenAI variants + Claude + Gemini
    main([
        "--prompt-file", "prompt.txt",
        "--openai-models", "o3:high", "gpt-4.1",
        "--claude-model", "claude-3-7-sonnet-latest",
        "--gemini-model", "gemini-2.5-pro-preview-05-06"
    ])

            # Other Options
            # gemini-2.5-pro-exp-03-25
            # gemini-2.0-flash
            # gemini-2.0-flash-lite (fast and cheap)
            # o3-mini
            # gpt-4o
            # o1
            # claude-3-5-haiku-latest (small and fast)