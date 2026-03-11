import openai
import anthropic
import os
import requests
from halo import Halo
import json
from pathlib import Path
try:
    from google import genai as google_genai_client
    from google.genai import types as google_genai_types
except ImportError:
    google_genai_client = None
    google_genai_types = None

try:
    import google.generativeai as legacy_genai
except ImportError:
    legacy_genai = None
import datetime
import re
import tiktoken
from openai.types.responses import WebSearchToolParam

# ANSI escape codes for some colors
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
RESET = "\033[0m"  # Resets the color to default

XAI_BASE_URL = "https://api.x.ai/v1"

RESPONSE_API_MODELS = {
    "o3-pro": {
        "max_tokens": 100_000,          # output ceiling
        "context_window": 200_000,      # input ceiling
        "supports_reasoning": True,
    },
    "o3-deep-research": {    # PRICEY
        "max_tokens": 100_000,
        "context_window": 300_000,
        "supports_reasoning": True,
        "is_deep_research": True
    },
    "gpt-5-chat-latest": {
        "max_tokens": 128_000,
        "context_window": 400_000,
        "supports_reasoning": True,
    },
    "gpt-5-nano": {
        "max_tokens": 128_000,         # max output tokens
        "context_window": 400_000,     # input ceiling (advertised)
        "supports_reasoning": True,
    },
    "gpt-5": {
        "max_tokens": 128000,
        "context_window": 400000,
        "supports_reasoning": True,
    },
    "gpt-5.1": {
        "max_tokens": 128000,
        "context_window": 400000,
        "supports_reasoning": True,
    },
    "gpt-5.2": {
        "max_tokens": 128000,
        "context_window": 400_000,
        "supports_reasoning": True,
    },
    "gpt-5.2-pro": {
        "max_tokens": 128000,
        "context_window": 400000,
        "supports_reasoning": True,
    },
    "gpt-5.4": {
        "max_tokens": 128_000,
        "context_window": 1_000_000,   
        "supports_reasoning": True,
    },
    "gpt-5.4-pro": {
        "max_tokens": 128000,
        "context_window": 1_000_000,
        "supports_reasoning": True,
    },
    "gpt-5-mini": {
        "max_tokens": 128000,
        "context_window": 400000,
        "supports_reasoning": True,
    },
    "gpt-5-pro": {   # PRICEY
        "max_tokens": 272_000,
        "context_window": 400_000,
        "supports_reasoning": True,
    },
    "gpt-4.1": {
        "max_tokens": 32_768,
        "context_window": 1_047_576,
        "supports_reasoning": False,
    }
}

# Model configurations
MODEL_CONFIGS = {
    # OpenAI models
    "gpt-4o": {
        "max_tokens": 16384,
        "supports_reasoning": False,
    },
    "o3-mini": {
        "max_tokens": 100000,
        "context_window": 199000,
        "supports_reasoning": True,
    },
    "o4-mini": {
        "max_tokens": 100000,
        "context_window": 199000,
        "supports_reasoning": True,
    },
    "o3": {
        "max_tokens": 100000,
        "context_window": 199000,
        "supports_reasoning": True,
    },
    "o1": {
        "max_tokens": 100000,
        "context_window": 199000,
        "supports_reasoning": True,
    },
    "claude-3-7-sonnet-latest": {
        "max_tokens": 30720,
        "thinking_enabled": True,
        "thinking_budget": 32000,
        "max_tokens_with_thinking": 128000,
        "beta_flags": ["output-128k-2025-02-19"]
    },
    "claude-sonnet-4-0": {
                        "context_window": 1_000_000,  # 1M with beta flag
                        "max_tokens": 64_000,
                        "thinking_enabled": True,
                        "thinking_budget": 30_000,  # safe default
                        "max_tokens_with_thinking": 64_000,
                        "beta_flags": ["context-1m-2025-08-07"]},
    "claude-opus-4-0": {  # Opus is sadly half as useful as sonnet in terms of tokens, despite being smarter :(
                        "context_window":200_000,
                        "max_tokens": 32_000,
                        "thinking_enabled": True,
                        "thinking_budget": 15_000,
                        "max_tokens_with_thinking": 32_000},
    "claude-opus-4-1": {  # Opus is sadly half as useful as sonnet in terms of tokens, despite being smarter :(
        "context_window": 200_000,
        "max_tokens": 32_000,
        "thinking_enabled": True,
        "thinking_budget": 15_000,
        "max_tokens_with_thinking": 32_000},
    "claude-opus-4-5": {  # Opus is sadly half as useful as sonnet in terms of tokens, despite being smarter :(
        "context_window": 200_000,
        "max_tokens": 32_000,
        "thinking_enabled": True,
        "thinking_budget": 15_000,
        "max_tokens_with_thinking": 32_000},
    "claude-opus-4-6": {  # Opus is sadly half as useful as sonnet in terms of tokens, despite being smarter :(
        "context_window": 200_000,
        "max_tokens": 32_000,
        "thinking_enabled": True,
        "thinking_budget": 15_000,
        "max_tokens_with_thinking": 32_000,
        "beta_flags": ["context-1m-2025-08-07"]},
    "claude-sonnet-4-6": {
        "context_window": 1_000_000,  # 1M with beta flag
        "max_tokens": 64_000,
        "thinking_enabled": True,
        "thinking_budget": 32_000,
        "max_tokens_with_thinking": 64_000,
        "beta_flags": ["context-1m-2025-08-07"]},
    "claude-sonnet-4-5": {
        "context_window": 1_000_000,  # 1M with beta flag
        "max_tokens": 64_000,
        "thinking_enabled": True,
        "thinking_budget": 32_000,
        "max_tokens_with_thinking": 64_000,
        "beta_flags": ["context-1m-2025-08-07"]},
    "gemini-2.5-pro-exp-03-25": {
        "max_tokens": 65636,
        "supports_web_search": True,
    },
    "gemini-2.5-pro": {
        "max_tokens": 65636,
        "supports_web_search": True,
    },
    "gemini-2.5-pro-preview-06-05":{
        "max_tokens": 65636,  # Output tokens
        "supports_web_search": True,
    },
    "gemini-2.5-pro-preview-03-25": {
        "max_tokens": 65636,
        "supports_web_search": True,
    },
    "gemini-2.5-pro-preview-05-06":{
        "max_tokens": 65636,
        "supports_web_search": True,
    },
    "gemini-3-pro-preview": {
        "max_tokens": 65636,
        "supports_web_search": True,
    },
    "gemini-3.1-pro-preview": {
        "max_tokens": 65636,
        "supports_web_search": True,
    },
    "gemini-3-flash-preview": {
        "max_tokens": 65636,
        "supports_web_search": True,
    },
    "gemini-3.1-flash-lite-preview": {
        "max_tokens": 65636,
        "supports_web_search": True,
    },
    "gemini-2.5-flash": {
        "max_tokens": 65636,
        "supports_web_search": True,
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
    # xAI Grok models (OpenAI-compatible endpoint at XAI_BASE_URL)
    "grok-4-latest": {
        "max_tokens": 16_000,
        "context_window": 256_000,
        "supports_reasoning": False,
        "supports_web_search": True,
        "xai_live_search": False,
    },
    "grok-4-1-fast-reasoning": {
        "max_tokens": 16_000,
        "context_window": 2_000_000,
        "supports_reasoning": True,
        "supports_web_search": True,
        "xai_live_search": False,
    },
    "grok-4-1-fast-non-reasoning": {
        "max_tokens": 16_000,
        "context_window": 2_000_000,
        "supports_reasoning": False,
        "supports_web_search": True,
        "xai_live_search": False,
    },
    "grok-4-fast-reasoning": {
        "max_tokens": 16_000,
        "context_window": 2_000_000,
        "supports_reasoning": False,
        "supports_web_search": True,
        "xai_live_search": False,
    },
    "grok-4-fast-non-reasoning": {
        "max_tokens": 16_000,
        "context_window": 2_000_000,
        "supports_reasoning": False,
        "supports_web_search": True,
        "xai_live_search": False,
    },
    "grok-code-fast-1": {
        "max_tokens": 16_000,
        "context_window": 256_000,
        "supports_reasoning": False,
        "supports_web_search": True,
        "xai_live_search": False,
    },
    "llama3.1": {
        "max_tokens": 4096,
    },
    "gemma3": {
        "max_tokens": 4096,
    },
}

MODEL_CONFIGS.update(RESPONSE_API_MODELS)

# Default configuration to use when model isn't found
DEFAULT_CONFIG = {
    "max_tokens": 4096,
    "supports_reasoning": False,
    "thinking_enabled": False,
}

# Allow extra time for slower-thinking models like GPT-5-Pro before timing out.
OPENAI_REQUEST_TIMEOUT = 900  # seconds


def get_model_config(model_name):
    """Get the configuration for a specific model, with fallback to defaults"""
    return MODEL_CONFIGS.get(model_name, DEFAULT_CONFIG)

# helpers.py
def _uses_responses_api(model: str) -> bool:
    cfg = get_model_config(model)
    return model in RESPONSE_API_MODELS or cfg.get("is_deep_research", False)


def get_available_models():
    """Get all available models organized by provider"""
    models_by_provider = {
        "OpenAI": [],
        "Claude": [],
        "Gemini": [],
        "Grok": [],
        "Ollama": []
    }
    
    for model_name in MODEL_CONFIGS.keys():
        config = MODEL_CONFIGS[model_name]
        model_info = {
            "name": model_name,
            "max_tokens": config.get("max_tokens", 4096),
            "context_window": config.get("context_window", "N/A"),
            "supports_reasoning": config.get("supports_reasoning", False),
            "thinking_enabled": config.get("thinking_enabled", False),
            "thinking_budget": config.get("thinking_budget", "N/A")
        }
        
        if model_name.startswith(("gpt-", "o3-", "o4-", "o1")) or model_name == "o3":
            models_by_provider["OpenAI"].append(model_info)
        elif model_name.startswith("claude-"):
            models_by_provider["Claude"].append(model_info)
        elif model_name.startswith("gemini-"):
            models_by_provider["Gemini"].append(model_info)
        elif model_name.startswith("grok-"):
            models_by_provider["Grok"].append(model_info)
        else:
            models_by_provider["Ollama"].append(model_info)
    
    return models_by_provider


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
        encoding = tiktoken.get_encoding("cl100k_base")

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

    def save_response(self, prompt, response, model, reasoning_effort=None):
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
        if reasoning_effort is not None:
            filename = f"{timestamp}.{model_clean}.{reasoning_effort}.{prompt_part}.md"
        else:
            filename = f"{timestamp}.{model_clean}.{prompt_part}.md"

        # Full path to file
        file_path = os.path.join(self.output_dir, filename)

        # Write response to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"# Prompt: {prompt}\n\n")
            model_line = f"## Model: {model} ({reasoning_effort})" if reasoning_effort is not None else f"## Model: {model}"
            f.write(model_line + "\n\n")
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

    def ask_with_thinking(self, prompt, model=None, max_tokens=None, thinking_budget=None, enable_web_search=True,
                          max_searches=5):
        """
        Send a message to Claude with thinking enabled, stream the response, and update conversation history.

        Args:
            prompt (str): The prompt to send to Claude
            model (str, optional): Model to use. Defaults to the instance's model.
            max_tokens (int, optional): Maximum tokens in the response. Defaults to model's config.
            thinking_budget (int, optional): Budget for thinking. Defaults to model's config.
            enable_web_search (bool, optional): Enable web search tool. Defaults to True.
            max_searches (int, optional): Maximum number of searches allowed. Defaults to 5.

        Returns:
            dict: The complete response from Claude
        """
        if model is None:
            model = self.model

        config = get_model_config(model)

        if max_tokens is None:
            if config.get("thinking_enabled", False):
                max_tokens = config.get("max_tokens_with_thinking", 64_000)
            else:
                max_tokens = config.get("max_tokens", 32_000)

        if thinking_budget is None:
            thinking_budget = config.get("thinking_budget", 30_000)

        output_ceiling = config.get("max_tokens", 64_000)
        max_tokens = min(max_tokens, output_ceiling)

        if config.get("thinking_enabled") and thinking_budget is not None:
            # Claude requires max_tokens to exceed thinking budget; clamp if needed.
            if max_tokens is not None and thinking_budget >= max_tokens:
                original_budget = thinking_budget
                adjusted_budget = max(max_tokens - 1, 1)
                if adjusted_budget < original_budget:
                    print_colored(
                        f"Adjusted thinking budget to {adjusted_budget} to satisfy max_tokens>{original_budget} requirement.",
                        YELLOW,
                    )
                thinking_budget = adjusted_budget

        messages = self.conversation_history + [
            {"role": "user", "content": prompt}
        ]

        print(f"MAX TOKENS:{max_tokens}")

        # Track the current block type
        current_block = None
        thinking_started = False
        response_started = False
        full_response = ""
        search_count = 0
        current_tool_payload = ""

        # Build the parameters
        thinking_params = {}
        if config.get("thinking_enabled"):
            thinking_params = {
                "thinking": {"type": "enabled",
                             "budget_tokens": thinking_budget}
            }
            if config.get("beta_flags"):
                # Ensure beta_flags is a list
                beta_flags = config["beta_flags"]
                if isinstance(beta_flags, str):
                    beta_flags = [beta_flags]
                thinking_params["betas"] = beta_flags

        # ADD WEB SEARCH TOOL HERE
        tools = []
        if enable_web_search:
            tools.append({
                "type": "web_search_20250305",  # This is the tool type identifier
                "name": "web_search",
                "max_uses": max_searches,
                # Optional: add user location for localized results
                # "user_location": {
                #     "type": "approximate",
                #     "city": "Sterling",
                #     "region": "Virginia",
                #     "country": "US"
                # }
            })
            thinking_params["tools"] = tools

        with self.client.beta.messages.stream(
                model=model,
                max_tokens=max_tokens,
                messages=messages,
                **thinking_params
        ) as stream:
            for event in stream:
                if event.type == "content_block_start":
                    current_block = event.content_block.type

                    if current_block == "thinking" and not thinking_started:
                        print("<thinking>")
                        thinking_started = True
                    # NEW: Handle web search tool use
                    elif current_block == "server_tool_use":
                        search_count += 1
                        print_colored(f"\n[Web Search #{search_count}]", CYAN)
                        current_tool_payload = ""

                elif event.type == "content_block_delta":
                    if event.delta.type == "thinking_delta":
                        print_colored(event.delta.thinking, CYAN)

                    elif event.delta.type == "text_delta":
                        if thinking_started and not response_started:
                            print("</thinking>\n")
                            response_started = True

                        print_colored(event.delta.text, self.color)
                        full_response += event.delta.text

                    # NEW: Handle search query streaming
                    elif event.delta.type in ("server_tool_use_delta", "input_json_delta"):
                        partial = getattr(event.delta, "partial_json", None) or getattr(event.delta, "input", None)
                        if partial:
                            current_tool_payload += partial

                elif event.type == "content_block_stop":
                    previous_block = current_block
                    if current_block == "thinking" and not response_started:
                        print("</thinking>\n")
                    current_block = None
                    if previous_block == "server_tool_use" and current_tool_payload:
                        try:
                            payload_data = json.loads(current_tool_payload)
                        except json.JSONDecodeError:
                            payload_data = {"query": current_tool_payload}
                        query_text = (
                            payload_data.get("query")
                            if isinstance(payload_data, dict)
                            else payload_data
                        )
                        if not query_text and isinstance(payload_data, dict):
                            query_text = payload_data.get("input")
                        if query_text:
                            print_colored(f" Query: {query_text}\n", YELLOW)
                        current_tool_payload = ""

                # NEW: Handle search results
                elif event.type == "web_search_tool_result":
                    results = getattr(event, "content", None) or getattr(event, "results", None) or []
                    results_list = list(results)
                    print_colored(f"\n[Search completed - {len(results_list)} results retrieved]\n", GREEN)
                    for i, result in enumerate(results_list, 1):
                        url = getattr(result, "url", None) or getattr(result, "uri", None)
                        title = getattr(result, "title", None) or url
                        snippet = getattr(result, "snippet", None) or getattr(result, "text", None)
                        if title:
                            print_colored(f"  {i}. {title}\n", GREEN)
                        if url:
                            print_colored(f"     {url}\n", GREEN)
                        if snippet:
                            print_colored(f"     {snippet}\n", GREEN)

                elif event.type == "message_delta":
                    pass

                elif event.type == "message_stop":
                    pass

        self.conversation_history.append({"role": "user", "content": prompt})
        self.conversation_history.append({"role": "assistant", "content": full_response})

        print(RESET)
        print()

        if search_count > 0:
            print_colored(f"\n💰 Cost: ~${(search_count * 0.01):.2f} for {search_count} searches (plus token costs)",
                          MAGENTA)

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
    """
    Conversation wrapper that ¹ never overflows the model context window and ²
    automatically chains requests when `finish_reason == "length"`.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "o3-mini",
        reasoning_effort: str | None = "medium",
        color: str | None = None,
        max_continue_loops: int = 5,          # ← safety valve
        base_url: str | None = None,
    ):
        if api_key:
            client_kwargs = {"api_key": api_key, "timeout": OPENAI_REQUEST_TIMEOUT}
            if base_url:
                client_kwargs["base_url"] = base_url
            self.client = openai.OpenAI(**client_kwargs)
        else:
            self.client = None
            print("ERROR: No OpenAI API key provided.")
        self.conversation_history: list[dict] = []
        self.model = model
        self.color = color
        self.reasoning_effort = reasoning_effort
        self.max_continue_loops = max_continue_loops
        self.base_url = base_url

    # --------------------------------------------------------------------- #
    def ask(self, prompt: str, model: str | None = None, max_completion_tokens: int | None = None) -> str:
        """
        Send `prompt` to the model, streaming the response, trimming history if needed,
        and automatically issuing “continue” follow-ups when the model hits the length cap.
        """
        if self.client is None:
            raise ValueError("OpenAI client not initialised.")

        model_to_use = model or self.model  # Use a different variable name to avoid confusion with the 'model' module

        # NEW: transparently route Requests
        if _uses_responses_api(model_to_use):
            return self._ask_via_responses(prompt,
                                           model=model_to_use,
                                           max_output_tokens=max_completion_tokens)

        cfg = get_model_config(model_to_use)
        ctx_limit = cfg.get("context_window", cfg.get("max_tokens", 4096))  # Use context_window if available, fallback to max_tokens
        max_completion_tokens_from_config = cfg.get("max_tokens", ctx_limit // 4)

        # Start with reasonable default for completion tokens
        if max_completion_tokens is None:
            effective_max_tokens = min(max_completion_tokens_from_config, ctx_limit // 4)
        else:
            effective_max_tokens = max_completion_tokens

        full_answer = ""
        follow_up_prompt = "Please continue."
        continue_loops = 0
        role_user_prompt = prompt
        cumulative_output_tokens = 0  # For debug tracking
        web_search_invocations = 0
        tool_arg_fragments: dict[str, str] = {}

        spinner = Halo(  # Add a spinner for OpenAI as well
            text=f'Waiting for response from {model_to_use}...',
            spinner='dots',
            color='yellow'
        )
        first_chunk_received = False

        while True:
            # ---------- build message list that fits -------------------- #
            hist = _shrink_history_to_fit(
                self.conversation_history.copy(),
                role_user_prompt,
                model_to_use,  # Pass the correct model name
                ctx_limit,
                effective_max_tokens,  # Use the calculated max tokens for completion
            )
            messages = hist + [{"role": "user", "content": role_user_prompt}]
            
            # Final validation of token count
            total_input_tokens = _count_tokens_messages(messages, model_to_use)
            available_for_completion = ctx_limit - total_input_tokens
            
            # Check for critical context issues
            if available_for_completion <= 0:
                raise ValueError(f"🚨 ERROR: Input ({total_input_tokens:,} tokens) exceeds context window ({ctx_limit:,} tokens). Please reduce input size.")
            elif available_for_completion < 1000:
                raise ValueError(f"🚨 ERROR: Not enough context for completion. Input: {total_input_tokens:,} tokens, Available: {available_for_completion} tokens (< 1000 tokens)")
            
            # Warn about low available context
            if available_for_completion < 10000:
                print_colored(f"⚠️  WARNING: Very limited context available! Input: {total_input_tokens:,} tokens, Available: {available_for_completion:,} tokens (< 10k)", RED)
            
            # Adjust completion tokens if they exceed available context    
            if effective_max_tokens > available_for_completion:
                safety_buffer = max(2000, available_for_completion // 50)  # 2% buffer, min 2000 tokens
                new_max_tokens = available_for_completion - safety_buffer
                print_colored(f"⚠️  WARNING: Requested completion tokens ({effective_max_tokens:,}) > available context ({available_for_completion:,}). Reducing to {new_max_tokens:,}.", RED)
                effective_max_tokens = max(500, new_max_tokens)  # Ensure minimum viable completion size

            enable_xai_web = (
                cfg.get("supports_web_search", False)
                and cfg.get("xai_live_search", False)
                and self.base_url
                and "x.ai" in self.base_url
            )

            params = {
                "model": model_to_use,
                "messages": messages,
                "max_completion_tokens": effective_max_tokens,  # API param is 'max_tokens' for output limit
                "stream": True  # <--- ENABLE STREAMING HERE
            }
            if cfg.get("supports_reasoning") and self.reasoning_effort:
                params["reasoning_effort"] = self.reasoning_effort
            if enable_xai_web:
                # Enable xAI's live search injection (no client-side tools).
                params["extra_body"] = {
                    "search_parameters": {
                        "mode": "auto"
                    }
                }

            if not first_chunk_received:  # Start spinner only for the initial part of a potentially long response
                spinner.start()

            print(f"Executing:{model_to_use} call with Max_Completion_Tokens:{max_completion_tokens} Streaming:{True} Supports_Reasoning:{self.reasoning_effort} Reasoning_Effort:{self.reasoning_effort}")
            stream_response_content = ""
            current_finish_reason = None
            api_usage_stats = None  # To store usage data from the stream if available

            try:
                response_stream = self.client.chat.completions.create(**params)
                for chunk in response_stream:
                    if not first_chunk_received:
                        spinner.stop()
                        first_chunk_received = True
                        if self.color:  # Start color if specified
                            print(self.color, end="", flush=True)

                    if chunk.choices:
                        delta = chunk.choices[0].delta
                        content_piece = delta.content

                        if content_piece:
                            print_colored(content_piece, self.color if self.color else RESET)  # Use RESET if no color
                            stream_response_content += content_piece

                        tool_calls = getattr(delta, "tool_calls", None)
                        if tool_calls:
                            for tc in tool_calls:
                                tc_type = getattr(tc, "type", None)
                                func = getattr(tc, "function", None)
                                func_name = getattr(func, "name", None) if func else None
                                is_search_call = tc_type in ("web_search", "live_search") or func_name in (
                                    "web_search",
                                    "live_search",
                                )
                                if is_search_call:
                                    # Each tool call streams its arguments; accumulate to extract the query.
                                    call_id = getattr(tc, "id", None) or f"web_search_{web_search_invocations + 1}"
                                    if call_id not in tool_arg_fragments:
                                        web_search_invocations += 1
                                        tool_arg_fragments[call_id] = ""
                                        print_colored(f"\n[Web Search #{web_search_invocations}]\n", CYAN)
                                    arg_fragment = getattr(func, "arguments", None) if func else None
                                    if arg_fragment:
                                        tool_arg_fragments[call_id] += arg_fragment
                                        query_text = None
                                        try:
                                            parsed = json.loads(tool_arg_fragments[call_id])
                                            if isinstance(parsed, dict):
                                                query_text = (
                                                    parsed.get("query")
                                                    or parsed.get("q")
                                                    or parsed.get("search")
                                                )
                                        except json.JSONDecodeError:
                                            query_text = tool_arg_fragments[call_id]
                                        if query_text:
                                            print_colored(f" Query: {query_text}\n", YELLOW)

                        if chunk.choices[0].finish_reason:
                            current_finish_reason = chunk.choices[0].finish_reason

                    # OpenAI often sends usage stats in the last chunk with stream=True
                    # or in a separate event if using event streams more directly.
                    # For basic streaming, it might be on the chunk if finish_reason is set.
                    if hasattr(chunk, 'usage') and chunk.usage:
                        api_usage_stats = chunk.usage


            except Exception as e:
                spinner.stop()  # Ensure spinner stops on error
                print_colored(f"\nError during OpenAI API call: {e}\n", RED)
                return f"Error: {e}"  # Or handle more gracefully

            if not first_chunk_received:  # If loop exited before receiving anything (e.g. error before stream)
                spinner.stop()

            full_answer += stream_response_content

            # ---- DEBUG: token counting (adjust for streaming) ---- #
            part_tokens = count_tokens(stream_response_content, model_to_use)  # Tokenize the streamed part
            cumulative_output_tokens += part_tokens

            # Try to get completion tokens from usage if available, otherwise estimate
            api_comp_tok = api_usage_stats.completion_tokens if api_usage_stats and hasattr(api_usage_stats,
                                                                                            'completion_tokens') else part_tokens

            debug_msg = (
                f"\n[DEBUG]"
                f" reason={current_finish_reason}"
                f" | api_completion_tokens={api_comp_tok}"  # This might be per segment if continued
                f" | part_tokens_estimated={part_tokens}"
                f" | cumulative_estimated={cumulative_output_tokens}"
                f" | ctx_limit={ctx_limit}"
            )
            print_colored(debug_msg + "\n", CYAN)
            # --------------------------------------------------------- #

            self.conversation_history.append({"role": "user", "content": role_user_prompt})
            self.conversation_history.append(
                {"role": "assistant", "content": stream_response_content})  # Save the streamed part

            if current_finish_reason != "length":
                break

            continue_loops += 1
            if continue_loops >= self.max_continue_loops:
                print_colored("\n[Stopped after max_continue_loops]\n", RED)
                break
            role_user_prompt = follow_up_prompt
            first_chunk_received = False  # Reset for the next part of the conversation if "continue"

        if web_search_invocations > 0:
            print_colored(f"\n[Web search: {web_search_invocations} request(s) dispatched]\n", CYAN)

        print(RESET)  # Reset color at the very end
        print()
        response_saver.save_response(prompt, full_answer, model_to_use, self.reasoning_effort)
        return full_answer
    # --------------------------------------------------------------------- #

    def reset_conversation(self):
        """Clear the conversation history"""
        self.conversation_history = []
        print("Conversation history has been reset.")

    def get_conversation_history(self):
        """Return the current conversation history"""
        return self.conversation_history

    # ─────────────────── NEW STREAMING IMPLEMENTATION ────────────────────
    def _ask_via_responses(self, prompt: str, *, model: str, max_output_tokens: int | None = None) -> str:
        cfg = get_model_config(model)
        ctx_limit = cfg.get("context_window", 200_000)
        max_output_default = cfg.get("max_tokens", 100_000)
        max_output = min(max_output_tokens or max_output_default, max_output_default)

        # Build messages and trim history to fit context + output.
        # Always preserve the current user prompt.
        new_user_message = {"role": "user", "content": prompt}
        messages = self.conversation_history + [new_user_message]
        needed = _count_tokens_messages(messages, model) + max_output
        if needed > ctx_limit:
            trimmed_history = _shrink_history_to_fit(
                self.conversation_history.copy(),
                prompt,
                model,
                ctx_limit,
                max_output,
            )
            messages = trimmed_history + [new_user_message]
            needed = _count_tokens_messages(messages, model) + max_output
            if needed > ctx_limit:
                input_tokens = _count_tokens_messages(messages, model)
                prompt_tokens = count_tokens(prompt, model)
                err_msg = (
                    f"🚨 ERROR: Context overflow for {model}. "
                    f"Prompt={prompt_tokens:,} tokens, "
                    f"Input(after trim)={input_tokens:,}, "
                    f"Requested output={max_output:,}, "
                    f"Context window={ctx_limit:,}. "
                    "Reduce prompt/history or lower max_output_tokens."
                )
                print_colored(err_msg + "\n", RED)
                return f"Error: {err_msg}"

        # Responses API accepts input as a list of {role, content}
        def _to_responses_input(msgs: list[dict]) -> list[dict]:
            out = []
            for m in msgs:
                content = m.get("content", "")
                if not isinstance(content, str):
                    content = str(content)
                out.append({"role": m.get("role", "user"), "content": content})
            return out

        responses_input = _to_responses_input(messages)
        if not responses_input:
            err_msg = "🚨 ERROR: No input messages available for /responses request."
            print_colored(err_msg + "\n", RED)
            return f"Error: {err_msg}"

        params = {
            "model": model,
            "input": responses_input,
            "max_output_tokens": max_output,
        }
        if cfg.get("supports_reasoning") and self.reasoning_effort:
            params["reasoning"] = {"effort": self.reasoning_effort}

        uses_gpt5_tools = "gpt-5" in model
        attach_web_tool = False
        if uses_gpt5_tools:
            effort_level = (self.reasoning_effort or "").lower()
            if effort_level == "minimal":
                print_colored(
                    "Skipping web search tool because reasoning_effort 'minimal' disallows tool use.",
                    YELLOW,
                )
            else:
                attach_web_tool = True

        if attach_web_tool:
            search_tool = WebSearchToolParam(
                type="web_search",
                user_location={
                    "type": "approximate",
                    "country": "US",
                },
            )
            params["tools"] = [search_tool]
            params["tool_choice"] = "auto"

        spinner = Halo(text=f"Waiting for {model} via /responses …", spinner="dots", color="yellow")
        spinner.start()

        full_text = ""
        stop_reason = None
        search_sources: list[str] = []
        try:
            # Correct streaming API usage
            with self.client.responses.stream(**params) as stream:
                first_chunk = False
                for event in stream:
                    # Stream the text deltas
                    if event.type == "response.output_text.delta":
                        if not first_chunk:
                            spinner.stop()
                            first_chunk = True
                        chunk = event.delta or ""
                        print_colored(chunk, self.color or RESET)
                        full_text += chunk

                    # Optional: surface errors
                    elif event.type == "response.error":
                        spinner.stop()
                        err = getattr(event, "error", "Unknown responses error")
                        print_colored(f"\n/responses error: {err}\n", RED)
                        return f"Error: {err}"
                    elif attach_web_tool and event.type == "response.tool_call.delta":
                        # Tool calls stream metadata; collect without interrupting text flow.
                        delta = getattr(event, "delta", None)
                        if delta and isinstance(delta, dict):
                            action = delta.get("action")
                            if action and isinstance(action, dict):
                                sources = action.get("sources")
                                if sources and isinstance(sources, list):
                                    search_sources.extend(str(src) for src in sources)

                # Final response contains metadata (usage, stop_reason, etc.)
                final = stream.get_final_response()
                stop_reason = getattr(final, "stop_reason", None)
                if attach_web_tool:
                    for item in getattr(final, "output", []):
                        if getattr(item, "type", None) == "web_search_call":
                            action = getattr(item, "action", None)
                            sources = getattr(action, "sources", None)
                            if sources and isinstance(sources, list):
                                search_sources.extend(str(src) for src in sources)

        except Exception as e:
            spinner.stop()
            print_colored(f"\n/responses error: {e}\n", RED)
            return f"Error: {e}"

        # Update history and save to file
        self.conversation_history += [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": full_text},
        ]

        print(RESET)
        print()
        if attach_web_tool and search_sources:
            deduped = []
            seen = set()
            for src in search_sources:
                if src not in seen:
                    deduped.append(src)
                    seen.add(src)
            print_colored("Search sources:", CYAN)
            for src in deduped:
                print_colored(f"- {src}", CYAN)
            print()
        response_saver.save_response(prompt, full_text, model, self.reasoning_effort)
        return full_text


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
        self.color = color
        self.model = model
        self.conversation_history = []
        self.using_new_sdk = False
        self.client = None
        self.genai_types = None
        self.legacy_genai = None
        self.model_instance = None
        self.chat_session = None

        try:
            if google_genai_client and google_genai_types:
                # New google-genai SDK (supports tools like Google Search)
                self.using_new_sdk = True
                self.client = google_genai_client.Client(api_key=api_key)
                self.genai_types = google_genai_types
            elif legacy_genai:
                # Fallback to legacy google-generativeai SDK
                self.using_new_sdk = False
                self.legacy_genai = legacy_genai
                self.legacy_genai.configure(api_key=api_key)
                self.model_instance = self.legacy_genai.GenerativeModel(model)
                self.chat_session = self.model_instance.start_chat(history=[])
            else:
                raise ImportError(
                    "google-genai (or legacy google-generativeai) package not installed. "
                    "Install with 'pip install google-genai' or 'pip install google-generativeai'."
                )
        except Exception as e:
            print_colored(f"Error initializing Gemini: {str(e)}", RED)
            raise

    def ask(self, prompt, model=None, max_tokens=None):
        """
        Dispatch Gemini requests to the appropriate SDK implementation (new google-genai or legacy).
        """
        target_model = model or self.model
        if target_model != self.model:
            self.model = target_model
            # Switching models resets conversation state to avoid mixing contexts
            self.conversation_history = []
            if not self.using_new_sdk:
                self.model_instance = self.legacy_genai.GenerativeModel(target_model)
                self.chat_session = self.model_instance.start_chat(history=[])

        config = get_model_config(target_model)
        if max_tokens is None:
            max_tokens = config.get("max_tokens", 8192)

        print(f"MAX TOKENS:{max_tokens}")

        if self.using_new_sdk:
            return self._ask_with_new_sdk(prompt, target_model, max_tokens, config)

        return self._ask_with_legacy_sdk(prompt, target_model, max_tokens)

    def _supports_google_search(self, model_name: str, config: dict) -> bool:
        """
        Determine whether the current Gemini model should invoke the Google Search grounding tool.
        """
        return bool(config.get("supports_web_search", False) and google_genai_types)

    def _build_contents_for_new_sdk(self, history: list[dict]) -> list:
        """
        Convert internal conversation history into google-genai Content objects.
        """
        contents = []
        if not self.genai_types:
            return contents

        for message in history:
            text = message.get("content", "")
            if not text:
                continue
            role = message.get("role", "user")
            # google-genai expects 'model' instead of 'assistant'
            sdk_role = "model" if role == "assistant" else "user"
            try:
                part = self.genai_types.Part.from_text(text=text)
            except (AttributeError, TypeError):
                part = self.genai_types.Part(text=text)
            contents.append(
                self.genai_types.Content(
                    role=sdk_role,
                    parts=[part]
                )
            )
        return contents

    def _ask_with_new_sdk(self, prompt: str, model: str, max_tokens: int, config: dict) -> str:
        """
        Handle Gemini requests via the new google-genai SDK with optional Google Search grounding.
        """
        history = self.conversation_history + [{"role": "user", "content": prompt}]
        contents = self._build_contents_for_new_sdk(history)

        generate_kwargs = {"max_output_tokens": max_tokens}
        if self._supports_google_search(model, config):
            generate_kwargs["tools"] = [
                self.genai_types.Tool(
                    google_search=self.genai_types.GoogleSearch()
                )
            ]

        generate_config = self.genai_types.GenerateContentConfig(**generate_kwargs)

        spinner = Halo(
            text=f"Waiting for {model} via google-genai…",
            spinner="dots",
            color="yellow"
        )
        spinner.start()

        full_response = ""
        started_stream = False
        final_chunk = None
        resolved_response = None

        try:
            stream = self.client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=generate_config,
            )
            for chunk in stream:
                final_chunk = chunk
                text = getattr(chunk, "text", None)
                if text:
                    if not started_stream:
                        spinner.stop()
                        started_stream = True
                    print_colored(text, self.color or RESET)
                    full_response += text
            # Attempt to capture the resolved response for metadata/citations.
            resolved_response = getattr(stream, "response", None)
            if resolved_response is None:
                result_attr = getattr(stream, "result", None)
                if callable(result_attr):
                    try:
                        resolved_response = result_attr()
                    except Exception:
                        resolved_response = None
                elif result_attr is not None:
                    resolved_response = result_attr
        except Exception as e:
            spinner.stop()
            error_msg = f"Gemini API error: {str(e)}"
            print_colored(error_msg, RED)
            return error_msg
        finally:
            spinner.stop()

        if not full_response:
            full_response = "[No textual content returned by Gemini]"

        # Update history and surface grounding metadata
        self.conversation_history.append({"role": "user", "content": prompt})
        self.conversation_history.append({"role": "assistant", "content": full_response})

        print(RESET)
        print()

        if self._supports_google_search(model, config):
            metadata_source = resolved_response or final_chunk
            if metadata_source:
                self._render_grounding_metadata(metadata_source)

        response_saver.save_response(prompt, full_response, model)
        return full_response

    def _render_grounding_metadata(self, response_chunk) -> None:
        """
        Pretty-print Google Search grounding metadata (queries, sources, entry point) if available.
        """
        candidates = getattr(response_chunk, "candidates", None)
        if not candidates:
            return

        candidate = candidates[0]

        grounding_metadata = (
            getattr(candidate, "grounding_metadata", None)
            or getattr(candidate, "groundingMetadata", None)
        )
        if not grounding_metadata:
            return

        queries = getattr(grounding_metadata, "web_search_queries", None) or getattr(
            grounding_metadata, "webSearchQueries", None
        )
        if queries:
            print_colored("Search queries:\n", CYAN)
            for query in queries:
                query_text = getattr(query, "text", None) or getattr(query, "query", None) or str(query)
                if query_text:
                    print_colored(f"- {query_text}\n", CYAN)

        chunks = getattr(grounding_metadata, "grounding_chunks", None) or getattr(
            grounding_metadata, "groundingChunks", None
        )
        if chunks:
            print_colored("Search sources:\n", CYAN)
            seen = set()
            index = 1
            for chunk in chunks:
                web = getattr(chunk, "web", None)
                if not web:
                    continue
                uri = getattr(web, "uri", None) or getattr(web, "url", None)
                if not uri or uri in seen:
                    continue
                seen.add(uri)
                title = getattr(web, "title", None) or uri
                print_colored(f"{index}. {title} — {uri}\n", CYAN)
                index += 1

        entry_point = getattr(grounding_metadata, "search_entry_point", None) or getattr(
            grounding_metadata, "searchEntryPoint", None
        )
        if entry_point:
            rendered = getattr(entry_point, "rendered_content", None) or getattr(
                entry_point, "renderedContent", None
            )
            if rendered:
                print_colored("Search entry point (HTML):\n", CYAN)
                print(rendered)

    def _ask_with_legacy_sdk(self, prompt, model=None, max_tokens=None):
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
                self.model_instance = self.legacy_genai.GenerativeModel(model)
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

            # Helper: safely extract textual content from a chunk/response without triggering
            # the library's quick accessor exception when no valid Part exists.
            def _safe_extract_text(obj):
                # 1. Try .text but swallow the known "quick accessor" exception
                try:
                    t = getattr(obj, "text", None)
                    if t:  # non-empty string
                        return t
                except Exception:
                    pass  # fall through to manual extraction

                # 2. Look for candidates -> content -> parts
                # Streaming partials often expose .candidates with partial content
                try:
                    cands = getattr(obj, "candidates", None)
                    if cands:
                        out_fragments = []
                        for cand in cands:
                            content = getattr(cand, "content", None)
                            if not content:
                                continue
                            parts = getattr(content, "parts", None)
                            if parts:
                                for p in parts:
                                    # Newer SDK: each part may have a 'text' attribute; else str(part)
                                    txt = getattr(p, "text", None)
                                    if txt:
                                        out_fragments.append(txt)
                                    else:
                                        out_fragments.append(str(p))
                        if out_fragments:
                            return "".join(out_fragments)
                except Exception:
                    pass

                # 3. Direct parts attribute
                try:
                    parts = getattr(obj, "parts", None)
                    if parts:
                        return "".join(
                            getattr(p, "text", None) or str(p) for p in parts
                        )
                except Exception:
                    pass

                return ""  # Nothing extracted

            # Send the message and stream the response
            response_stream = self.chat_session.send_message(
                prompt,
                stream=True,
                generation_config={
                    "max_output_tokens": max_tokens
                }
            )

            full_response = ""
            had_any_text = False
            last_finish_reason = None
            for chunk in response_stream:
                # Some chunk objects may expose metadata (finish_reason) without text
                # We collect text defensively.
                text = _safe_extract_text(chunk)
                if text:
                    had_any_text = True
                    print_colored(text, self.color)
                    full_response += text
                # Try to capture finish reason if available (naming differs across SDK versions)
                try:
                    if hasattr(chunk, "candidates") and chunk.candidates:
                        fr = getattr(chunk.candidates[0], "finish_reason", None) or getattr(chunk.candidates[0], "finishReason", None)
                        if fr is not None:
                            last_finish_reason = fr
                except Exception:
                    pass

            # If no text surfaced during streaming but finish_reason indicates a normal stop,
            # attempt a non-stream fallback single-shot call (without streaming) to recover content.
            if not had_any_text:
                try:
                    fallback = self.chat_session.send_message(
                        prompt,
                        stream=False,
                        generation_config={
                            "max_output_tokens": max_tokens
                        }
                    )
                    recovered = _safe_extract_text(fallback)
                    if recovered:
                        full_response = recovered
                        had_any_text = True
                        print_colored(full_response, self.color)
                    else:
                        # Provide a diagnostic string so caller/logs show context
                        full_response = f"[No textual content returned by Gemini; finish_reason={last_finish_reason}]"
                except Exception as fe:
                    diag = f"[Gemini streaming produced no text and fallback failed: {fe}]"
                    print_colored(diag, RED)
                    full_response = diag

            # Add the response to conversation history for tracking
            self.conversation_history.append({"role": "assistant", "content": full_response})

            print(RESET)
            print()
            # Save response to file
            response_saver.save_response(prompt, full_response, model or self.model)

            return full_response
        except Exception as e:
            error_msg = f"Gemini API error: {str(e)}"
            print_colored(error_msg, RED)
            return error_msg

    def reset_conversation(self):
        """Clear the conversation history by starting a new chat session (legacy) or wiping state (new SDK)."""
        self.conversation_history = []
        if self.using_new_sdk:
            print("Conversation history has been reset.")
            return

        self.chat_session = self.model_instance.start_chat(history=[])
        print("Conversation history has been reset.")

    def get_conversation_history(self):
        """Return the current conversation history in a format compatible with other models"""
        return self.conversation_history

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

            print_colored(f"Loaded {filename} ({token_count} tokens)\n", GREEN)
            return prompt
    except Exception as e:
        print_colored(f"Error loading {filename}: {str(e)}\n", RED)
    return None


def run_openai_query(prompt, api_key=None, model="gpt-5.4", key_file="apikeys.json", reasoning_effort=None):
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
        print_colored(f"Note: {model} does not support reasoning_effort. This parameter will be ignored.\n", RED)
        reasoning_effort = None

    openai_chat = OpenAIConversation(api_key, model=model, color=YELLOW, reasoning_effort=reasoning_effort)
    return openai_chat.ask(prompt, max_completion_tokens=max_tokens)


def run_claude_query(prompt, api_key=None, model="claude-sonnet-4-6", key_file="apikeys.json"):
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
        print_colored(f"Note: {model} does not support thinking. Using standard API call.\n", RED)

    claude = ClaudeConversation(api_key, YELLOW)
    return claude.ask_with_thinking(prompt, model=model)


def run_gemini_query(prompt, api_key=None, model="gemini-2.5-flash", key_file="apikeys.json"):
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
        print_colored("Skipping Gemini (google-generativeai package not installed)\n", RED)
        return None
    except Exception as e:
        print_colored(f"Error initializing Gemini: {str(e)}\n", RED)
        return None


def run_grok_query(prompt, api_key=None, model="grok-4-1-fast-reasoning", key_file="apikeys.json"):
    """Run a query against xAI Grok models via the OpenAI-compatible API."""
    if not api_key:
        # Environment variables take precedence for convenience
        api_key = os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")
        if not api_key:
            key_manager = APIKeyManager(key_file)
            # Accept multiple common key labels
            api_key = (
                key_manager.get_key("xai")
                or key_manager.get_key("grok")
                or key_manager.get_key("xai/grok")
            )
        if not api_key:
            print_colored("Error: No xAI (Grok) API key found\n", RED)
            return

    config = get_model_config(model)
    max_tokens = config.get("max_tokens", 16_000)

    grok_chat = OpenAIConversation(
        api_key,
        model=model,
        color=GREEN,
        reasoning_effort=None,
        base_url=XAI_BASE_URL,
    )
    return grok_chat.ask(prompt, max_completion_tokens=max_tokens)


def run_ollama_query(prompt, model="llama3.1", system_prompt=None):
    """Run a query against Ollama models"""
    ollama = OllamaConversation(model=model, color=GREEN)
    return ollama.ask(prompt, system_prompt=system_prompt)

# ----  helpers -------------------------------------------------------------- #
def _count_tokens_messages(messages: list[dict], model_name: str) -> int:
    """Rough-and-ready token counter for a list of chat messages."""
    return sum(count_tokens(m["content"], model_name) for m in messages)

def _shrink_history_to_fit(
    history: list[dict],
    prompt: str,
    model_name: str,
    max_ctx: int,
    max_completion: int,
) -> list[dict]:
    """
    Trim the left-most (oldest) messages until
    tokens(history)+tokens(prompt)+max_completion <= max_ctx
    """
    system_and_new = [{"role": "user", "content": prompt}]
    while True:
        tokens_needed = (
            _count_tokens_messages(history + system_and_new, model_name)
            + max_completion
        )
        if tokens_needed <= max_ctx or not history:
            break
        # drop the oldest (pairs are stored user/assistant sequentially)
        history = history[2:] if len(history) >= 2 else history[1:]
    return history
# ---------------------------------------------------------------------------- #

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

    # Ask Grok (xAI)
    if key_manager.get_key("xai") or key_manager.get_key("grok") or os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY"):
        print_colored("\n=== GROK RESPONSE ===\n", GREEN)
        run_grok_query(q)

    # #Ask Ollama
    # print_colored("\n=== OLLAMA RESPONSE ===\n", GREEN)
    # run_ollama_query(q, system_prompt=system_prompt)


if __name__ == "__main__":
    main()
