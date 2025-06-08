#!/usr/bin/env python3
"""
AI Testbed GUI - A graphical interface for testing multiple AI models concurrently - Created agentically by Claude 4.0 using the aitestbed.py as a source.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import threading
import os
import sys
import subprocess
import json
import time
from datetime import datetime
from typing import Dict, List, Optional

# Import our existing modules
try:
    import aitestbed
    from aitestbed import get_available_models
    import prep_context
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure aitestbed.py and prep_context.py are in the same directory")
    sys.exit(1)


class AITestbedGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Testbed - Multi-Model Testing Interface")
        self.root.geometry("1400x900")
        
        # State variables
        self.context_content = ""
        self.context_timestamp = None
        self.selected_directory = ""
        self.model_conversations = {}
        self.model_threads = {}
        
        # Load available models from aitestbed.py
        self.available_models = get_available_models()
        self.model_info_labels = {}
        
        # Model configurations
        self.models_config = {
            'openai': {
                'enabled': tk.BooleanVar(),
                'models': [model['name'] for model in self.available_models['OpenAI']],
                'selected_model': tk.StringVar(value='o3-mini' if 'o3-mini' in [m['name'] for m in self.available_models['OpenAI']] else (self.available_models['OpenAI'][0]['name'] if self.available_models['OpenAI'] else 'gpt-4o')),
                'reasoning_effort': tk.StringVar(value='auto')
            },
            'claude': {
                'enabled': tk.BooleanVar(),
                'models': [model['name'] for model in self.available_models['Claude']],
                'selected_model': tk.StringVar(value='claude-3-7-sonnet-latest' if 'claude-3-7-sonnet-latest' in [m['name'] for m in self.available_models['Claude']] else (self.available_models['Claude'][0]['name'] if self.available_models['Claude'] else 'claude-3-5-sonnet-latest'))
            },
            'gemini': {
                'enabled': tk.BooleanVar(),
                'models': [model['name'] for model in self.available_models['Gemini']],
                'selected_model': tk.StringVar(value='gemini-2.0-flash' if 'gemini-2.0-flash' in [m['name'] for m in self.available_models['Gemini']] else (self.available_models['Gemini'][0]['name'] if self.available_models['Gemini'] else 'gemini-2.0-flash'))
            },
            'ollama': {
                'enabled': tk.BooleanVar(),
                'models': [model['name'] for model in self.available_models['Ollama']],
                'selected_model': tk.StringVar(value='llama3.1' if 'llama3.1' in [m['name'] for m in self.available_models['Ollama']] else (self.available_models['Ollama'][0]['name'] if self.available_models['Ollama'] else 'llama3.1'))
            }
        }
        
        self.setup_ui()
        self.api_key_manager = None
        self.check_api_keys()
    
    def setup_ui(self):
        """Setup the main UI layout"""
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure root grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Setup sections
        self.setup_context_section(main_frame, row=0)
        self.setup_model_selection(main_frame, row=1)
        self.setup_prompt_section(main_frame, row=2)
        self.setup_control_section(main_frame, row=3)
        self.setup_output_section(main_frame, row=4)
        
        # Configure main frame grid weights
        main_frame.rowconfigure(4, weight=1)
    
    def setup_context_section(self, parent, row):
        """Setup context preparation section"""
        context_frame = ttk.LabelFrame(parent, text="Context Preparation", padding="5")
        context_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        context_frame.columnconfigure(1, weight=1)
        
        # Directory selection
        ttk.Label(context_frame, text="Directory:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.dir_var = tk.StringVar()
        dir_entry = ttk.Entry(context_frame, textvariable=self.dir_var, state='readonly')
        dir_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
        
        ttk.Button(context_frame, text="Browse", command=self.browse_directory).grid(row=0, column=2)
        ttk.Button(context_frame, text="Prepare Context", command=self.prepare_context).grid(row=0, column=3, padx=(5, 0))
        
        # Options row
        options_frame = ttk.Frame(context_frame)
        options_frame.grid(row=1, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(5, 0))
        options_frame.columnconfigure(1, weight=1)
        
        # File extensions
        ttk.Label(options_frame, text="Extensions:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.extensions_var = tk.StringVar(value=".cs")
        extensions_entry = ttk.Entry(options_frame, textvariable=self.extensions_var, width=20)
        extensions_entry.grid(row=0, column=1, sticky=tk.W, padx=(0, 10))
        ttk.Label(options_frame, text="(comma-separated)", foreground="gray").grid(row=0, column=2, sticky=tk.W, padx=(0, 10))
        
        # Recursive checkbox
        self.recursive_var = tk.BooleanVar(value=True)
        recursive_check = ttk.Checkbutton(options_frame, text="Recursive search", variable=self.recursive_var)
        recursive_check.grid(row=0, column=3, sticky=tk.W)
        
        # Context display with timestamp
        context_info_frame = ttk.Frame(context_frame)
        context_info_frame.grid(row=2, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(10, 0))
        context_info_frame.columnconfigure(0, weight=1)
        
        ttk.Label(context_info_frame, text="Context Preview:").grid(row=0, column=0, sticky=tk.W)
        self.context_status_label = ttk.Label(context_info_frame, text="No context loaded", foreground="gray")
        self.context_status_label.grid(row=0, column=1, sticky=tk.E)
        
        self.context_text = scrolledtext.ScrolledText(context_info_frame, height=6, wrap=tk.WORD)
        self.context_text.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))
        self.context_text.config(state=tk.DISABLED)
        
        # Copy context button
        ttk.Button(context_info_frame, text="Copy Context to Clipboard", 
                  command=self.copy_context_to_clipboard).grid(row=2, column=0, pady=(5, 0), sticky=tk.W)
    
    def setup_model_selection(self, parent, row):
        """Setup model selection dropdowns and model info display"""
        model_frame = ttk.LabelFrame(parent, text="Model Selection", padding="5")
        model_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Create model selection UI
        col = 0
        for model_name, config in self.models_config.items():
            model_col_frame = ttk.Frame(model_frame)
            model_col_frame.grid(row=0, column=col, sticky=(tk.W, tk.N), padx=(0, 20))
            
            # Checkbox to enable/disable
            cb = ttk.Checkbutton(model_col_frame, text=model_name.title(), variable=config['enabled'])
            cb.grid(row=0, column=0, sticky=tk.W)
            
            # Model dropdown
            ttk.Label(model_col_frame, text="Model:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
            model_combo = ttk.Combobox(model_col_frame, textvariable=config['selected_model'], 
                                     values=config['models'], width=25, state="readonly")
            model_combo.grid(row=2, column=0, sticky=tk.W)
            model_combo.bind('<<ComboboxSelected>>', lambda e, provider=model_name: self.update_model_info(provider))
            
            # Special handling for OpenAI reasoning effort
            if model_name == 'openai':
                ttk.Label(model_col_frame, text="Reasoning:").grid(row=3, column=0, sticky=tk.W, pady=(5, 0))
                reasoning_combo = ttk.Combobox(model_col_frame, textvariable=config['reasoning_effort'], 
                                             values=['auto', 'high', 'medium', 'low'], width=22, state="readonly")
                reasoning_combo.grid(row=4, column=0, sticky=tk.W)
                
                # Model info display
                info_label = ttk.Label(model_col_frame, text="", font=('TkDefaultFont', 8), foreground='blue')
                info_label.grid(row=5, column=0, sticky=tk.W, pady=(5, 0))
                self.model_info_labels[model_name] = info_label
            else:
                # Model info display
                info_label = ttk.Label(model_col_frame, text="", font=('TkDefaultFont', 8), foreground='blue')
                info_label.grid(row=3, column=0, sticky=tk.W, pady=(5, 0))
                self.model_info_labels[model_name] = info_label
            
            col += 1
        
        # Initialize model info display
        for provider in self.models_config.keys():
            self.update_model_info(provider)
    
    def update_model_info(self, provider):
        """Update the model info display for a provider"""
        if provider not in self.model_info_labels:
            return
            
        selected_model = self.models_config[provider]['selected_model'].get()
        
        # Map provider names to correct keys in available_models
        provider_mapping = {
            'openai': 'OpenAI',
            'claude': 'Claude', 
            'gemini': 'Gemini',
            'ollama': 'Ollama'
        }
        
        # Find model info
        provider_key = provider_mapping.get(provider, provider.title())
        provider_models = self.available_models.get(provider_key, [])
        model_info = next((m for m in provider_models if m['name'] == selected_model), None)
        
        if model_info:
            info_parts = []
            info_parts.append(f"Max tokens: {model_info['max_tokens']}")
            
            if model_info['context_window'] != "N/A":
                info_parts.append(f"Context: {model_info['context_window']}")
            
            if model_info['supports_reasoning']:
                info_parts.append("Reasoning: Yes")
            
            if model_info['thinking_enabled']:
                info_parts.append("Thinking: Yes")
                if model_info['thinking_budget'] != "N/A":
                    info_parts.append(f"Think budget: {model_info['thinking_budget']}")
            
            info_text = "\n".join(info_parts)
            self.model_info_labels[provider].config(text=info_text)
        else:
            self.model_info_labels[provider].config(text="Model info not available")
    
    def setup_prompt_section(self, parent, row):
        """Setup prompt editing section"""
        prompt_frame = ttk.LabelFrame(parent, text="Prompt", padding="5")
        prompt_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        prompt_frame.columnconfigure(0, weight=1)
        
        self.prompt_text = scrolledtext.ScrolledText(prompt_frame, height=8, wrap=tk.WORD)
        self.prompt_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Prompt controls
        prompt_controls = ttk.Frame(prompt_frame)
        prompt_controls.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        ttk.Button(prompt_controls, text="Load from File", command=self.load_prompt_from_file).grid(row=0, column=0)
        ttk.Button(prompt_controls, text="Save to File", command=self.save_prompt_to_file).grid(row=0, column=1, padx=(5, 0))
        
        # Context attachment checkbox
        self.attach_context_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(prompt_controls, text="Attach context to prompt", 
                       variable=self.attach_context_var).grid(row=0, column=2, padx=(20, 0))
    
    def setup_control_section(self, parent, row):
        """Setup control buttons"""
        control_frame = ttk.Frame(parent)
        control_frame.grid(row=row, column=0, columnspan=2, pady=(0, 10))
        
        self.submit_button = ttk.Button(control_frame, text="Submit to Selected Models", 
                                       command=self.submit_to_models, style="Accent.TButton")
        self.submit_button.grid(row=0, column=0)
        
        ttk.Button(control_frame, text="Clear All Outputs", command=self.clear_outputs).grid(row=0, column=1, padx=(10, 0))
        ttk.Button(control_frame, text="Stop All", command=self.stop_all_models).grid(row=0, column=2, padx=(10, 0))
    
    def setup_output_section(self, parent, row):
        """Setup model output section"""
        output_frame = ttk.LabelFrame(parent, text="Model Outputs", padding="5")
        output_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 0))
        
        # Create notebook for tabbed output
        self.output_notebook = ttk.Notebook(output_frame)
        self.output_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Initialize output tabs (will be created dynamically)
        self.output_tabs = {}
    
    def browse_directory(self):
        """Browse and select directory for context preparation"""
        directory = filedialog.askdirectory(title="Select Directory for Context Preparation")
        if directory:
            self.selected_directory = directory
            self.dir_var.set(directory)
    
    def prepare_context(self):
        """Prepare context from selected directory"""
        if not self.selected_directory:
            messagebox.showwarning("No Directory", "Please select a directory first.")
            return
        
        # Run context preparation in thread to avoid blocking UI
        thread = threading.Thread(target=self._prepare_context_thread)
        thread.daemon = True
        thread.start()
    
    def _prepare_context_thread(self):
        """Thread function for context preparation"""
        try:
            # Update status
            self.root.after(0, lambda: self.context_status_label.config(text="Preparing context...", foreground="blue"))
            
            # Get extensions from input
            extensions_text = self.extensions_var.get().strip()
            if not extensions_text:
                extensions = ['.cs']  # Default
            else:
                extensions = [ext.strip() for ext in extensions_text.split(',') if ext.strip()]
            
            # Get recursive setting
            recursive = self.recursive_var.get()
            
            # Find files automatically
            found_files = prep_context.find_files_by_extension(
                self.selected_directory, 
                extensions=extensions,
                recursive=recursive, 
                excluded_dirs=['obj', 'bin', '.git', '.vs']
            )
            
            if not found_files:
                ext_display = ', '.join(extensions)
                self.root.after(0, lambda: self.context_status_label.config(
                    text=f"No files found with extensions [{ext_display}]", foreground="red"))
                return
            
            # Create context content
            context_lines = []
            for file_path in found_files:  # Include all files
                try:
                    filename_header = os.path.basename(file_path)
                    context_lines.append(f"--------------\n{filename_header}\n--------------")
                    
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        context_lines.append(content)
                        context_lines.append("")  # Empty line separator
                        
                except Exception as e:
                    context_lines.append(f"Error reading {file_path}: {e}")
            
            self.context_content = "\n".join(context_lines)
            self.context_timestamp = datetime.now()
            
            # Update UI
            self.root.after(0, self._update_context_display)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to prepare context: {e}"))
    
    def _update_context_display(self):
        """Update context display in UI"""
        # Update status
        file_count = len([line for line in self.context_content.split('\n') if line.startswith('--------------')])
        time_str = self.context_timestamp.strftime("%H:%M:%S")
        self.context_status_label.config(
            text=f"{file_count} files loaded at {time_str}", 
            foreground="green"
        )
        
        # Update preview (first 2000 characters)
        preview = self.context_content[:2000]
        if len(self.context_content) > 2000:
            preview += "\n\n... (truncated for preview)"
        
        self.context_text.config(state=tk.NORMAL)
        self.context_text.delete(1.0, tk.END)
        self.context_text.insert(1.0, preview)
        self.context_text.config(state=tk.DISABLED)
    
    def copy_context_to_clipboard(self):
        """Copy context to clipboard"""
        if self.context_content:
            self.root.clipboard_clear()
            self.root.clipboard_append(self.context_content)
            messagebox.showinfo("Copied", "Context copied to clipboard!")
        else:
            messagebox.showwarning("No Context", "No context to copy. Please prepare context first.")
    
    def load_prompt_from_file(self):
        """Load prompt from file"""
        file_path = filedialog.askopenfilename(
            title="Load Prompt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.prompt_text.delete(1.0, tk.END)
                self.prompt_text.insert(1.0, content)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load prompt: {e}")
    
    def save_prompt_to_file(self):
        """Save prompt to file"""
        file_path = filedialog.asksaveasfilename(
            title="Save Prompt",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            try:
                content = self.prompt_text.get(1.0, tk.END).strip()
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                messagebox.showinfo("Saved", "Prompt saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save prompt: {e}")
    
    def submit_to_models(self):
        """Submit prompt to selected models"""
        # Get prompt text
        prompt = self.prompt_text.get(1.0, tk.END).strip()
        if not prompt:
            messagebox.showwarning("No Prompt", "Please enter a prompt.")
            return
        
        # Attach context if requested
        if self.attach_context_var.get() and self.context_content:
            full_prompt = f"{prompt}\n\n# CODEBASE\n{self.context_content}"
        else:
            full_prompt = prompt
        
        # Get selected models
        selected_models = [name for name, config in self.models_config.items() 
                          if config['enabled'].get()]
        
        if not selected_models:
            messagebox.showwarning("No Models", "Please select at least one model.")
            return
        
        # Disable submit button
        self.submit_button.config(state=tk.DISABLED)
        
        # Create output tabs for selected models
        self._create_output_tabs(selected_models)
        
        # Submit to each model in separate threads
        for model_name in selected_models:
            thread = threading.Thread(target=self._submit_to_model_thread, 
                                     args=(model_name, full_prompt))
            thread.daemon = True
            self.model_threads[model_name] = thread
            thread.start()
    
    def _create_output_tabs(self, model_names):
        """Create output tabs for selected models"""
        # Clear existing tabs
        for tab_id in list(self.output_tabs.keys()):
            self.output_notebook.forget(self.output_tabs[tab_id]['frame'])
        self.output_tabs.clear()
        
        # Create new tabs
        for model_name in model_names:
            frame = ttk.Frame(self.output_notebook)
            
            # Status label
            status_label = ttk.Label(frame, text="Waiting...", foreground="blue")
            status_label.pack(pady=5)
            
            # Output text area
            text_area = scrolledtext.ScrolledText(frame, wrap=tk.WORD)
            text_area.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Add tab
            config = self.models_config[model_name]
            tab_title = f"{model_name.title()} ({config['selected_model'].get()})"
            self.output_notebook.add(frame, text=tab_title)
            
            # Store references
            self.output_tabs[model_name] = {
                'frame': frame,
                'status': status_label,
                'text': text_area
            }
    
    def _submit_to_model_thread(self, model_name, prompt):
        """Thread function for submitting to a specific model"""
        try:
            # Update status
            self.root.after(0, lambda: self._update_model_status(model_name, "Processing...", "blue"))
            
            config = self.models_config[model_name]
            model_name_val = config['selected_model'].get()
            
            # Create a simple response capture
            response_text = ""
            
            if model_name == 'openai':
                reasoning_effort = config['reasoning_effort'].get() if config['reasoning_effort'].get() != 'auto' else None
                # Use a simplified version of the OpenAI call
                response_text = self._call_openai_model(prompt, model_name_val, reasoning_effort)
                
            elif model_name == 'claude':
                response_text = self._call_claude_model(prompt, model_name_val)
                
            elif model_name == 'gemini':
                response_text = self._call_gemini_model(prompt, model_name_val)
                
            elif model_name == 'ollama':
                response_text = self._call_ollama_model(prompt, model_name_val)
            
            # Update UI with response
            self.root.after(0, lambda: self._update_model_output(model_name, response_text, "Completed", "green"))
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self.root.after(0, lambda: self._update_model_output(model_name, error_msg, "Error", "red"))
        
        finally:
            # Re-enable submit button when all threads complete
            self.root.after(0, self._check_all_threads_complete)
    
    def _call_openai_model(self, prompt, model_name, reasoning_effort):
        """Call OpenAI model and capture response"""
        try:
            if not self.api_key_manager:
                return "OpenAI Error: No API key manager available"
            
            api_key = self._extract_api_key('openai')
            if not api_key:
                return "OpenAI Error: No OpenAI API key found"
            
            conversation = aitestbed.OpenAIConversation(
                api_key=api_key, 
                model=model_name, 
                reasoning_effort=reasoning_effort
            )
            response = conversation.ask(prompt)
            return response
        except Exception as e:
            return f"OpenAI Error: {str(e)}"
    
    def _call_claude_model(self, prompt, model_name):
        """Call Claude model and capture response"""
        try:
            if not self.api_key_manager:
                return "Claude Error: No API key manager available"
            
            api_key = self._extract_api_key('anthropic')
            if not api_key:
                return "Claude Error: No Anthropic API key found"
            
            conversation = aitestbed.ClaudeConversation(api_key=api_key)
            response = conversation.ask_with_thinking(prompt, model=model_name)
            return response
        except Exception as e:
            return f"Claude Error: {str(e)}"
    
    def _call_gemini_model(self, prompt, model_name):
        """Call Gemini model and capture response"""
        try:
            if not self.api_key_manager:
                return "Gemini Error: No API key manager available"
            
            api_key = self._extract_api_key('gemini')
            if not api_key:
                return "Gemini Error: No Gemini API key found"
            
            conversation = aitestbed.GeminiConversation(api_key=api_key, model=model_name)
            response = conversation.ask(prompt)
            return response
        except Exception as e:
            return f"Gemini Error: {str(e)}"
    
    def _call_ollama_model(self, prompt, model_name):
        """Call Ollama model and capture response"""
        try:
            conversation = aitestbed.OllamaConversation(model=model_name)
            response = conversation.ask(prompt)
            return response
        except Exception as e:
            return f"Ollama Error: {str(e)}"
    
    def _update_model_status(self, model_name, status_text, color):
        """Update model status in UI"""
        if model_name in self.output_tabs:
            self.output_tabs[model_name]['status'].config(text=status_text, foreground=color)
    
    def _update_model_output(self, model_name, response_text, status_text, status_color):
        """Update model output in UI"""
        if model_name in self.output_tabs:
            # Update status
            self.output_tabs[model_name]['status'].config(text=status_text, foreground=status_color)
            
            # Update text area
            text_area = self.output_tabs[model_name]['text']
            text_area.delete(1.0, tk.END)
            text_area.insert(1.0, response_text)
    
    def _check_all_threads_complete(self):
        """Check if all model threads are complete and re-enable submit button"""
        active_threads = [t for t in self.model_threads.values() if t.is_alive()]
        if not active_threads:
            self.submit_button.config(state=tk.NORMAL)
            self.model_threads.clear()
    
    def clear_outputs(self):
        """Clear all model outputs"""
        for model_data in self.output_tabs.values():
            model_data['text'].delete(1.0, tk.END)
            model_data['status'].config(text="Cleared", foreground="gray")
    
    def stop_all_models(self):
        """Stop all running model threads (best effort)"""
        # Note: This is a best effort stop - actual API calls may continue
        for thread in self.model_threads.values():
            if hasattr(thread, '_stop'):
                thread._stop()
        
        self.submit_button.config(state=tk.NORMAL)
        
        # Update status for all active outputs
        for model_name, model_data in self.output_tabs.items():
            if model_data['status'].cget('text') == "Processing...":
                model_data['status'].config(text="Stopped", foreground="orange")
    
    def check_api_keys(self):
        """Check for API keys file and warn if missing"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        api_keys_path = os.path.join(script_dir, 'apikeys.json')
        
        if not os.path.exists(api_keys_path):
            messagebox.showwarning(
                "API Keys Missing",
                f"API keys file not found at: {api_keys_path}\n\n"
                "This file is required for OpenAI, Claude, and Gemini APIs.\n"
                "Only Ollama will work without it."
            )
            self.api_key_manager = None
        else:
            # Set environment variable for aitestbed module
            os.environ['AITESTBED_API_KEYS_PATH'] = api_keys_path
            # Initialize API key manager
            try:
                self.api_key_manager = aitestbed.APIKeyManager(api_keys_path)
            except Exception as e:
                messagebox.showerror("API Key Error", f"Failed to load API keys: {e}")
                self.api_key_manager = None

    def _extract_api_key(self, provider):
        """Extract API key from the loaded keys, handling both direct strings and nested structures"""
        if not self.api_key_manager:
            return None
        
        provider_data = self.api_key_manager.get_key(provider)
        if not provider_data:
            return None
        
        # Handle both direct strings and nested structures
        if isinstance(provider_data, dict):
            # Look for 'api_key' field in nested structure
            return provider_data.get('api_key')
        elif isinstance(provider_data, str):
            # Direct string format
            return provider_data
        else:
            return None

def main():
    """Main entry point"""
    root = tk.Tk()
    
    # Set up styling
    style = ttk.Style()
    style.theme_use('clam')  # Modern looking theme
    
    # Create and run the application
    app = AITestbedGUI(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("Application interrupted by user")
    except Exception as e:
        messagebox.showerror("Application Error", f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()