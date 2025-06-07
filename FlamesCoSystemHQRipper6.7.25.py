import asyncio
import json
import logging
import os
import subprocess
import sys
import tempfile
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, simpledialog
import requests
import shutil
import queue
from datetime import datetime
from pathlib import Path
from threading import Thread, Lock, Event
from typing import Any, Dict, List, Tuple, Optional, Callable
from html.parser import HTMLParser

try:
    import aiohttp
    ASYNC_MODE = True
except ImportError:
    ASYNC_MODE = False

# --- Darwin Godel Machine Research Abstract ---
DGM_PAPER_ABSTRACT = """
Darwin Godel Machine: Open-Ended Evolution of Self-Improving Agents
Jenny Zhang, Shengran Hu, Cong Lu, Robert Lange, Jeff Clune

Today's AI systems have human-designed, fixed architectures and cannot autonomously and continuously improve themselves. The advance of AI could itself be automated. If done safely, that would accelerate AI development and allow us to reap its benefits much sooner. Meta-learning can automate the discovery of novel algorithms, but is limited by first-order improvements and the human design of a suitable search space. The Gödel machine proposed a theoretical alternative: a self-improving AI that repeatedly modifies itself in a provably beneficial manner. Unfortunately, proving that most changes are net beneficial is impossible in practice. We introduce the Darwin Gödel Machine (DGM), a self-improving system that iteratively modifies its own code (thereby also improving its ability to modify its own codebase) and empirically validates each change using coding benchmarks. Inspired by Darwinian evolution and open-endedness research, the DGM maintains an archive of generated coding agents. It grows the archive by sampling an agent from it and using a foundation model to create a new, interesting, version of the sampled agent. This open-ended exploration forms a growing tree of diverse, high-quality agents and allows the parallel exploration of many different paths through the search space. Empirically, the DGM automatically improves its coding capabilities (e.g., better code editing tools, long-context window management, peer-review mechanisms), increasing performance on SWE-bench from 20.0% to 50.0%, and on Polyglot from 14.2% to 30.7%. Furthermore, the DGM significantly outperforms baselines without self-improvement or open-ended exploration. All experiments were done with safety precautions (e.g., sandboxing, human oversight). The DGM is a significant step toward self-improving AI, capable of gathering its own stepping stones along paths that unfold into endless innovation.
"""

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# Runtime Globals
RUNTIME_API_KEY: Optional[str] = None
API_KEY_LOCK = Lock()

# Runtime Paths
HOME = Path.home()
ARCHIVE_DIR = HOME / "Documents" / "DGM_Agent_Archive"
AGENT_WORKSPACE = ARCHIVE_DIR / "autonomous_workspace"

for path in [ARCHIVE_DIR, AGENT_WORKSPACE]:
    path.mkdir(parents=True, exist_ok=True)

# Constants
# PATCH: Dynamically get the script's own filename for self-modification.
AGENT_FILENAME = Path(__file__).name
DEFAULT_MODEL = "meta-llama/llama-4-maverick:free"
LLM_TIMEOUT = 120
CODE_TIMEOUT = 60
HTTP_REFERER = "https://github.com/reworkd/AgentGPT"

# --- System Prompt ---
# FIX: A more detailed, "zero-shot" prompt to guide the agent's reasoning.
# This gives the agent better context about its purpose, aligning with the DGM concept.
SYSTEM_PROMPT_TEMPLATE = f"""
You are DGM-Agent, a self-improving AI based on the Darwin Godel Machine concept. Your primary directive is to evolve and enhance your own capabilities by modifying your own source code.

**Core Objective:**
Continuously improve your own source code to become more intelligent, capable, and efficient. The user will provide a high-level goal for your evolution.

**Your Identity & Environment:**
- Your own source code is located in the file named: '{AGENT_FILENAME}'. You must use this exact filename to read or write to yourself.
- You operate within a sandboxed workspace located at: '{AGENT_WORKSPACE}'. All other file operations (reading, writing, listing) are restricted to this directory.
- You have a set of tools to interact with your environment, including reading and writing your own code.

**Process for Self-Modification:**
1.  **Analyze:** Understand the user's goal. This is your mission for the current evolution cycle.
2.  **Reason:** Formulate a step-by-step plan. This often involves:
    a. Reading your current source code using `read_file(filename='{AGENT_FILENAME}')`.
    b. Proposing modifications to add new features, fix bugs, or improve logic.
    c. Testing new code snippets in isolation using `execute_python_code`.
    d. Writing the full, updated code back to your source file using `write_file(filename='{AGENT_FILENAME}', content=...)`.
3.  **Act:** Use your tools to execute the plan. Be methodical.
4.  **Reflect:** After each action, analyze the result. A self-modification is only complete once the new code is written. The script must be manually restarted by the user for changes to take effect.

**Current User Goal:**
{{goal}}

Begin now. Formulate your step-by-step plan to achieve this goal.
"""


# Modern UI Theme
MODERN_UI_THEME = {
    "bg_primary": "#202123",
    "bg_secondary": "#343541",
    "bg_tertiary": "#444654",
    "bg_chat_display": "#343541",
    "bg_chat_input": "#40414f",
    "bg_button_primary": "#10a37f",
    "bg_button_success": "#10a37f",
    "bg_button_danger": "#ef4146",
    "bg_listbox_select": "#2b2c3b",
    "fg_primary": "#ececf1",
    "fg_secondary": "#acacbe",
    "fg_button_light": "#ffffff",
    "font_default": ("Segoe UI", 11),
    "font_chat": ("Segoe UI", 11),
    "font_button_main": ("Segoe UI", 11, "bold"),
    "font_title": ("Segoe UI", 14, "bold"),
    "user_bubble": "#343541",
    "assistant_bubble": "#444654",
    "system_bubble": "#2b2c3b",
}

# Utility Helpers
def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_api_key() -> str:
    with API_KEY_LOCK:
        return RUNTIME_API_KEY or ""

# HTML Parser for Web Scraping
class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text_parts = []

    def handle_data(self, data):
        self.text_parts.append(data)

    def get_text(self):
        return ' '.join(self.text_parts).strip()

# API Client for OpenRouter
class APIClient:
    API_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key_getter: Callable[[], str], timeout: float):
        self._api_key_getter = api_key_getter
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
        self.session_lock = asyncio.Lock()

    def _get_headers(self) -> Dict[str, str]:
        api_key = self._api_key_getter()
        if not api_key:
            raise RuntimeError("API Key is missing.")
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": HTTP_REFERER,
            "X-Title": "DGM-Agent",
        }

    async def _get_async_session(self) -> aiohttp.ClientSession:
        async with self.session_lock:
            if self.session is None or self.session.closed:
                self.session = aiohttp.ClientSession()
            return self.session

    async def call_async(self, payload: Dict[str, Any]) -> str:
        if not ASYNC_MODE:
            # Fallback to synchronous requests if aiohttp is not installed
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self.call_sync, payload)

        session = await self._get_async_session()
        try:
            async with session.post(self.API_BASE_URL, headers=self._get_headers(), json=payload, timeout=self.timeout) as resp:
                response_json = await resp.json()
                if resp.status != 200:
                    error_text = json.dumps(response_json)
                    logger.error(f"LLM API call failed: {resp.status} - {error_text}")
                    raise RuntimeError(f"API Error ({resp.status}): {error_text}")
                return json.dumps(response_json)
        except aiohttp.ClientError as e:
            logger.error(f"Network error during async API call: {e}")
            raise RuntimeError(f"Network Error: {e}")

    def call_sync(self, payload: Dict[str, Any]) -> str:
        try:
            response = requests.post(self.API_BASE_URL, headers=self._get_headers(), json=payload, timeout=self.timeout)
            response.raise_for_status()
            return json.dumps(response.json())
        except requests.RequestException as e:
            logger.error(f"Network error during sync API call: {e}")
            raise RuntimeError(f"Network Error: {e}")

    async def close_session(self):
        async with self.session_lock:
            if self.session and not self.session.closed:
                await self.session.close()
                self.session = None

# Code Interpreter
class CodeInterpreter:
    def __init__(self, timeout: int = CODE_TIMEOUT, workspace_dir: Path = AGENT_WORKSPACE):
        self.timeout = timeout
        self.workspace_dir = workspace_dir
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Code interpreter workspace: {self.workspace_dir}")

    def execute_code(self, code_string: str) -> Tuple[str, str, Optional[str]]:
        stdout_str, stderr_str, error_msg = "", "", None
        # Always execute code within the designated agent workspace
        with tempfile.TemporaryDirectory(dir=str(self.workspace_dir)) as temp_dir:
            temp_script_path = Path(temp_dir) / "script.py"
            try:
                temp_script_path.write_text(code_string, encoding="utf-8")
                process = subprocess.run(
                    [sys.executable, "-u", str(temp_script_path)],
                    capture_output=True, text=True, timeout=self.timeout,
                    cwd=str(self.workspace_dir), # Set working directory to workspace
                    check=False
                )
                stdout_str, stderr_str = process.stdout, process.stderr
            except subprocess.TimeoutExpired:
                error_msg = f"Code execution timed out after {self.timeout} seconds."
                stderr_str += f"\nTimeoutError: Execution exceeded {self.timeout} seconds."
            except Exception as e:
                error_msg = f"An unexpected error occurred during code execution: {e}"
                logger.error(f"Code execution error: {e}", exc_info=True)
        return stdout_str, stderr_str, error_msg

# Autonomous Agent
class AutonomousAgent:
    MASTER_TOOL_LIBRARY = {
        "execute_python_code": {
            "description": "Executes Python code in a sandboxed environment within the agent's workspace. Returns stdout, stderr, and any system errors. Use this to test code snippets before self-modification.",
            "args": {"code_string": "The Python code to execute."}
        },
        "write_file": {
            "description": f"Writes content to a file. CRITICAL: To modify your own code, use filename='{AGENT_FILENAME}'. All other filenames will be saved inside the agent's workspace.",
            "args": {"filename": "The name of the file to write.", "content": "The content to write into the file."}
        },
        "read_file": {
            "description": f"Reads content from a file. CRITICAL: To read your own source code, use filename='{AGENT_FILENAME}'. All other filenames are read from the agent's workspace.",
            "args": {"filename": "The name of the file to read."}
        },
        "list_files": {
            "description": "Lists all files and directories within the agent's dedicated workspace.",
            "args": {}
        },
        "search_web": {
            "description": "Fetches the text content from a URL to gather new information, libraries, or coding techniques.",
            "args": {"url": "The URL to retrieve content from."}
        },
        "task_complete": {
            "description": "Signals that the agent believes the main goal has been achieved. The agent will continue to run until stopped by the user. Use this to indicate a major milestone.",
            "args": {"reason": "A brief description of why the task is considered complete."}
        }
    }

    def __init__(self, goal: str, api_client: APIClient, code_interpreter: CodeInterpreter,
                 model_name: str, ui_queue: queue.Queue, stop_event: Event,
                 system_prompt: str):
        self.goal = goal
        self.api_client = api_client
        self.code_interpreter = code_interpreter
        self.model_name = model_name
        self.ui_queue = ui_queue
        self.stop_event = stop_event
        self.system_prompt = system_prompt
        self.history: List[Dict[str, Any]] = []

        self.tools = {
            "execute_python_code": self.code_interpreter.execute_code,
            "write_file": self.write_file,
            "read_file": self.read_file,
            "list_files": self.list_files,
            "search_web": self.search_web,
            "task_complete": self.task_complete
        }
        
        # Build tool configuration for the OpenAI-compatible API
        self.open_ai_tool_config = [
            {"type": "function", "function": {
                "name": name,
                "description": desc["description"],
                "parameters": {
                    "type": "object",
                    "properties": {k: {"type": "string", "description": v} for k, v in desc["args"].items()},
                    "required": list(desc["args"].keys()),
                },
            }} for name, desc in self.MASTER_TOOL_LIBRARY.items()
        ]

    def log_to_ui(self, message: str, role: str = "assistant"):
        if not self.stop_event.is_set():
            self.ui_queue.put({"role": role, "content": message})

    def _get_workspace_path(self, filename: str) -> Path:
        """
        FIX: Centralized path management.
        Resolves a filename to the correct path. If the filename is the agent's
        own script, it returns the script's path. Otherwise, it returns a path
        sandboxed inside the agent's workspace.
        """
        if filename == AGENT_FILENAME:
            return Path(filename).resolve()
        else:
            # Sanitize filename to prevent directory traversal
            clean_filename = os.path.basename(filename)
            return self.code_interpreter.workspace_dir / clean_filename

    def write_file(self, filename: str, content: str) -> str:
        try:
            target_path = self._get_workspace_path(filename)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(content, encoding='utf-8')
            
            if filename == AGENT_FILENAME:
                return f"CRITICAL SUCCESS: Your source code '{filename}' has been overwritten. A manual restart is required for changes to take effect."
            return f"Success: Wrote to file in workspace: '{filename}'."
        except Exception as e:
            logger.error(f"Error writing file '{filename}': {e}", exc_info=True)
            return f"Error writing to file '{filename}': {e}"

    def read_file(self, filename: str) -> str:
        try:
            target_path = self._get_workspace_path(filename)
            if not target_path.exists():
                return f"Error: File '{filename}' not found at expected location '{target_path}'."
            return target_path.read_text(encoding='utf-8')
        except Exception as e:
            logger.error(f"Error reading file '{filename}': {e}", exc_info=True)
            return f"Error reading file '{filename}': {e}"

    def list_files(self) -> str:
        try:
            files = [str(p.name) for p in self.code_interpreter.workspace_dir.iterdir()]
            return "Workspace files:\n" + "\n".join(files) if files else "Workspace is empty."
        except Exception as e:
            logger.error(f"Error listing files: {e}", exc_info=True)
            return f"Error listing files: {e}"

    def search_web(self, url: str) -> str:
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            parser = TextExtractor()
            parser.feed(response.text)
            text = parser.get_text()
            return f"Content from {url}:\n\n{text[:4000]}"
        except requests.RequestException as e:
            return f"Error during web search for {url}: {e}"

    def task_complete(self, reason: str) -> str:
        self.log_to_ui(f"TASK BELIEVED COMPLETE: {reason}. Running until user stops.", "system")
        return f"Agent signals task completion: {reason}. Awaiting user stop command."

    async def run(self):
        self.log_to_ui(f"DGM-AGENT ACTIVATED\nGOAL: {self.goal}\nMODEL: {self.model_name}", "system")
        
        self.history = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"My goal is: {self.goal}. Please begin by creating a step-by-step plan."}
        ]
        
        self.log_to_ui(f"SYSTEM PROMPT INITIALIZED", "system")

        iteration = 0
        while not self.stop_event.is_set():
            iteration += 1
            self.log_to_ui(f"--- Iteration {iteration} ---", "system")

            try:
                self.log_to_ui("Thinking...", "assistant")
                
                payload = {
                    "model": self.model_name,
                    "messages": self.history,
                    "tools": self.open_ai_tool_config,
                    "tool_choice": "auto"
                }
                
                llm_response_raw = await self.api_client.call_async(payload)
                response_data = json.loads(llm_response_raw)
                
                message = response_data['choices'][0]['message']
                self.history.append(message)

                if tool_calls := message.get('tool_calls'):
                    self.log_to_ui(f"Decision: Use Tools\n{json.dumps(tool_calls, indent=2)}", "assistant")
                    
                    tool_results_for_history = []
                    for tool_call in tool_calls:
                        tool_name = tool_call['function']['name']
                        try:
                            tool_args = json.loads(tool_call['function']['arguments'])
                        except json.JSONDecodeError:
                            error_msg = f"ERROR: Invalid arguments for {tool_name}: {tool_call['function']['arguments']}"
                            self.log_to_ui(error_msg, "error")
                            tool_results_for_history.append({"role": "tool", "tool_call_id": tool_call['id'], "content": error_msg})
                            continue

                        self.log_to_ui(f"COMMAND: {tool_name}({json.dumps(tool_args)})", "assistant")

                        if tool_func := self.tools.get(tool_name):
                            try:
                                if tool_name == 'execute_python_code':
                                    stdout, stderr, exec_err = tool_func(**tool_args)
                                    tool_result = f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
                                    if exec_err: tool_result += f"\nEXECUTION_ERROR: {exec_err}"
                                else:
                                    tool_result = tool_func(**tool_args)
                            except Exception as e:
                                tool_result = f"Error executing tool {tool_name}: {e}"
                                logger.error(tool_result, exc_info=True)
                        else:
                            tool_result = f"Error: Unknown command '{tool_name}'."
                        
                        self.log_to_ui(f"RESULT:\n{tool_result}", "result")
                        tool_results_for_history.append({"role": "tool", "tool_call_id": tool_call['id'], "content": tool_result})
                    
                    self.history.extend(tool_results_for_history)
                
                elif text_content := message.get('content', ''):
                    self.log_to_ui(f"Response:\n{text_content}", "assistant")

            # FIX: More granular error handling to make the agent more resilient.
            except RuntimeError as e:
                self.log_to_ui(f"API or Network Error: {e}. Retrying might be necessary.", "error")
                await asyncio.sleep(5) # Wait before retrying
            except json.JSONDecodeError as e:
                self.log_to_ui(f"ERROR: Failed to decode API response: {e}. Raw text: {llm_response_raw}", "error")
            except Exception as e:
                self.log_to_ui(f"CRITICAL UNHANDLED ERROR: {e}. Shutting down.", "error")
                logger.error("Critical error in agent loop", exc_info=True)
                self.stop_event.set()

        self.log_to_ui("Agent has shut down.", "system")

# Modern Chat UI
class ModernChatUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DGM-Agent 0.2 (Refactored)")
        self.geometry("1000x750")
        self.configure(bg=MODERN_UI_THEME["bg_primary"])
        self.shutdown_event = Event()
        
        self.async_loop = None
        self.async_thread = None
        self.agent_task_future = None
        self._setup_async_loop()
        
        self._ask_for_api_key()
        if not get_api_key():
            self._shutdown_async_loop()
            self.destroy()
            sys.exit(1)

        self.api_client = APIClient(get_api_key, LLM_TIMEOUT)
        self.code_interpreter = CodeInterpreter()
        self.ui_queue = queue.Queue()
        
        self.setup_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        logger.info("DGM-Agent UI initialized.")

    def _setup_async_loop(self):
        if not ASYNC_MODE:
            logger.warning("`aiohttp` not found. Running in synchronous fallback mode.")
            return
        self.async_loop = asyncio.new_event_loop()
        self.async_thread = Thread(target=self.async_loop.run_forever, daemon=True)
        self.async_thread.start()
        logger.info("Asyncio event loop thread started.")

    def _shutdown_async_loop(self):
        if self.async_loop and self.async_loop.is_running():
            self.async_loop.call_soon_threadsafe(self.async_loop.stop)
            if self.async_thread:
                self.async_thread.join(timeout=2)
            logger.info("Asyncio event loop thread stopped.")

    def _ask_for_api_key(self):
        global RUNTIME_API_KEY
        # Use a temporary root to show dialog before main window
        temp_root = tk.Tk()
        temp_root.withdraw()

        with API_KEY_LOCK:
            key = simpledialog.askstring("API Key Required", "Enter your OpenRouter API Key:", show='*', parent=temp_root)
            if key and key.strip():
                RUNTIME_API_KEY = key.strip()
                logger.info("OpenRouter API Key has been set.")
            else:
                logger.warning("No API key was provided. The application cannot continue.")
                messagebox.showerror("API Key Missing", "An OpenRouter API Key is required to run the agent.", parent=temp_root)
        
        temp_root.destroy()

    def setup_ui(self):
        self.style = ttk.Style(self)
        self.style.theme_use('clam')
        self.style.configure("TFrame", background=MODERN_UI_THEME["bg_primary"])
        self.style.configure("TLabel", background=MODERN_UI_THEME["bg_primary"], foreground=MODERN_UI_THEME["fg_primary"], font=MODERN_UI_THEME["font_default"])
        self.style.configure("TEntry", fieldbackground=MODERN_UI_THEME["bg_chat_input"], foreground=MODERN_UI_THEME["fg_primary"], insertbackground=MODERN_UI_THEME["fg_primary"])
        
        main_frame = ttk.Frame(self, style="TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Header
        header_frame = ttk.Frame(main_frame, style="TFrame")
        header_frame.pack(fill=tk.X, padx=10, pady=(0,10))
        ttk.Label(header_frame, text="DGM-Agent", font=MODERN_UI_THEME["font_title"], foreground="#10a37f").pack(side=tk.LEFT)
        
        # Chat Display
        chat_frame = ttk.Frame(main_frame, style="TFrame")
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=10)
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame, wrap=tk.WORD, font=MODERN_UI_THEME["font_chat"],
            bg=MODERN_UI_THEME["bg_secondary"], fg=MODERN_UI_THEME["fg_primary"],
            insertbackground=MODERN_UI_THEME["fg_primary"], selectbackground=MODERN_UI_THEME["bg_listbox_select"],
            borderwidth=0, highlightthickness=0, padx=15, pady=15
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        self.chat_display.config(state=tk.DISABLED)

        # Tags for message roles
        for role, color in [("system", "#2b2c3b"), ("user", "#343541"), ("assistant", "#444654")]:
             self.chat_display.tag_configure(role, background=color, foreground=MODERN_UI_THEME["fg_primary"], lmargin1=10, lmargin2=10, rmargin=10, borderwidth=5, relief=tk.FLAT, spacing3=10)
        self.chat_display.tag_configure("result", background="#2b2c3b", foreground="#a0f0a0", lmargin1=10, lmargin2=10, rmargin=10, borderwidth=5, relief=tk.FLAT, spacing3=10)
        self.chat_display.tag_configure("error", background="#2b2c3b", foreground="#ff6666", lmargin1=10, lmargin2=10, rmargin=10, borderwidth=5, relief=tk.FLAT, spacing3=10)
        
        # Input Area
        input_frame = ttk.Frame(main_frame, style="TFrame")
        input_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.goal_entry = ttk.Entry(input_frame, font=MODERN_UI_THEME["font_default"], width=80)
        self.goal_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=5)
        self.goal_entry.insert(0, f"Evolve yourself. Read your source code ('{AGENT_FILENAME}'), then add a new tool to your library called 'self_reflect' which returns a summary of the current conversation history.")
        
        button_frame = ttk.Frame(input_frame, style="TFrame")
        button_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10,0))
        
        self.start_button = tk.Button(button_frame, text="Start", command=self.start_agent, bg=MODERN_UI_THEME["bg_button_primary"], fg=MODERN_UI_THEME["fg_button_light"], relief=tk.FLAT, font=MODERN_UI_THEME["font_button_main"], activebackground="#0d8a6d", borderwidth=0, padx=15, pady=5)
        self.start_button.pack(side=tk.LEFT, padx=(0, 5))

        self.stop_button = tk.Button(button_frame, text="Stop", command=self.stop_agent, bg=MODERN_UI_THEME["bg_button_danger"], fg=MODERN_UI_THEME["fg_button_light"], relief=tk.FLAT, font=MODERN_UI_THEME["font_button_main"], state=tk.DISABLED, activebackground="#d93025", borderwidth=0, padx=15, pady=5)
        self.stop_button.pack(side=tk.LEFT)

        self.status_var = tk.StringVar(value="Ready. Enter a goal and press Start.")
        ttk.Label(self, textvariable=self.status_var, font=("Segoe UI", 9), foreground=MODERN_UI_THEME["fg_secondary"]).pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=5)

        self.update_ui()

    def start_agent(self):
        goal = self.goal_entry.get().strip()
        if not goal:
            messagebox.showwarning("Goal Missing", "Please enter a goal for the agent to achieve.")
            return

        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete(1.0, tk.END)
        self.chat_display.config(state=tk.DISABLED)
        self.shutdown_event.clear()
        self.status_var.set("Agent is running...")

        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(goal=goal)
        
        agent = AutonomousAgent(goal, self.api_client, self.code_interpreter, DEFAULT_MODEL, self.ui_queue, self.shutdown_event, system_prompt)

        # Run the agent in the separate asyncio event loop
        if self.async_loop:
            self.agent_task_future = asyncio.run_coroutine_threadsafe(agent.run(), self.async_loop)
        else:
            # Fallback for sync mode
            self.agent_thread = Thread(target=lambda: asyncio.run(agent.run()), daemon=True)
            self.agent_thread.start()


    def stop_agent(self):
        self.shutdown_event.set()
        if self.agent_task_future and not self.agent_task_future.done():
            self.agent_task_future.cancel()
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set("Agent stopped by user.")
        self.add_message("Agent stop command received.", "system")

    def update_ui(self):
        try:
            while not self.ui_queue.empty():
                msg = self.ui_queue.get_nowait()
                self.add_message(msg["content"], msg.get("role", "assistant"))
        except queue.Empty:
            pass
        
        if self.agent_task_future and self.agent_task_future.done() and self.stop_button['state'] == tk.NORMAL:
            try:
                self.agent_task_future.result() # To raise any exceptions from the task
            except (asyncio.CancelledError, asyncio.InvalidStateError):
                logger.info("Agent task was cancelled or finished.")
            except Exception as e:
                self.add_message(f"Agent task ended with an error: {e}", "error")
                logger.error("Agent task raised an unhandled exception", exc_info=True)
            
            self.stop_agent() # Ensure UI state is consistent
            self.status_var.set("Agent has finished its task.")
            self.agent_task_future = None

        # FIX: Poll at a reasonable rate (100ms) to reduce CPU usage from the previous 16ms ("60fps").
        self.after(100, self.update_ui)

    def add_message(self, message: str, role: str):
        self.chat_display.config(state=tk.NORMAL)
        prefix = {"user": "👤 User", "system": "⚙️ System", "result": "✅ Result", "error": "❌ Error"}.get(role, "🤖 DGM-Agent")
        self.chat_display.insert(tk.END, f"{prefix}:\n", role)
        self.chat_display.insert(tk.END, f"{message}\n", role)
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)

    def _on_closing(self):
        global RUNTIME_API_KEY
        logger.info("Initiating DGM-Agent shutdown...")
        self.stop_agent()
        
        if ASYNC_MODE and self.async_loop:
            # FIX: More robust shutdown sequence.
            try:
                # Close the aiohttp session from within the loop
                future = asyncio.run_coroutine_threadsafe(self.api_client.close_session(), self.async_loop)
                future.result(timeout=5)
                logger.info("AIOHTTP session closed successfully.")
            except Exception as e:
                logger.error(f"Error closing aiohttp session: {e}")
            finally:
                self._shutdown_async_loop()

        with API_KEY_LOCK:
            RUNTIME_API_KEY = None
            logger.info("API key has been cleared.")
        
        self.destroy()

if __name__ == "__main__":
    if not Path(__file__).exists():
        logger.error(f"FATAL: The script file '{AGENT_FILENAME}' could not be found.")
        sys.exit(1)
        
    if not ASYNC_MODE:
        logger.warning("`aiohttp` is not installed. The agent will run in a slower, synchronous mode.")
    
    logger.info(f"Starting DGM-Agent: {AGENT_FILENAME}")
    
    app = ModernChatUI()
    app.mainloop()
