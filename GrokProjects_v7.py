"""
Grok Projects CLI - Website-Parity Coding Companion (Enhanced, v7 for Grok)
====================================================================

Feature highlights (Adapted for Grok models by xAI):
- Project isolation: add/remove/list files in a per-project sandbox (Grok sees ONLY current project files)
- Strict scope control: no ambient filesystem access beyond the active project's /files folder
- Rich chat with selective file inclusion (-n, -s, -f, -m) and token-aware guards
- Automatic artifact extraction: large code blocks saved as versioned files under /artifacts
- Model management: grok-beta (default), grok-4, grok-3, grok-3-mini; live `models` list from API
- Streamed responses with tool support (new in v7) and automatic non-stream fallback
- Token estimation (tiktoken optional)
- Conversations saved to JSON for auditability
- Consolidated project history saved at conversations/project_history.json
- Re-open last project at startup; `history [N]` to view recent chats
- Tool integration for real-time capabilities (e.g., current date, web search)
- Command completion for file-related commands (new in v7)
- Persistent CLI command history

Hardening:
- Auto-retry non-streaming if streaming fails
- Compatible with xAI's OpenAI-style API
- Early API key validation at startup
- Tool handling loop for function calls

Usage quickstart:
  pip install --upgrade openai python-dotenv tiktoken requests beautifulsoup4 pyreadline3  # pyreadline3 for Windows history
  export XAI_API_KEY="xai-..."
  python GrokProjects_v7.py

Batch mode (run instructions from a file):
  python GrokProjects_v7.py my_instructions.txt

Note: For API pricing and limits, visit https://x.ai/api
"""

import os
import json
import cmd
import shutil
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Iterable
from datetime import datetime
import mimetypes
import re
import sys
import requests
from bs4 import BeautifulSoup

# --- Persistent CLI command history (readline/pyreadline3) ---
try:
    import readline  # built-in on macOS/Linux
except Exception:
    try:
        import pyreadline3 as readline  # Windows: pip install pyreadline3
    except Exception:
        readline = None

from dotenv import load_dotenv
load_dotenv()

# --- Optional tokenizer for better estimates ---
try:
    import tiktoken
    _HAS_TIKTOKEN = True
except Exception:
    _HAS_TIKTOKEN = False

# --- xAI client bootstrap (using OpenAI-compatible SDK) ---
try:
    from openai import OpenAI
    _XAI_CLIENT = OpenAI(
        api_key=os.getenv("XAI_API_KEY"),
        base_url="https://api.x.ai/v1"
    )
except Exception:
    _XAI_CLIENT = None

# ---- Pricing: Not available in code; redirect to https://x.ai/api ----
# Removed pricing features as per guidelines

DEFAULT_MODEL_NAME = "grok-beta"  # Updated default for v7
# Grok limits (conservative)
MAX_INPUT_TOKENS = 200_000   # Actual context may vary
MAX_TOTAL_TOKENS = 256_000   # input + output

SYSTEM_INSTRUCTIONS = (
    "You are Grok, a helpful and maximally truthful AI built by xAI, an elite software engineer and assistant."
    " When asked to write code, produce a complete, runnable program with clear structure,"
    " include any helpers/configs required, and avoid omissions. If third-party packages are"
    " needed, briefly note them at the top in a comment: '# deps: ...'."
    " Use tools when necessary for real-time or external information."
)

# Tool definitions
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_date",
            "description": "Get the current date and time in UTC.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Perform a web search and return a summary of results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    # Add more tools as needed
]

# =====================
# Core Project classes
# =====================
class Project:
    """Manages project files, conversations, and metadata (strictly scoped)"""
    def __init__(self, name: str, base_path: Optional[Path] = None):
        self.name = name
        self.base_path = base_path or Path.cwd() / "Grok_Projects"
        self.project_path = self.base_path / name
        self.files_path = self.project_path / "files"
        self.artifacts_path = self.project_path / "artifacts"
        self.conversations_path = self.project_path / "conversations"
        self.metadata_path = self.project_path / "metadata.json"

        self.files_path.mkdir(parents=True, exist_ok=True)
        self.artifacts_path.mkdir(parents=True, exist_ok=True)
        self.conversations_path.mkdir(parents=True, exist_ok=True)

        self.history_path = self.conversations_path / "project_history.json"
        self.metadata = self._load_metadata()
        self.history = self._load_history()

    # ----- File scope strictly inside this project's /files -----
    def add_file(self, filepath: Path) -> str:
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        dest = self.files_path / filepath.name
        if dest.exists():
            stem, suffix = filepath.stem, filepath.suffix
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            dest = self.files_path / f"{stem}_{ts}{suffix}"
        shutil.copy2(filepath, dest)
        self.metadata['files'][dest.name] = {
            'added': datetime.now().isoformat(),
            'original_path': str(filepath),
            'size': dest.stat().st_size,
        }
        self._save_metadata()
        return dest.name

    def remove_file(self, filename: str) -> bool:
        p = self.files_path / filename
        if p.exists():
            p.unlink()
            self.metadata['files'].pop(filename, None)
            self._save_metadata()
            return True
        return False

    def list_files(self) -> List[Tuple[str, int, str]]:
        out: List[Tuple[str, int, str]] = []
        for p in sorted(self.files_path.iterdir()):
            if p.is_file():
                size = p.stat().st_size
                modified = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                out.append((p.name, size, modified))
        return out

    def get_file_content(self, filename: str) -> str:
        p = self.files_path / filename
        if not p.exists():
            raise FileNotFoundError(f"File not found: {filename}")
        mime, _ = mimetypes.guess_type(str(p))
        if mime and mime.startswith('text'):
            return p.read_text(encoding='utf-8', errors='replace')
        if p.suffix.lower() in {'.txt', '.md', '.py', '.js', '.ts', '.tsx', '.html', '.css', '.json', '.xml', '.yaml', '.yml', '.sql'}:
            return p.read_text(encoding='utf-8', errors='replace')
        return f"[Binary file: {filename} ({mime or 'unknown type'})]"

    def get_project_context(self) -> str:
        parts = [f"Project: {self.name}\n{'='*60}\n", "Files in this project:\n"]
        any_file = False
        for p in sorted(self.files_path.iterdir()):
            if p.is_file():
                any_file = True
                parts.append(f"\n--- File: {p.name} ---\n")
                try:
                    parts.append(self.get_file_content(p.name))
                except Exception as e:
                    parts.append(f"[Error reading file: {e}]")
                parts.append("\n")
        if not any_file:
            parts.append("(No files in project yet)\n")
        return "".join(parts)

    def save_conversation(
        self,
        messages: List[Dict[str, Any]],
        response_text: str,
        artifacts: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_{ts}.json"
        convo = {
            'timestamp': datetime.now().isoformat(),
            'messages': messages,
            'response': response_text,
            'artifacts': artifacts or [],
            'context': context or {},
        }
        fp = self.conversations_path / filename
        with open(fp, 'w', encoding='utf-8') as f:
            json.dump(convo, f, indent=2, ensure_ascii=False)

        # Also append to consolidated project history
        self.add_history_entry(messages, response_text, artifacts or [], context or {})

        return filename

    def list_conversations(self) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        for p in sorted(self.conversations_path.iterdir(), reverse=True):
            if p.suffix == '.json' and p.name != 'project_history.json':
                try:
                    d = json.loads(p.read_text(encoding='utf-8'))
                    out.append((p.name, d.get('timestamp', 'Unknown')))
                except Exception:
                    out.append((p.name, 'Unknown'))
        return out

    def sync_files(self) -> List[str]:
        synced: List[str] = []
        # purge metadata for missing files
        missing = [fn for fn in list(self.metadata['files'].keys()) if not (self.files_path / fn).exists()]
        for fn in missing:
            self.metadata['files'].pop(fn, None)
        # add metadata for new files
        for p in self.files_path.iterdir():
            if p.is_file() and p.name not in self.metadata['files']:
                self.metadata['files'][p.name] = {
                    'added': datetime.now().isoformat(),
                    'original_path': 'manually_added',
                    'size': p.stat().st_size,
                }
                synced.append(p.name)
        if missing or synced:
            self._save_metadata()
        return synced

    def _load_metadata(self) -> Dict[str, Any]:
        if self.metadata_path.exists():
            try:
                return json.loads(self.metadata_path.read_text(encoding='utf-8'))
            except Exception:
                pass
        return {'created': datetime.now().isoformat(), 'files': {}, 'settings': {}}

    def _save_metadata(self) -> None:
        self.metadata['updated'] = datetime.now().isoformat()
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

    # -------- Project consolidated history ----------
    def _load_history(self) -> Dict[str, Any]:
        if self.history_path.exists():
            try:
                return json.loads(self.history_path.read_text(encoding='utf-8'))
            except Exception:
                pass
        return {"project": self.name, "created": datetime.now().isoformat(), "entries": []}

    def _save_history(self) -> None:
        self.history["updated"] = datetime.now().isoformat()
        with open(self.history_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)

    def add_history_entry(
        self,
        messages: List[Dict[str, Any]],
        response_text: str,
        artifacts: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ts = datetime.now().isoformat()
        entry = {
            "timestamp": ts,
            "messages": messages,
            "response": {
                "text_len": len(response_text),
                "word_count": len(response_text.split()),
            },
            "artifacts": [
                {
                    "filename": a.get("filename"),
                    "language": a.get("language"),
                    "title": a.get("title"),
                } for a in (artifacts or [])
            ],
            "context": context or {},
        }
        self.history.setdefault("entries", []).append(entry)
        self._save_history()
        return entry

# =====================
# Artifacts
# =====================
class ArtifactManager:
    def __init__(self, project: Project):
        self.path = project.artifacts_path

    def create_artifact(self, content: str, language: str, title: str) -> Dict[str, Any]:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = re.sub(r"[^a-zA-Z0-9_\-]", "_", title).strip("_") or "artifact"
        ext = self._language_ext(language)
        filename = f"{safe_title}_{ts}.{ext}"
        fp = self.path / filename
        with open(fp, 'w', encoding='utf-8') as f:
            f.write(content)
        return {
            'filename': filename,
            'path': str(fp),
            'language': language,
            'title': title,
            'created': datetime.now().isoformat(),
        }

    def extract_code_blocks(self, response_text: str) -> List[Dict[str, Any]]:
        artifacts: List[Dict[str, Any]] = []
        # Match triple-fenced code with language: ```python\n...``` (greedy DOTALL)
        for i, m in enumerate(re.finditer(r"```(\w+)\n(.*?)```", response_text, re.DOTALL)):
            language, code = m.group(1), m.group(2)
            if len(code.strip()) < 100:  # skip tiny snippets
                continue
            title = self._derive_title(code, language, i)
            artifacts.append(self.create_artifact(code, language, title))
        return artifacts

    def _derive_title(self, code: str, language: str, idx: int) -> str:
        if language.lower() == 'python':
            m = re.search(r"class\s+(\w+)", code)
            if m: return m.group(1)
            m = re.search(r"def\s+(\w+)", code)
            if m: return m.group(1)
        # comment header
        first = code.strip().split("\n", 1)[0]
        if first.startswith(('#', '//', '/*', '--')):
            return re.sub(r"^([#/\-*\s]+)", "", first).strip().replace(' ', '_')[:50] or f"Code_Artifact_{idx+1}"
        return f"Code_Artifact_{idx+1}"

    def _language_ext(self, lang: str) -> str:
        m = lang.lower()
        return {
            'python': 'py', 'javascript': 'js', 'typescript': 'ts', 'tsx': 'tsx', 'jsx': 'jsx',
            'java': 'java', 'c': 'c', 'cpp': 'cpp', 'csharp': 'cs', 'go': 'go', 'rust': 'rs',
            'php': 'php', 'ruby': 'rb', 'swift': 'swift', 'kotlin': 'kt', 'json': 'json',
            'yaml': 'yml', 'markdown': 'md', 'shell': 'sh', 'bash': 'sh', 'html': 'html', 'css': 'css'
        }.get(m, m)


# =====================
# Batch Instruction Runner
# =====================
def run_instruction_file(cli, file_path: Path, *, stop_on_error: bool = True, echo: bool = True) -> None:
    """
    Execute a sequence of CLI commands from a text file.

    Rules:
      • Each non-empty, non-comment line is executed as if typed at the prompt.
      • Lines starting with '#' are treated as comments and skipped.
      • Use a trailing backslash '\' to continue a command onto the next line.
      • Whitespace-only lines are ignored.
      • On error, the default is to stop; set stop_on_error=False to continue.

    Example file:
        # create and open a project, then chat
        create my_project
        open my_project
        add ./some_code.py
        chat "Analyze the code and suggest improvements"
        tokens

    Tip: Multi-line prompts can be written with a trailing backslash:
        chat Analyze this code and provide a plan:\
             What are the modules?\
             Which tests are missing?
    """
    p = Path(file_path).expanduser()
    if not p.exists():
        print(f"❌ Instruction file not found: {p}")
        return

    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    buffer = []
    def flush_buffer(line_no: int):
        if not buffer:
            return
        cmd_line = "".join(buffer).strip()
        if cmd_line:
            if echo:
                print(f"{cli.prompt}{cmd_line}")
            try:
                result = cli.onecmd(cmd_line)
                # If a command signals exit (returns True), stop execution.
                if result:
                    raise SystemExit
            except SystemExit:
                # Propagate exit to stop script processing.
                raise
            except Exception as e:
                print(f"❌ Error on line {line_no}: {e}")
                if stop_on_error:
                    raise
        buffer.clear()

    for idx, raw in enumerate(lines, start=1):
        line = raw.rstrip("\n")
        stripped = line.strip()

        # Skip blank lines and comments
        if not stripped or stripped.startswith("#"):
            # If we had a continued line, blank/comment ends it.
            flush_buffer(idx)
            continue

        # Handle line continuation with a trailing backslash
        if stripped.endswith("\\"):
            # Append without the trailing backslash; keep a single space for separation
            buffer.append(stripped[:-1] + " ")
            continue

        # Regular line: add to buffer and flush
        buffer.append(stripped)
        flush_buffer(idx)

    # Flush any remaining buffer at EOF
    flush_buffer(len(lines))

# =====================
# xAI Grok Client wrapper
# =====================
class GrokClient:
    WEBSITE_STYLE = (
        "When analyzing code: first explain what it does, then identify key components,"
        " outline the workflow, and provide specific actionable improvements."
        " When suggesting improvements: return complete code blocks (with language fences),"
        " and explain benefits. Use section headers and bullet points judiciously."
    )

    def __init__(self):
        if _XAI_CLIENT is None:
            raise RuntimeError("xAI client not initialized. Install openai and set XAI_API_KEY.")
        self.client = _XAI_CLIENT

    def list_models(self) -> Iterable[str]:
        try:
            return [getattr(m, 'id', None) or m.get('id') for m in self.client.models.list().data]
        except Exception as e:
            raise RuntimeError(f"Error listing models: {e}")

    def chat_with_tools(self, *, model: str, messages: List[Dict[str, Any]], temperature: Optional[float] = 0.2) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Handle chat completion with tool calls in a loop, using streaming for real-time output.
        Accumulates deltas for content (printed incrementally) and tool calls (processed after stream).
        Returns the final response content and updated messages.
        Falls back to non-streaming if streaming fails.
        """
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    temperature=temperature if temperature is not None else 0.2,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    stream=True,
                )

                # Accumulators for streaming
                response_message = {"role": "assistant", "content": ""}
                tool_calls = []

                for chunk in response:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta

                    if delta.role:
                        response_message["role"] = delta.role

                    if delta.content is not None:
                        response_message["content"] += delta.content
                        print(delta.content, end='', flush=True)

                    if delta.tool_calls:
                        for tool_delta in delta.tool_calls:
                            index = tool_delta.index
                            while len(tool_calls) <= index:
                                tool_calls.append({
                                    "id": "",
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                })
                            if tool_delta.id:
                                tool_calls[index]["id"] += tool_delta.id
                            if tool_delta.type:
                                tool_calls[index]["type"] = tool_delta.type
                            if tool_delta.function and tool_delta.function.name:
                                tool_calls[index]["function"]["name"] += tool_delta.function.name
                            if tool_delta.function and tool_delta.function.arguments:
                                tool_calls[index]["function"]["arguments"] += tool_delta.function.arguments

                print()  # Newline after stream

                if tool_calls:
                    response_message["tool_calls"] = tool_calls
                    messages.append(response_message)
                    for tc in tool_calls:
                        func_name = tc["function"]["name"]
                        try:
                            args = json.loads(tc["function"]["arguments"])
                        except json.JSONDecodeError:
                            args = {}
                            print(f"⚠️ Invalid arguments for {func_name}")
                        if func_name == "get_current_date":
                            result = datetime.utcnow().isoformat() + " UTC"
                        elif func_name == "web_search":
                            query = args.get("query")
                            try:
                                search_url = f"https://lite.duckduckgo.com/lite/?q={requests.utils.quote(query)}"
                                headers = {'User-Agent': 'Mozilla/5.0'}
                                resp = requests.get(search_url, headers=headers)
                                resp.raise_for_status()
                                soup = BeautifulSoup(resp.text, 'html.parser')
                                results = []
                                for row in soup.find_all('tr', {'valign': 'top'}):
                                    link = row.find('a', class_='result-link')
                                    if link:
                                        title = link.text.strip()
                                        href = link.get('href')
                                        snippet_td = row.find('td', class_='result-snippet')
                                        snippet = snippet_td.text.strip() if snippet_td else ""
                                        results.append(f"Title: {title}\nURL: {href}\nSnippet: {snippet}")
                                result = "\n\n".join(results[:5]) or "No results found."
                            except Exception as e:
                                result = f"Error during search: {str(e)}"
                        else:
                            result = "Unknown tool"
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "name": func_name,
                            "content": result,
                        })
                else:
                    return response_message["content"], messages

            except Exception as e:
                print(f"⚠️ Streaming failed: {e}. Falling back to non-streaming.")
                # Fallback to non-streaming
                try:
                    response = self.client.chat.completions.create(
                        model=model,
                        temperature=temperature if temperature is not None else 0.2,
                        messages=messages,
                        tools=TOOLS,
                        tool_choice="auto",
                        stream=False,
                    )
                    choice = response.choices[0]
                    msg = choice.message
                    if msg.content:
                        print(msg.content)
                        return msg.content, messages
                    # Handle tool calls similarly (non-streaming code from v6)
                    if msg.tool_calls:
                        tool_calls_dicts = [
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments
                                }
                            } for tc in msg.tool_calls
                        ]
                        messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": tool_calls_dicts})
                        for tc in msg.tool_calls:
                            func_name = tc.function.name
                            args = json.loads(tc.function.arguments)
                            if func_name == "get_current_date":
                                result = datetime.utcnow().isoformat() + " UTC"
                            elif func_name == "web_search":
                                query = args.get("query")
                                try:
                                    search_url = f"https://lite.duckduckgo.com/lite/?q={requests.utils.quote(query)}"
                                    headers = {'User-Agent': 'Mozilla/5.0'}
                                    resp = requests.get(search_url, headers=headers)
                                    resp.raise_for_status()
                                    soup = BeautifulSoup(resp.text, 'html.parser')
                                    results = []
                                    for row in soup.find_all('tr', {'valign': 'top'}):
                                        link = row.find('a', class_='result-link')
                                        if link:
                                            title = link.text.strip()
                                            href = link.get('href')
                                            snippet_td = row.find('td', class_='result-snippet')
                                            snippet = snippet_td.text.strip() if snippet_td else ""
                                            results.append(f"Title: {title}\nURL: {href}\nSnippet: {snippet}")
                                    result = "\n\n".join(results[:5]) or "No results found."
                                except Exception as e:
                                    result = f"Error during search: {str(e)}"
                            else:
                                result = "Unknown tool"
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "name": func_name,
                                "content": result,
                            })
                    else:
                        return "No response content.", messages
                except Exception as fallback_e:
                    raise RuntimeError(f"Chat error (fallback failed): {fallback_e}")

# =====================
# CLI
# =====================
class GrokCLI(cmd.Cmd):
    intro = (
        "\n" +
        "╔" + "═"*63 + "╗\n" +
        "║                 Grok Projects CLI (Enhanced, v7)               ║\n" +
        "║  Manage isolated project files and chat with Grok about them    ║\n" +
        "╠" + "═"*63 + "╣\n" +
        "║  📁 Projects are isolated - Grok sees ONLY current project      ║\n" +
        "╚" + "═"*63 + "╝\n\n" +
        "Type 'help' for commands or 'help <command>' for details.\n"
    )
    prompt = '(grok) > '

    MODELS: Dict[str, str] = {
        'grok': 'grok-beta',  # Updated alias for v7
        'grok4': 'grok-4-0709',
        'grok3': 'grok-3',
        'mini': 'grok-3-mini',
    }

    def __init__(self):
        super().__init__()
        self._already_exited = False
        self.client = None
        try:
            temp_client = GrokClient()
            # Validate API key by listing models
            temp_client.list_models()
            self.client = temp_client
        except Exception as e:
            print(f"❌ Error initializing Grok client: {e}")
            print("Please ensure XAI_API_KEY is set correctly. Obtain a valid key from https://console.x.ai.")
            self.client = None
        self.projects_base = Path.cwd() / "Grok_Projects"
        self.projects_base.mkdir(exist_ok=True)
        # Persistent CLI command history
        self.history_file = self.projects_base / ".cmd_history"
        self._max_history_len = 20000  # Increased for v7
        self._init_cmd_history()
        self.current_project: Optional[Project] = None
        self.current_model_key = 'grok'  # Updated default
        self._load_settings()
        if self.current_project:
            self.prompt = f'({self.current_project.name}) > '

    def _load_settings(self) -> None:
        settings_path = self.projects_base / ".settings.json"
        if settings_path.exists():
            try:
                s = json.loads(settings_path.read_text(encoding='utf-8'))
                last = s.get('last_project')
                if last and (self.projects_base / last).exists():
                    self.current_project = Project(last)
                self.current_model_key = s.get('model', self.current_model_key)
            except Exception:
                pass

    def _save_settings(self) -> None:
        settings_path = self.projects_base / ".settings.json"
        s = {
            'last_project': self.current_project.name if self.current_project else None,
            'model': self.current_model_key,
        }
        with open(settings_path, 'w', encoding='utf-8') as f:
            json.dump(s, f, indent=2)

    def _format_size(self, size: int) -> str:
        if size == 0:
            return "0 B"
        units = ["B", "KB", "MB", "GB", "TB"]
        s = float(size)
        i = 0
        while s >= 1024 and i < len(units) - 1:
            s /= 1024
            i += 1
        return f"{s:.1f} {units[i]}"

    def estimate_tokens(self, text: str) -> int:
        if _HAS_TIKTOKEN:
            try:
                enc = tiktoken.get_encoding("o200k_base")
                return len(enc.encode(text))
            except Exception:
                pass
        # fallback heuristic ~4 chars/token
        return max(1, int(len(text)/4))

    # ----- command history helpers -----
    def _init_cmd_history(self):
        if readline is None:
            return
        try:
            if hasattr(readline, "set_history_length"):
                readline.set_history_length(self._max_history_len)
            if self.history_file.exists():
                readline.read_history_file(str(self.history_file))
        except Exception:
            pass

    def _save_cmd_history(self):
        if readline is None:
            return
        try:
            # Trim oldest items if exceeding max
            if hasattr(readline, "get_current_history_length") and hasattr(readline, "remove_history_item"):
                length = readline.get_current_history_length()
                excess = max(0, length - self._max_history_len)
                for _ in range(excess):
                    try:
                        readline.remove_history_item(0)
                    except Exception:
                        break
            tmp = self.history_file.with_suffix(".tmp")
            readline.write_history_file(str(tmp))
            tmp.replace(self.history_file)
        except Exception:
            pass

    # ----- Completion helpers (new in v7) -----
    def complete_remove(self, text, line, begidx, endidx):
        if not self.current_project:
            return []
        files = [f[0] for f in self.current_project.list_files()]
        return [f for f in files if f.startswith(text)]

    def complete_view(self, text, line, begidx, endidx):
        return self.complete_remove(text, line, begidx, endidx)

    def complete_chat(self, text, line, begidx, endidx):
        # Basic completion for -s option
        if '-s ' in line and self.current_project:
            files = [f[0] for f in self.current_project.list_files()]
            return [f for f in files if f.startswith(text)]
        return []

    # =========================
    # Project management
    # =========================
    def do_create(self, name: str):
        """Create a new project: create <project_name>"""
        if not name:
            print("❌ Please provide a project name")
            return
        self.current_project = Project(name)
        self.prompt = f'({name}) > '
        print(f"✅ Created project: {name}")
        synced = self.current_project.sync_files()
        if synced:
            print(f"📂 Found and loaded {len(synced)} existing files in project directory")
        self._save_settings()

    def do_open(self, name: str):
        """Open an existing project: open <project_name>"""
        if not name:
            print("❌ Please provide a project name")
            return
        p = self.projects_base / name
        if not p.exists():
            print(f"❌ Project '{name}' not found")
            return
        self.current_project = Project(name)
        self.prompt = f'({name}) > '
        print(f"✅ Opened project: {name}")
        synced = self.current_project.sync_files()
        if synced:
            print(f"📂 Auto-loaded {len(synced)} files from project directory")
        self._save_settings()

    def do_projects(self, _):
        """List all projects"""
        projects = [p.name for p in self.projects_base.iterdir() if p.is_dir() and not p.name.startswith('.')]
        if projects:
            print("\n📁 Available projects:")
            for pr in sorted(projects):
                marker = "→" if self.current_project and self.current_project.name == pr else " "
                print(f" {marker} {pr}")
        else:
            print("No projects found. Create one with: create <name>")

    def do_add(self, filepath: str):
        """Add a file to the current project: add <filepath>"""
        if not self.current_project:
            print("❌ No project open. Use: open <project_name>")
            return
        if not filepath:
            print("❌ Please provide a file path")
            return
        try:
            path = Path(filepath).expanduser()
            filename = self.current_project.add_file(path)
            print(f"✅ Added: {filename}")
        except Exception as e:
            print(f"❌ Error: {e}")

    def do_files(self, _):
        """List all files in the current project"""
        if not self.current_project:
            print("❌ No project open")
            return
        files = self.current_project.list_files()
        if files:
            print(f"\n📄 Files in {self.current_project.name}:")
            print(f"{'Name':<40} {'Size':>10} {'Modified'}")
            print("-"*65)
            for name, size, modified in files:
                print(f"{name:<40} {self._format_size(size):>10} {modified}")
        else:
            print("No files in project. Add with: add <filepath>")

    def do_remove(self, filename: str):
        """Remove a file from the project: remove <filename>"""
        if not self.current_project:
            print("❌ No project open")
            return
        if not filename:
            print("❌ Please provide a filename")
            return
        if self.current_project.remove_file(filename):
            print(f"✅ Removed: {filename}")
        else:
            print(f"❌ File not found: {filename}")

    def do_update(self, _):
        """Sync metadata with files directory"""
        if not self.current_project:
            print("❌ No project open")
            return
        synced = self.current_project.sync_files()
        if synced:
            print(f"✅ Loaded {len(synced)} new files:")
            for fn in synced:
                print(f"   + {fn}")
        else:
            print("✅ All files are up to date")
        print(f"\n📄 Total files in project: {len(self.current_project.list_files())}")

    # =========================
    # Viewing & tokens
    # =========================
    def do_view(self, filename: str):
        """View contents of a file: view <filename>"""
        if not self.current_project:
            print("❌ No project open")
            return
        if not filename:
            print("❌ Please provide a filename")
            return
        try:
            content = self.current_project.get_file_content(filename)
            print("\n"+"="*70)
            print(f"📄 File: {filename}")
            print("="*70+"\n")
            print(content)
            print("\n"+"="*70)
            print(f"📊 Estimated tokens: {self.estimate_tokens(content):,}")
        except Exception as e:
            print(f"❌ Error: {e}")

    def do_tokens(self, _):
        """Show token estimates for files in the project"""
        if not self.current_project:
            print("❌ No project open")
            return
        print(f"\n📊 Token Analysis: {self.current_project.name}")
        print("="*70)
        total = 0
        file_rows = []
        for p in sorted(self.current_project.files_path.iterdir()):
            if p.is_file():
                try:
                    content = self.current_project.get_file_content(p.name)
                    toks = self.estimate_tokens(content)
                    total += toks
                    file_rows.append((p.name, toks, len(content)))
                except Exception:
                    file_rows.append((p.name, 0, 0))
        file_rows.sort(key=lambda x: x[1], reverse=True)
        print(f"\n{'File':<40} {'Tokens':>12} {'Size':>10}")
        print("-"*65)
        for fn, toks, size in file_rows:
            icon = "🔴" if toks>50_000 else ("🟡" if toks>20_000 else "🟢")
            print(f"{icon} {fn:<38} {toks:>11,} {self._format_size(size):>10}")
        print("-"*65)
        overhead = 2_000
        print(f"TOTAL with ~overhead: {(total+overhead):,} / {MAX_INPUT_TOKENS:,} input ({MAX_TOTAL_TOKENS:,} total)")

    # =========================
    # Models
    # =========================
    def do_models(self, _):
        """List models (static aliases + live from API)"""
        print("\n🤖 Model aliases:")
        print(f"{'Alias':<10} {'Model ID':<20} {'Current'}")
        print("-"*45)
        for k, v in self.MODELS.items():
            mark = "✓" if k == self.current_model_key else ""
            print(f"{k:<10} {v:<20} {mark}")
        if self.client:
            try:
                api_models = list(self.client.list_models())
                if api_models:
                    print("\nFrom API (abbrev):")
                    for mid in sorted(api_models)[:30]:
                        print("  -", mid)
            except Exception as e:
                print(f"(Model listing error: {e})")

    def do_model(self, name: str):
        """Switch model by alias or exact id: model <grok|grok4|grok3|mini|model-id>"""
        if not name:
            print(f"Current: {self.MODELS.get(self.current_model_key, self.current_model_key)}")
            print("Aliases:", ", ".join(self.MODELS.keys()))
            return
        if name in self.MODELS:
            self.current_model_key = name
            self._save_settings()
            print(f"✅ Switched to {name} ({self.MODELS[name]})")
        else:
            # accept exact id
            self.MODELS['custom'] = name
            self.current_model_key = 'custom'
            self._save_settings()
            print(f"✅ Using custom model id: {name}")

    # =========================
    # Chat (with strict project scope)
    # =========================
    def do_chat(self, message: str):
        """Chat with Grok about your project files.

Usage:
  chat <message>                    - Include all files (if under limit)
  chat -n <message>                 - No file context
  chat -s file1,file2 -- <message>  - Include only specific files
  chat -f <filename>                - Read message from a file
  chat -m                           - Multi-line input (end with 'EOF')
"""
        if self.client is None:
            print("❌ Invalid or missing XAI_API_KEY. Please set a valid key from https://console.x.ai and restart.")
            return
        if not message:
            print("❌ Provide a message. See 'help chat'")
            return

        include_context = True
        selected_files: Optional[List[str]] = None

        if message.startswith("-n "):
            include_context = False
            message = message[3:]
            print("📄 Running without file context")
        elif message.startswith("-s "):
            parts = message[3:].split(" -- ", 1)
            if len(parts) != 2:
                print("❌ Usage: chat -s file1,file2 -- Your message")
                return
            selected_files = [s.strip() for s in parts[0].split(',') if s.strip()]
            message = parts[1]
            print(f"📎 Including only: {', '.join(selected_files)}")
        elif message.startswith("-f "):
            fn = message[3:].strip()
            if not self.current_project:
                print("❌ No project open")
                return
            try:
                message = self.current_project.get_file_content(fn)
                print(f"📄 Read message from: {fn}")
            except Exception as e:
                print(f"❌ {e}")
                return
        elif message.strip() == "-m":
            print("📝 Multi-line mode. End with 'EOF' on its own line.")
            lines: List[str] = []
            try:
                while True:
                    line = input("... ")
                    if line.strip() == "EOF":
                        break
                    lines.append(line)
            except KeyboardInterrupt:
                print("\n❌ Cancelled")
                return
            message = "\n".join(lines)
            if not message.strip():
                print("❌ No message provided")
                return

        # Build project context
        context = ""
        context_tokens = 0
        if include_context and self.current_project:
            if selected_files:
                ctx_parts = [f"Project: {self.current_project.name} (Selected Files)\n{'='*60}\n"]
                included = []
                for fn in selected_files:
                    try:
                        content = self.current_project.get_file_content(fn)
                        sec = f"\n--- File: {fn} ---\n{content}\n"
                        tok = self.estimate_tokens(sec)
                        if context_tokens + tok + self.estimate_tokens(message) > MAX_INPUT_TOKENS - 5_000:
                            print(f"⚠️  Skipping {fn} (would exceed input budget)")
                            continue
                        ctx_parts.append(sec)
                        context_tokens += tok
                        included.append(fn)
                    except Exception as e:
                        print(f"⚠️  Could not include {fn}: {e}")
                context = "".join(ctx_parts)
                print(f"📎 Included {len(included)} files (~{context_tokens:,} tokens)")
            else:
                context = self.current_project.get_project_context()
                context_tokens = self.estimate_tokens(context)
                if context_tokens + self.estimate_tokens(message) > MAX_INPUT_TOKENS - 5_000:
                    print(f"⚠️  Project files exceed input budget (~{context_tokens:,})")
                    print("   Use: chat -n <msg>  or  chat -s file1,file2 -- <msg>")
                    return
                print(f"📎 Including {len(self.current_project.list_files())} files (~{context_tokens:,} tokens)")
        else:
            if not include_context:
                print("📄 No file context included")
            elif not self.current_project:
                print("⚠️  No project open - chatting without file context")

        message_tokens = self.estimate_tokens(message)
        total_est = context_tokens + message_tokens + 2000
        print(f"📊 Estimated tokens: {total_est:,} / {MAX_INPUT_TOKENS:,} input ({MAX_TOTAL_TOKENS:,} total)")
        if total_est > MAX_INPUT_TOKENS - 1_000:
            print("❌ Request too large (input tokens). Reduce context or message size.")
            return

        # Compose final prompt
        if context:
            user_text = (
                "I'm sharing files from my project. These are ALL and ONLY the files in my current project:\n\n"
                + context + "\n\nMy question/request: " + message
            )
        else:
            user_text = message

        model_id = self.MODELS.get(self.current_model_key, self.current_model_key)
        system = SYSTEM_INSTRUCTIONS + " " + GrokClient.WEBSITE_STYLE

        print("🤔 Thinking (tools enabled, streaming)...")
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user_text}]
        try:
            response_text, full_messages = self.client.chat_with_tools(model=model_id, messages=messages)
        except Exception as e:
            print(f"\n❌ Error: {e}")
            traceback.print_exc()
            return

        # Extract artifacts
        artifacts = []
        if self.current_project:
            am = ArtifactManager(self.current_project)
            artifacts = am.extract_code_blocks(response_text)
            if artifacts:
                print(f"📦 Created {len(artifacts)} code artifacts:")
                for a in artifacts:
                    print(f"   ✓ {a['title']} ({a['language']}) → {a['filename']}")

            # Build a compact context summary for the consolidated history
            context_info = {
                "model": model_id,
                "context_tokens": context_tokens,
                "message_tokens": message_tokens,
                "total_est_tokens": context_tokens + message_tokens,
                "included_files": [],
            }
            if include_context and self.current_project:
                if selected_files:
                    context_info["included_files"] = selected_files
                else:
                    context_info["included_files"] = [fn for (fn, _, _) in self.current_project.list_files()]

            conv_file = self.current_project.save_conversation(
                full_messages,
                response_text,
                artifacts,
                context=context_info,
            )
            print(f"💾 Conversation saved: {conv_file}")

        print("\n📊 Response Metrics:")
        print(f"   • Length: {len(response_text)} characters")
        print(f"   • Words: {len(response_text.split())} words")
        print(f"   • Code artifacts: {len(artifacts)}")

    # =========================
    # Utilities
    # =========================
    def do_artifacts(self, _):
        """List artifacts in current project"""
        if not self.current_project:
            print("❌ No project open")
            return
        arts = list(self.current_project.artifacts_path.iterdir())
        if arts:
            print(f"\n📦 Artifacts in {self.current_project.name}:")
            print(f"{'Name':<50} {'Size':>10} {'Modified'}")
            print("-"*75)
            for a in sorted(arts):
                if a.is_file():
                    print(f"{a.name:<50} {self._format_size(a.stat().st_size):>10} {datetime.fromtimestamp(a.stat().st_mtime).strftime('%Y-%m-%d %H:%M')}")
            print(f"\nTotal: {len(arts)} artifacts")
        else:
            print("No artifacts yet")

    def do_list(self, _):
        """List files and artifacts"""
        if not self.current_project:
            print("❌ No project open")
            return
        self.do_files("")
        self.do_artifacts("")

    def do_export(self, args: str):
        """Export artifacts or files: export artifacts <path> | export files <path>"""
        if not self.current_project:
            print("❌ No project open")
            return
        parts = args.split(maxsplit=1)
        if not parts:
            print("❌ Usage: export artifacts [path] | export files [path]")
            return
        kind = parts[0]
        dest = Path(parts[1]).expanduser() if len(parts) > 1 else Path.cwd()/f"exported_{kind}"
        if kind == 'artifacts':
            src = self.current_project.artifacts_path
        elif kind == 'files':
            src = self.current_project.files_path
        else:
            print("❌ Usage: export artifacts [path] | export files [path]")
            return
        if not list(src.iterdir()):
            print(f"No {kind} to export")
            return
        dest.mkdir(parents=True, exist_ok=True)
        n = 0
        for it in src.iterdir():
            if it.is_file():
                shutil.copy2(it, dest)
                n += 1
        print(f"✅ Exported {n} {kind} to: {dest.absolute()}")

    def do_clear_artifacts(self, _):
        """Delete all artifacts (with confirmation)"""
        if not self.current_project:
            print("❌ No project open")
            return
        arts = [a for a in self.current_project.artifacts_path.iterdir() if a.is_file()]
        if not arts:
            print("No artifacts to clear")
            return
        print(f"⚠️  This will delete {len(arts)} artifacts from {self.current_project.name}")
        if input("Are you sure? (yes/no): ").strip().lower() == 'yes':
            for a in arts:
                a.unlink()
            print("✅ Removed artifacts")
        else:
            print("❌ Cancelled")

    def do_conversations(self, _):
        """List recent conversations"""
        if not self.current_project:
            print("❌ No project open")
            return
        conv = self.current_project.list_conversations()
        if conv:
            print(f"\n💬 Conversations in {self.current_project.name}:")
            for fn, ts in conv[:10]:
                print(f"  {fn} - {ts}")
            if len(conv) > 10:
                print(f"  ... and {len(conv)-10} more")
        else:
            print("No conversations yet. Start with: chat <message>")

    def do_history(self, args: str):
        """
        Show consolidated project chat history (from conversations/project_history.json).

        Usage:
          history            -> show last 10 entries
          history 25         -> show last 25 entries
        """
        if not self.current_project:
            print("❌ No project open")
            return

        try:
            limit = 10
            if args.strip():
                limit = max(1, int(args.strip()))
        except Exception:
            print("❌ Usage: history [N]")
            return

        hist = self.current_project.history or {}
        entries = hist.get("entries", [])
        if not entries:
            print("No history yet for this project.")
            return

        shown = min(limit, len(entries))
        print(f"\n🕘 Project history for {self.current_project.name} (showing last {shown} of {len(entries)}):")
        print("-"*90)
        start_index = len(entries) - shown
        for i, entry in enumerate(entries[start_index:], start=start_index+1):
            ts = entry.get("timestamp", "Unknown")
            msgs = entry.get("messages", [])
            prompt = ""
            for m in msgs:
                if m.get("role") == "user":
                    prompt = (m.get("content") or "").strip().split("\n", 1)[0]
                    break
            resp = entry.get("response", {})
            words = resp.get("word_count", 0)
            artifacts = entry.get("artifacts", [])
            ctx = entry.get("context", {})
            model = ctx.get("model", "unknown")
            included_files = ctx.get("included_files", [])
            print(f"[{i}] {ts} | model={model} | words={words} | artifacts={len(artifacts)}")
            if prompt:
                print(f"     prompt: {prompt[:120]}{'…' if len(prompt)>120 else ''}")
            if included_files:
                preview = ", ".join(included_files[:3])
                more = f" (+{len(included_files)-3} more)" if len(included_files) > 3 else ""
                print(f"     files: {preview}{more}")
        print("-"*90)
        print(f"📄 Full file: {self.current_project.history_path}")

    def do_open_project_folder(self, _):
        """Open current project folder in system file manager"""
        if not self.current_project:
            print("❌ No project open")
            return
        p = self.current_project.project_path
        import platform, subprocess
        try:
            if platform.system() == 'Darwin':
                subprocess.run(['open', str(p)])
            elif platform.system() == 'Windows':
                subprocess.run(['explorer', str(p)])
            else:
                subprocess.run(['xdg-open', str(p)])
            print(f"✅ Opened: {p}")
        except Exception as e:
            print(f"❌ Could not open folder: {e}\n📁 {p}")

    def do_help_chat(self, _):
        print(
            """
💬 Chat Command Options
=======================
  chat <message>                    - Chat with all project files included
  chat -n <message>                 - No file context
  chat -s file1,file2 -- <message>  - Include specific files only
  chat -f <filename>                - Read message from a file in your project
  chat -m                           - Multi-line mode (end with 'EOF')

Tips:
  • Use 'tokens' to see largest files
  • If you hit limits, try -n or -s
  • Artifacts (code blocks) are saved automatically
  • Tools enabled for real-time info
"""
        )

    def do_help(self, arg):
        if arg:
            return super().do_help(arg)
        print(
            """
🎯 Grok Projects CLI - Command Reference
==========================================
PROJECTS:
  create <name>          - Create new project
  open <name>            - Open existing project
  projects               - List projects
  summary                - Project statistics

FILES:
  add <path>             - Add file
  files                  - List files
  remove <filename>      - Remove file
  update                 - Sync metadata
  view <filename>        - View content (+ tokens)
  tokens                 - Token analysis

CHAT & AI:
  chat <message>         - Chat (with context and tools)
  chat -n/-s/-f/-m       - Context control options
  model <alias|id>       - Switch model
  models                 - Show aliases + API models

ARTIFACTS & EXPORT:
  artifacts              - List artifacts
  export artifacts <p>   - Export artifacts
  export files <p>       - Export files
  clear_artifacts        - Delete all artifacts

META:
  conversations          - List chat history (per-call files)
  history [N]            - Show last N consolidated entries (default 10)
  open_project_folder    - Open in file manager
  clear                  - Clear screen
  exit / quit            - Exit CLI

Quick start:
  create my_project
  add path/to/file.py
  tokens
  chat Analyze the codebase and propose improvements

Batch mode:
  python GrokProjects_v7.py my_instructions.txt
  # file lines like:
  # create project_name
  # open project_name
  # chat "some text"

Note: For API pricing, visit https://x.ai/api
"""
        )

    def do_summary(self, _):
        if not self.current_project:
            print("❌ No project open")
            return
        files = self.current_project.list_files()
        arts = [a for a in self.current_project.artifacts_path.iterdir() if a.is_file()]
        conv = self.current_project.list_conversations()
        total_file_size = sum(s for _, s, _ in files)
        total_art_size = sum(a.stat().st_size for a in arts)
        print(f"\n📊 Project Summary: {self.current_project.name}")
        print("="*60)
        print(f"📄 Files: {len(files)} ({self._format_size(total_file_size)})")
        print(f"📦 Artifacts: {len(arts)} ({self._format_size(total_art_size)})")
        print(f"💬 Conversations: {len(conv)}")
        print(f"📁 Location: {self.current_project.project_path.absolute()}")

    def do_clear(self, _):
        os.system('clear' if os.name=='posix' else 'cls')

    def do_exit(self, _):
        if not self._already_exited:
            self._already_exited = True
            self._save_settings()
            try:
                self._save_cmd_history()
            except Exception:
                pass
            print("👋 Goodbye!")
        return True

    def do_quit(self, arg):
        return self.do_exit(arg)

    def emptyline(self):
        pass


def main():
    print("\n🚀 Starting Grok Projects CLI (Enhanced, v7)\n")
    cli = GrokCLI()

    # If a single positional argument is provided, treat it as an instruction file.
    if len(sys.argv) > 1:
        # Support a common form: python GrokProjects_v7.py my_instructions.txt
        script_path = sys.argv[1]
        try:
            run_instruction_file(cli, Path(script_path), stop_on_error=True, echo=True)
        except SystemExit:
            # Allow 'exit' in scripts to stop early without a stack trace
            pass
        except Exception as e:
            print(f"❌ Aborted due to error while running instructions: {e}")
        finally:
            # Ensure settings persist after batch run, but avoid double goodbye
            if not getattr(cli, '_already_exited', False):
                try:
                    cli.do_exit("")
                except Exception:
                    pass
        return

    try:
        cli.cmdloop()
    except KeyboardInterrupt:
        try:
            cli._save_cmd_history()
        except Exception:
            pass
        print("\n\n👋 Goodbye!")


if __name__ == "__main__":
    main()

