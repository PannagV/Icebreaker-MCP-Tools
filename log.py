"""
Stylized console logger for MCP servers.
Uses ANSI escape codes for colored, structured debug output.
"""
import sys
import time
from datetime import datetime

# ── ANSI Color Codes ──────────────────────────────────────────
RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"

GREEN   = "\033[38;2;0;255;136m"     # neon green  — incoming requests
WHITE   = "\033[97m"                 # bright white — service responses
CYAN    = "\033[38;2;0;200;255m"     # cyan         — info / lifecycle
YELLOW  = "\033[38;2;255;200;0m"     # amber        — warnings
RED     = "\033[38;2;255;60;60m"     # red          — errors
GRAY    = "\033[38;2;100;100;100m"   # dark gray    — separators / timing

# ── Box-drawing characters ────────────────────────────────────
TOP     = "╭"
MID     = "│"
BOT     = "╰"
LINE    = "─"

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]

def _write(msg: str):
    """Write to stderr so MCP stdio transport is not polluted."""
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()

# ── Public API ────────────────────────────────────────────────

def banner(server_name: str, detail: str = ""):
    """Print a startup banner."""
    _write("")
    _write(f"{CYAN}{BOLD}{'━' * 52}{RESET}")
    _write(f"{CYAN}{BOLD}  ⚡ {server_name}{RESET}")
    if detail:
        _write(f"{CYAN}  {detail}{RESET}")
    _write(f"{CYAN}{BOLD}{'━' * 52}{RESET}")
    _write("")

def info(msg: str):
    _write(f"{GRAY}[{_ts()}]{RESET} {CYAN}ℹ {msg}{RESET}")

def warn(msg: str):
    _write(f"{GRAY}[{_ts()}]{RESET} {YELLOW}⚠ {msg}{RESET}")

def error(msg: str):
    _write(f"{GRAY}[{_ts()}]{RESET} {RED}✖ {msg}{RESET}")

def request(tool_name: str, params: dict):
    """Log an incoming tool call (green)."""
    _write("")
    _write(f"{GRAY}[{_ts()}]{RESET} {GREEN}{BOLD}{TOP}{LINE} REQUEST ▸ {tool_name}{RESET}")
    for k, v in params.items():
        val = str(v) if v is not None else f"{DIM}(none){RESET}"
        _write(f"         {GREEN}{MID}  {BOLD}{k}:{RESET} {GREEN}{val}{RESET}")
    _write(f"         {GREEN}{BOT}{LINE}{LINE}{LINE}{RESET}")

def response(tool_name: str, summary: str, elapsed_ms: float = None):
    """Log a service response (white)."""
    time_str = f" ({elapsed_ms:.0f}ms)" if elapsed_ms is not None else ""
    _write(f"{GRAY}[{_ts()}]{RESET} {WHITE}{BOLD}{TOP}{LINE} RESPONSE ◂ {tool_name}{DIM}{time_str}{RESET}")
    # Truncate long summaries for readability
    lines = summary.split("\n")
    for line in lines[:6]:
        _write(f"         {WHITE}{MID}  {line[:120]}{RESET}")
    if len(lines) > 6:
        _write(f"         {WHITE}{MID}  {DIM}... ({len(lines) - 6} more lines){RESET}")
    _write(f"         {WHITE}{BOT}{LINE}{LINE}{LINE}{RESET}")
    _write("")

class Timer:
    """Context manager to measure elapsed time in ms."""
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    def __exit__(self, *_):
        self.elapsed_ms = (time.perf_counter() - self.start) * 1000
