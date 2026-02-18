"""Shared utilities for the orchestrator, planning assistant, and tool scripts.

Centralises TTY handling, multi-line input reading, intent detection,
workflow phase tracking, execution-completion signalling, and text
analysis utilities so that all modules stay in sync.
"""

import sys
import re
import atexit
import asyncio

try:
    import readline  # noqa: F401 — enables line-editing/history for input()
except ImportError:
    readline = None

try:
    import termios
except ImportError:
    termios = None

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.key_binding import KeyBindings
except ImportError:
    PromptSession = None
    KeyBindings = None

from enum import Enum, auto


# -------------------------------------------------------------------------
# Phase Enum
# -------------------------------------------------------------------------

class Phase(Enum):
    """High-level orchestrator workflow phases."""
    COLLECT_SAMPLE = auto()
    GATHER_INFO    = auto()
    EXEC_FAILED    = auto()
    DONE           = auto()


# -------------------------------------------------------------------------
# TTY State Management
# -------------------------------------------------------------------------

_ORIGINAL_TTY_ATTRS = None
if termios is not None and sys.stdin.isatty():
    try:
        _ORIGINAL_TTY_ATTRS = termios.tcgetattr(sys.stdin.fileno())
    except termios.error:
        _ORIGINAL_TTY_ATTRS = None


def restore_tty_state() -> None:
    """Restore stdin terminal mode if another component left it altered."""
    if termios is None or _ORIGINAL_TTY_ATTRS is None or not sys.stdin.isatty():
        return
    try:
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, _ORIGINAL_TTY_ATTRS)
    except termios.error:
        pass


atexit.register(restore_tty_state)


# -------------------------------------------------------------------------
# Prompt Session (Multiline Editing)
# -------------------------------------------------------------------------

_PROMPT_SESSION = None
_PROMPT_KEY_BINDINGS = None

if KeyBindings is not None:
    _PROMPT_KEY_BINDINGS = KeyBindings()

    @_PROMPT_KEY_BINDINGS.add("enter")
    def _newline_or_submit(event) -> None:
        """Insert newline, or submit on a trailing blank line."""
        buffer = event.current_buffer
        document = buffer.document
        if document.is_cursor_at_the_end and document.current_line == "":
            buffer.validate_and_handle()
            return
        buffer.insert_text("\n")

    @_PROMPT_KEY_BINDINGS.add("c-d")
    def _submit_multiline(event) -> None:
        """Submit current multiline buffer."""
        event.current_buffer.validate_and_handle()


def _get_prompt_session():
    """Lazily create prompt_toolkit session when running in an interactive TTY."""
    global _PROMPT_SESSION
    if PromptSession is None:
        return None
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return None
    if _PROMPT_SESSION is None:
        _PROMPT_SESSION = PromptSession(key_bindings=_PROMPT_KEY_BINDINGS)
    return _PROMPT_SESSION


def _has_running_event_loop() -> bool:
    """Return True when called from a thread with an active asyncio event loop."""
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


# -------------------------------------------------------------------------
# Execution Tracking
# -------------------------------------------------------------------------

_execution_completed = False
_last_worker_context = ""
_last_worker_run_state: dict[str, object] = {}


def mark_execution_completed() -> None:
    """Signal that worker execution finished (called by worker_agent)."""
    global _execution_completed
    _execution_completed = True


def check_and_clear_execution_flag() -> bool:
    """Return True (once) if execution completed since last check."""
    global _execution_completed
    if _execution_completed:
        _execution_completed = False
        return True
    return False


def set_last_worker_context(context: str) -> None:
    """Persist last worker execution context for recovery flows."""
    global _last_worker_context
    _last_worker_context = (context or "").strip()


def get_last_worker_context() -> str:
    """Return the latest worker execution context."""
    return _last_worker_context


def clear_last_worker_context() -> None:
    """Clear stored worker execution context."""
    global _last_worker_context
    _last_worker_context = ""


def set_last_worker_run_state(state: dict) -> None:
    """Persist worker run state (process-local)."""
    global _last_worker_run_state
    _last_worker_run_state = dict(state or {})


def get_last_worker_run_state() -> dict:
    """Fetch worker run state (process-local)."""
    return dict(_last_worker_run_state)


def clear_last_worker_run_state() -> None:
    """Clear worker run state."""
    global _last_worker_run_state
    _last_worker_run_state = {}


# -------------------------------------------------------------------------
# Input Helpers
# -------------------------------------------------------------------------

def read_multiline_input() -> str:
    """Read multiline user input with full in-buffer editing when available."""
    restore_tty_state()
    print("\nYou:")
    session = _get_prompt_session()
    if session is not None:
        try:
            text = session.prompt(
                "> ",
                multiline=True,
                bottom_toolbar="Enter: newline (blank line submits)  |  Ctrl-D: submit",
                prompt_continuation=lambda *_: ". ",
                in_thread=_has_running_event_loop(),
            )
        except EOFError:
            return ""
        return text.strip()

    lines: list[str] = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    return "\n".join(lines).strip()


# -------------------------------------------------------------------------
# Intent Detection
# -------------------------------------------------------------------------

_NEW_REQUEST_PHRASES = (
    "new conversation",
    "new request",
    "start over",
    "restart",
    "switch topic",
    "different topic",
    "forget previous",
    "ignore previous",
    "help me create",
    "help me build",
    "i want to build",
    "i want to create",
)

_EXECUTION_INTENT_PHRASES = (
    "proceed",
    "go ahead",
    "setup opensearch",
    "set up opensearch",
    "index data",
    "index this dataset",
    "implement",
)

_CANCEL_PHRASES = (
    "cancel",
    "nevermind",
    "never mind",
    "abort",
)

_CLEANUP_PHRASES = (
    "cleanup",
    "clean up",
    "remove verification",
    "delete verification",
    "remove sample docs",
    "delete sample docs",
    "clear test docs",
    "clear verification docs",
)

_WORKER_RETRY_PHRASES = (
    "retry",
    "resume",
    "redo worker",
    "retry failed step",
    "continue from failure",
    "redo index_verification_docs",
    "retry verification",
    "reindex verification docs",
    "redo verification docs",
    "rerun index_verification_docs",
)


def looks_like_new_request(user_input: str) -> bool:
    """Detect if user input indicates a brand-new / unrelated request."""
    text = user_input.lower()
    return any(phrase in text for phrase in _NEW_REQUEST_PHRASES)


def looks_like_execution_intent(user_input: str) -> bool:
    """Detect if user wants to proceed with execution."""
    text = user_input.lower()
    return any(phrase in text for phrase in _EXECUTION_INTENT_PHRASES)


def looks_like_cancel(user_input: str) -> bool:
    """Detect if user wants to cancel the current flow."""
    text = user_input.lower()
    return any(phrase in text for phrase in _CANCEL_PHRASES)


def looks_like_cleanup_request(user_input: str) -> bool:
    """Detect if user asks to clean verification/sample docs."""
    text = user_input.lower()
    return any(phrase in text for phrase in _CLEANUP_PHRASES)


def looks_like_worker_retry(user_input: str) -> bool:
    """Detect if user asks to resume/retry worker execution after failure."""
    text = user_input.lower()
    return any(phrase in text for phrase in _WORKER_RETRY_PHRASES)


def looks_like_url_message(user_input: str) -> bool:
    """Detect if user input contains an HTTP/HTTPS URL."""
    text = user_input.strip()
    if not text:
        return False
    return re.search(r"https?://[^\s]+", text) is not None


def looks_like_local_path_message(user_input: str) -> bool:
    """Detect if user input contains a local file path."""
    text = user_input.strip()
    if not text:
        return False
    if "http://" in text or "https://" in text:
        return False
    if "~/" in text:
        return True
    if re.search(r"/[A-Za-z0-9._-]+", text):
        return True
    if re.search(r"\.(tsv|csv|json|jsonl|txt)\b", text.lower()):
        return True
    return False


# -------------------------------------------------------------------------
# Text Analysis Utilities
# -------------------------------------------------------------------------

def normalize_text(value: object) -> str:
    """Normalize any value into a compact single-line string with collapsed whitespace.

    Examples:
        "  hello   \\n\\t  world  " -> "hello world"
        "foo\\nbar\\nbaz"           -> "foo bar baz"
        123.45                      -> "123.45" (numbers converted to string first)
    """
    # str(value): convert input to string (handles numbers, None, etc.)
    # .split(): split on any whitespace (spaces, tabs, newlines), discarding empty tokens
    # " ".join(...): rejoin tokens with single spaces
    # .strip(): remove leading and trailing whitespace
    return " ".join(str(value).split()).strip()


def value_shape(text: str) -> dict[str, object]:
    """Compute structural characteristics of a text value.

    Returns a dict describing the value's shape — token breakdown, character
    composition ratios, and heuristic flags — so callers can classify or score
    values without reimplementing the same logic.

    Args:
        text: The raw text value to analyse.

    Returns:
        A dict with the following keys:

        - ``"text"`` (str): Whitespace-normalised text.
        - ``"length"`` (int): Length of the normalised text.
        - ``"tokens"`` (list[str]): Alphanumeric tokens extracted via regex.
        - ``"token_count"`` (int): Number of tokens.
        - ``"alpha_ratio"`` (float): Fraction of characters that are alphabetic.
        - ``"digit_ratio"`` (float): Fraction of characters that are digits.
        - ``"looks_numeric"`` (bool): True if value matches a numeric pattern.
        - ``"looks_date"`` (bool): True if value matches a date-like pattern.
    """
    compact = normalize_text(text)
    tokens = re.findall(r"[A-Za-z0-9_]+", compact)
    alpha_count = sum(1 for ch in compact if ch.isalpha())
    digit_count = sum(1 for ch in compact if ch.isdigit())
    length = len(compact)
    alpha_ratio = (alpha_count / length) if length else 0.0
    digit_ratio = (digit_count / length) if length else 0.0
    return {
        "text": compact,
        "length": length,
        "tokens": tokens,
        "token_count": len(tokens),
        "alpha_ratio": alpha_ratio,
        "digit_ratio": digit_ratio,
        "looks_numeric": bool(re.fullmatch(r"[+-]?\d+(\.\d+)?", compact)),
        "looks_date": bool(
            re.fullmatch(r"\d{4}([-/]\d{1,2}([-/]\d{1,2})?)?", compact)
        ),
    }


def text_richness_score(value: str) -> float:
    """Score a string by how rich it is as human-readable text.

    Higher scores indicate multi-word, mostly-alphabetic content —
    the kind of text most useful as search suggestions, language
    detection samples, or display previews.  Numeric strings, short
    codes, and IDs score low or zero.

    Uses ``value_shape`` internally so the scoring criteria are
    consistent across all callers.

    Args:
        value: The raw string to score.

    Returns:
        A non-negative float score.  Typical ranges:

        - 0.0 for empty / purely numeric / very short strings
        - 10–30 for single-word alphabetic strings
        - 40+ for multi-word descriptive text
    """
    shape = value_shape(value)
    length = int(shape["length"])
    if length < 2:
        return 0.0
    alpha_ratio = float(shape["alpha_ratio"])
    token_count = int(shape["token_count"])
    return (
        alpha_ratio * 20.0               # mostly letters, not codes/IDs
        + token_count * 10.0             # multi-word text is more descriptive
        + min(length, 100) / 100.0 * 5.0 # reasonable length bonus (max 5.0)
    )
