from rich.console import Console
from rich.rule import Rule
from rich.text import Text
from rich.theme import Theme

# AGENT_THEME = Theme(
#     {
#         # General - Softer grays instead of harsh cyan/white
#         "info": "deep_sky_blue3",
#         "warning": "dark_orange",
#         "error": "salmon1", # A softer red
#         "success": "spring_green3",
#         "dim": "grey62",
#         "muted": "grey46",
#         "border": "grey30",
#         "highlight": "plum3", # A gentle purple instead of cyan
        
#         # Roles - Warmer colors for a personal feel
#         "user": "steel_blue1", 
#         "assistant": "navajo_white1", # A cozy, off-white/beige
        
#         # Tools - Pastel-like shades
#         "tool": "medium_purple3",
#         "tool.read": "cadet_blue",
#         "tool.write": "dark_khaki",
#         "tool.shell": "rosy_brown",
#         "tool.network": "sky_blue2",
#         "tool.memory": "dark_sea_green3",
#         "tool.mcp": "light_steel_blue",
        
#         # Code / blocks
#         "code": "ivory3",
#     }
# )

AGENT_THEME = Theme(
    {
        # General
        "info": "cyan",
        "warning": "yellow",
        "error": "bright_red bold",
        "success": "green",
        "dim": "dim",
        "muted": "grey50",
        "border": "grey35",
        "highlight": "bold cyan",
        # Roles
        "user": "bright_blue bold",
        "assistant": "bright_white",
        # Tools
        "tool": "bright_magenta bold",
        "tool.read": "cyan",
        "tool.write": "yellow",
        "tool.shell": "magenta",
        "tool.network": "bright_blue",
        "tool.memory": "green",
        "tool.mcp": "bright_cyan",
        # Code / blocks
        "code": "white",
    }
)


_console : Console | None = None

def get_console() -> Console: 

    global _console
    if _console is None:
        _console = Console(theme=AGENT_THEME, highlight=False)

    return _console


class TUI:

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or get_console()
        self._assistant_stream_open = False

    def stream_assistant_delta (self, content: str) -> None:
        self.console.print(content, end='', markup=False)

    def begin_assistant (self) -> None:
        self.console.print()
        self.console.print(Rule(Text("Assistant", style="assistant")))
        self._assistant_stream_open = True

    def close_assistant (self):
        if self._assistant_stream_open:
            self.console.print()
        self._assistant_stream_open = False 