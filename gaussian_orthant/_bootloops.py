"""Load the real BootLoops engines from the user's local checkout."""
import os
import sys
from pathlib import Path


def engines():
    root = Path(os.environ.get("BOOTLOOPS_ROOT", Path(__file__).resolve().parents[1] / "bootloops"))
    if not (root / "tools" / "wayfinder" / "transport.py").is_file():
        raise ImportError("BootLoops checkout missing; set BOOTLOOPS_ROOT to its repository directory")
    tools = str(root / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    from wayfinder.transport import transport_fixed_eps
    from wayfinder.quad import quad_refine
    from wayfinder.ratfun import RF
    return transport_fixed_eps, quad_refine, RF

