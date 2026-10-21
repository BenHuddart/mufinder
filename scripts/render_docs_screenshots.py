"""Render the documentation screenshots; see the ``docs_screenshots`` package beside this script.

    python scripts/render_docs_screenshots.py --help
"""

from __future__ import annotations

from pathlib import Path
import sys

# Run as a script, Python already puts this directory first on the path;
# inserting it again keeps an import of this file working too.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from docs_screenshots.__main__ import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
