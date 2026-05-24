"""Generate a WiX v4 fragment file from a PyInstaller onedir bundle.

WiX 4.0.x does not support the <Files> glob element (that is a WiX 5 feature).
This script replicates the equivalent harvesting behaviour so the installer can be
built with the project-pinned WiX 4.0.6 toolchain.
"""

from __future__ import annotations

import argparse
import hashlib
from html import escape as _html_escape
import re
import sys
from pathlib import Path


def _xml_attr(value: str) -> str:
    """Escape a string for safe use inside an XML double-quoted attribute value.

    Converts ``&`` → ``&amp;``, ``<`` → ``&lt;``, ``>`` → ``&gt;``,
    ``"`` → ``&quot;``.  Required because ``&`` is valid in NTFS file and
    directory names but is an XML well-formedness error when unescaped.
    """
    return _html_escape(value, quote=True)


def _wix_id(path_str: str, prefix: str = "") -> str:
    digest = hashlib.md5(path_str.encode()).hexdigest()[:8]
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", path_str)
    if sanitized and sanitized[0].isdigit():
        sanitized = "_" + sanitized
    # Always suffix with the hash so IDs are unique even when sanitized paths collide
    # (e.g. "Etc/GMT+0" and "Etc/GMT-0" both sanitize to "Etc_GMT_0").
    readable = sanitized[:56]
    return f"{prefix}{readable}_{digest}"


def _collect(dir_path: Path, app_dir: Path, lines: list[str], components: list[str], depth: int) -> None:
    pad = "  " * depth
    for item in sorted(dir_path.iterdir()):
        # Dangling symlinks are not file or directory — skip them with a warning.
        if item.is_symlink() and not item.exists():
            print(f"WARNING: skipping dangling symlink: {item}", file=sys.stderr)
            continue
        rel = item.relative_to(app_dir)
        rel_str = str(rel)
        if item.is_file():
            comp_id = _wix_id(rel_str, "comp_")
            file_id = _wix_id(rel_str, "file_")
            lines.append(f'{pad}<Component Id="{comp_id}" Guid="*">')
            lines.append(f'{pad}  <File Id="{file_id}" Source="$(var.AppDir)\\{_xml_attr(rel_str)}" KeyPath="yes" />')
            lines.append(f"{pad}</Component>")
            components.append(comp_id)
        elif item.is_dir():
            dir_id = _wix_id(rel_str, "dir_")
            lines.append(f'{pad}<Directory Id="{dir_id}" Name="{_xml_attr(item.name)}">')
            _collect(item, app_dir, lines, components, depth + 1)
            # Empty directories need a CreateFolder component so WiX creates them on install.
            if not any(True for _ in item.rglob("*") if not (_.is_symlink() and not _.exists()) and _.is_file()):
                empty_comp_id = _wix_id(rel_str, "emptydir_")
                lines.append(f'{pad}  <Component Id="{empty_comp_id}" Guid="*">')
                lines.append(f"{pad}    <CreateFolder />")
                lines.append(f"{pad}  </Component>")
                components.append(empty_comp_id)
            lines.append(f"{pad}</Directory>")


def generate_fragment(app_dir: Path, output: Path) -> None:
    app_dir = app_dir.resolve()
    body: list[str] = []
    components: list[str] = []
    _collect(app_dir, app_dir, body, components, depth=3)

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<Wix xmlns="http://wixtoolset.org/schemas/v4/wxs">',
        "  <Fragment>",
        '    <DirectoryRef Id="INSTALLFOLDER">',
        *body,
        "    </DirectoryRef>",
        "  </Fragment>",
        "  <Fragment>",
        '    <ComponentGroup Id="AppFiles">',
        *[f'      <ComponentRef Id="{c}" />' for c in components],
        "    </ComponentGroup>",
        "  </Fragment>",
        "</Wix>",
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Generated {output} ({len(components)} components).")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("app_dir", type=Path, help="PyInstaller onedir output (e.g. dist/MuFinder)")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("packaging/windows/AppFiles.wxs"),
        help="Destination WiX fragment file (default: packaging/windows/AppFiles.wxs)",
    )
    args = parser.parse_args()
    generate_fragment(args.app_dir, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
