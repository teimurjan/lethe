#!/usr/bin/env python3
"""Print a single turn from a Claude Code JSONL transcript as plain text.

Usage:
  transcript.py <path> [--turn <uuid>]

If --turn is omitted, prints the last user/assistant pair in the transcript.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def message_text(msg: object) -> str:
    if not isinstance(msg, dict):
        return ""
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text" and isinstance(block.get("text"), str):
                parts.append(block["text"])
            elif btype == "tool_use":
                parts.append(f"[tool_use: {block.get('name', '?')}]")
            elif btype == "tool_result":
                parts.append("[tool_result]")
        return "\n".join(parts)
    return ""


def is_tool_result_only(msg: object) -> bool:
    if not isinstance(msg, dict):
        return False
    content = msg.get("content")
    if not isinstance(content, list) or not content:
        return False
    return all(
        isinstance(b, dict) and b.get("type") == "tool_result" for b in content
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("path", type=Path)
    p.add_argument("--turn", default=None, help="user-turn UUID to surface")
    args = p.parse_args()

    if not args.path.exists():
        print(f"transcript not found: {args.path}", file=sys.stderr)
        return 1

    user_turn: str | None = None
    assistant_turn: str | None = None
    captured_user: str | None = None

    with args.path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg = rec.get("message") or rec
            role = msg.get("role") or rec.get("type")
            uid = rec.get("uuid") or rec.get("id")
            if role == "user":
                if is_tool_result_only(msg):
                    continue
                text = message_text(msg)
                if not text:
                    continue
                if args.turn and uid == args.turn:
                    captured_user = text
                    user_turn = text
                    assistant_turn = None
                elif args.turn is None:
                    user_turn = text
                    assistant_turn = None
            elif role == "assistant":
                text = message_text(msg)
                if not text:
                    continue
                if args.turn and captured_user is not None and assistant_turn is None:
                    assistant_turn = text
                    break
                if args.turn is None:
                    assistant_turn = text

    if user_turn is None and assistant_turn is None:
        print("(no matching turn)", file=sys.stderr)
        return 1

    print("USER:")
    print((user_turn or "").strip())
    print()
    print("ASSISTANT:")
    print((assistant_turn or "").strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
