#!/usr/bin/env python3
import ast
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: read_prompts.py /path/to/prompt.txt")
    text = Path(sys.argv[1]).read_text(encoding="utf-8").strip()
    data = ast.literal_eval("{" + text + "}")
    prompts = data["prompts"]
    if not isinstance(prompts, list):
        raise SystemExit("prompt.txt must contain a prompts list")
    for prompt in prompts:
        print(prompt.replace("\n", " ").strip())


if __name__ == "__main__":
    main()
