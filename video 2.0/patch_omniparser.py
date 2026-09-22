#!/usr/bin/env python3

import sys

PATH = "util/utils.py"

FIXES = [
    (
        "ocr_bbox = None",
        "ocr_bbox = []",
        "empty-OCR ocr_bbox None->[] fix",
    ),
    (
        "            else:\n                filtered_boxes.append(box1)\n    return filtered_boxes",
        "            else:\n                filtered_boxes.append(box1_elem)\n    return filtered_boxes",
        "remove_overlap_new raw-bbox append fix",
    ),
]


def main():
    with open(PATH, "r", encoding="utf-8") as f:
        content = f.read()

    for old, new, label in FIXES:
        count = content.count(old)
        if count == 0:
            print(f"ERROR: pattern for '{label}' not found in {PATH}; "
                  f"upstream file may have changed, patch needs updating.", file=sys.stderr)
            sys.exit(1)
        if count > 1:
            print(f"ERROR: pattern for '{label}' matched {count} times (expected 1); "
                  f"refusing to apply ambiguous patch.", file=sys.stderr)
            sys.exit(1)
        content = content.replace(old, new)
        print(f"Applied fix: {label}")

    with open(PATH, "w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    main()
