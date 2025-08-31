#!/usr/bin/env python3
"""
sanitize_flat_folder.py

Non-recursively sanitize filenames in a single folder for Kaggle-friendly slugs.

Usage (dry-run, default):
    python sanitize_flat_folder.py /path/to/folder

To actually rename:
    python sanitize_flat_folder.py /path/to/folder --apply

Options:
    --apply      perform renames (dry-run if omitted)
    --lower      convert resulting names to lowercase
    --verbose    print more info
"""
import argparse
import os
import re
import sys

def slugify(name: str, lowercase: bool = False) -> str:
    # replace underscores with hyphens
    name = name.replace("_", "-")
    # remove any character not alphanumeric or hyphen
    name = re.sub(r"[^A-Za-z0-9-]+", "", name)
    # collapse multiple hyphens
    name = re.sub(r"-{2,}", "-", name)
    # strip leading/trailing hyphens
    name = name.strip("-")
    if lowercase:
        name = name.lower()
    if not name:
        name = "file"
    return name

def unique_name(folder: str, base: str, ext: str) -> str:
    candidate = f"{base}{ext}"
    i = 1
    while os.path.exists(os.path.join(folder, candidate)):
        candidate = f"{base}-{i}{ext}"
        i += 1
    return candidate

def process_folder(folder: str, apply: bool, lowercase: bool, verbose: bool):
    if not os.path.isdir(folder):
        print(f"Error: not a directory: {folder}", file=sys.stderr)
        sys.exit(1)

    entries = sorted(os.listdir(folder))
    ops = []  # (old_fullpath, new_fullpath)

    for name in entries:
        old_path = os.path.join(folder, name)
        if not os.path.isfile(old_path):
            if verbose:
                print(f"[skip-nonfile] {name}")
            continue

        base, ext = os.path.splitext(name)  # ext includes leading dot or empty
        new_base = slugify(base, lowercase=lowercase)
        new_name = f"{new_base}{ext}"
        if new_name == name:
            if verbose:
                print(f"[ok] {name}")
            continue

        # handle collisions (don't clash with existing files or other planned new names)
        # Temporarily consider current planned targets to avoid two files mapping to same new name.
        planned_targets = {os.path.basename(tgt) for _, tgt in ops}
        target_candidate = new_name
        i = 1
        while (os.path.exists(os.path.join(folder, target_candidate)) and os.path.join(folder, target_candidate) != old_path) \
              or target_candidate in planned_targets:
            target_candidate = f"{new_base}-{i}{ext}"
            i += 1

        new_full = os.path.join(folder, target_candidate)
        ops.append((old_path, new_full))

    if not ops:
        print("Nothing to rename.")
        return

    print("Planned renames:")
    for a, b in ops:
        print(f"  {os.path.basename(a)}  ->  {os.path.basename(b)}")

    if not apply:
        print("\nDry-run: no files were changed. Re-run with --apply to perform renames.")
        return

    succeeded = 0
    failed = []
    for old, new in ops:
        try:
            os.replace(old, new)
            succeeded += 1
            if verbose:
                print(f"[renamed] {os.path.basename(old)} -> {os.path.basename(new)}")
        except Exception as e:
            failed.append((old, new, str(e)))
            print(f"[ERROR] {old} -> {new} : {e}", file=sys.stderr)

    print(f"\nDone. {succeeded} succeeded, {len(failed)} failed.")
    if failed:
        for old, new, err in failed:
            print(f"  FAILED: {os.path.basename(old)} -> {os.path.basename(new)}  : {err}")

def main():
    p = argparse.ArgumentParser(description="Sanitize filenames (non-recursive) in a folder.")
    p.add_argument("folder", help="Target folder (single level).")
    p.add_argument("--apply", action="store_true", help="Perform the renames (default: dry-run).")
    p.add_argument("--lower", action="store_true", dest="lower", help="Convert names to lowercase.")
    p.add_argument("--verbose", action="store_true", help="Verbose output.")
    args = p.parse_args()

    process_folder(os.path.abspath(args.folder), apply=args.apply, lowercase=args.lower, verbose=args.verbose)

if __name__ == "__main__":
    main()
