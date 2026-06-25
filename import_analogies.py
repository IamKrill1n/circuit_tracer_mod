"""Backward-compatible wrapper for import_dataset.py."""

from import_dataset import _replace_with_summary_link, main

__all__ = ["_replace_with_summary_link", "main"]

if __name__ == "__main__":
    main()
