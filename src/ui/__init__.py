"""
src/ui/__init__.py
==================
UI components and styling for the Streamlit dashboard.

This package contains:
  - styles.py     — Custom CSS styling
  - components.py — Reusable UI components (future)
  - sidebar.py    — Sidebar logic (future)
"""

from .styles import apply_custom_styles

__all__ = ["apply_custom_styles"]
