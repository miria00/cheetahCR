"""Regenerate the current grouped four-decoder feature pies."""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).with_name("generate_figure3_clean_pies.py")), run_name="__main__")
