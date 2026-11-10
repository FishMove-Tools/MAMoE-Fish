"""Portable model/resource directory for MAMoE-Fish."""
from pathlib import Path
import os, sys
HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('MAMOE_MODEL_DIR', os.environ.get('FAMOE_RELEASE_DIR', str(HERE.parent / 'model')))).expanduser().resolve()
if not (ROOT / 'group.xml').is_file():
    raise FileNotFoundError(f'Missing model: {ROOT / "group.xml"}')
PREVIOUS = ROOT
sys.dont_write_bytecode = True
