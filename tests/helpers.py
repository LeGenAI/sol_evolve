import shutil
import sys
from pathlib import Path


def console_script(name: str) -> str:
    installed = Path(sys.executable).parent / name
    if installed.exists():
        return str(installed)
    found = shutil.which(name)
    assert found is not None
    return found
