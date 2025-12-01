import sys
from pathlib import Path

# Ensure src is on sys.path so we can import segmentx
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from segmentx.app import main


if __name__ == "__main__":
    main()
