import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from botcopier.scripts.generate_mql4_from_model import main

if __name__ == "__main__":
    main()
