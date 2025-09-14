import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from generate_mql4 import main

if __name__ == "__main__":
    main()
