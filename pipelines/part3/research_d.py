"""Part 3 entry point: development/heldout D mechanism study."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.research_direction_d import main

if __name__=='__main__': main()
