from __future__ import annotations

import pandas as pd

from romcomma.base.definitions import *
from romcomma.base.classes import Frame

ROOT: Path = Path('C:/Users/fc1ram/Documents/Research/dat/SoftwareTest/0.0')     #: The root folder to house all data repositories.

if __name__ == '__main__':
    data = pd.DataFrame([[0, 1],[2, 3]])
    result = Frame(ROOT / 'data', data)
