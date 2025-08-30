# scripts/clean_merge.py
from pathlib import Path
import pandas as pd
import re

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
OUT  = BASE / "outputs"
OUT.mkdir(exist_ok=True)

PATH_91  = DATA / "HBS Table No. 213 _ Auctions of 91-Day Government of India Treasury Bills.xlsx"
PATH_SGL = DATA / "HBS Table No. 180 _ Yield of SGL Transactions in Government Dated Securiites for Various Maturities.xlsx"

# below func looks for date of auction which wont be there in sgl file if thats what you are looking for

def first_sheet(path: Path) -> str:
    return pd.ExcelFile(path).sheet_names[0]

def preview_columns(path: Path, n=12):
    df = pd.read_excel(path, sheet_name=first_sheet(path), header=None)
    # try to guess header row by scanning for “date” or “auction”
    header_row = None
    for i in range(min(15, len(df))):
        row = " ".join(df.iloc[i].astype(str).tolist())
        if re.search(r"date.*auction|auction.*date|^date$", row, re.I):
            header_row = i
            break
    if header_row is None:
        header_row = 0
    df = pd.read_excel(path, sheet_name=first_sheet(path), header=header_row)
    print("\n[91d] detected header row:", header_row)
    print("[91d] first columns:", df.columns.tolist()[:n])

if __name__ == "__main__":
    preview_columns(PATH_91)
