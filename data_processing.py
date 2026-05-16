import pandas as pd
import glob
import os
from pathlib import Path

def load_sales_data(data_dir: str = "./") -> pd.DataFrame:
    """Load all Ventes_*.xlsx files, concatenate, and perform basic cleaning.
    
    Args:
        data_dir: Directory containing the Excel files.
    Returns:
        DataFrame with columns: ['Date', 'Produit', 'Quantite', 'Montant', ...] (actual columns depend on source files).
    """
    # Find all Excel files matching pattern
    pattern = os.path.join(data_dir, "Ventes_*.xlsx")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No sales files found in {data_dir}")

    dfs = []
    for f in files:
        df = pd.read_excel(f, engine="openpyxl")
        dfs.append(df)
    # Concatenate all
    data = pd.concat(dfs, ignore_index=True)

    # Basic cleaning: ensure date column exists and is datetime
    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    else:
        # Try infer from file name (YYYY-MM)
        # Extract month-year from filename
        month_year = Path(f).stem.split("_")[1]  # e.g., 2013-01
        year, month = month_year.split("-")
        data["Date"] = pd.to_datetime(f"{year}-{month}-01")

    # Remove rows with missing critical values
    data = data.dropna(subset=["Date", "Montant"])
    # Ensure numeric types
    data["Montant"] = pd.to_numeric(data["Montant"], errors="coerce")
    data = data.dropna(subset=["Montant"]).reset_index(drop=True)
    return data
