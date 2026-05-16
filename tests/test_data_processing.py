import unittest
import pandas as pd
from data_processing import load_sales_data

class TestDataProcessing(unittest.TestCase):
    def test_load_data_not_empty(self):
        # Load from the Ventes folder relative to repo root
        df = load_sales_data(data_dir="./Ventes")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty, "Loaded DataFrame should not be empty")

if __name__ == '__main__':
    unittest.main()
