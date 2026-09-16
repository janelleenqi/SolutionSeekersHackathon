import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.database import load_database


class DatabaseTests(unittest.TestCase):
    def test_loads_schema_and_relationships(self):
        with tempfile.TemporaryDirectory() as temp:
            database = Path(temp) / "wealth.db"
            counts = load_database(Path("data/raw"), database)
            self.assertEqual(counts, {
                "clients": 15,
                "products": 13,
                "holdings": 70,
                "transactions": 55,
                "email_threads": 7,
                "email_messages": 18,
            })
            connection = sqlite3.connect(database)
            self.assertEqual(
                connection.execute(
                    """SELECT c.name, h.product_name
                    FROM clients c JOIN holdings h ON h.client_id = c.client_id
                    WHERE c.client_id = 'CL002' AND h.product_name LIKE 'APEX%'"""
                ).fetchone(),
                ("Robert Chua", "APEX Global Multi-Asset Autocallable Note Series 7"),
            )
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM email_messages WHERE thread_id = 'EML-001'").fetchone()[0],
                3,
            )
            connection.close()


if __name__ == "__main__":
    unittest.main()