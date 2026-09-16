import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.hallucination import check_hallucination
from src.structural_retrieval import retrieve_structured_data


class HallucinationDiagnosticsTests(unittest.TestCase):
    def test_contradiction_never_certified(self):
        result = check_hallucination('The product is principal protected.',
                                     ['The product is not principal protected.'])
        self.assertFalse(result['supported'])
        self.assertEqual(result['status'], 'unverified')
        self.assertEqual(result['confidence'], 0)

    def test_pipeline_evidence_and_wrong_amount(self):
        result = check_hallucination('Minimum investment is SGD 100 [S1].',
            [{'text': 'Minimum investment is SGD 10,000.', 'metadata': {}}])
        self.assertIn('numbers_absent_from_evidence', result['issues'])
        self.assertEqual(result['unknown_citations'], [])

    def test_labels_and_empty_evidence(self):
        result = check_hallucination('Evidence [S2]', [{'text': 'Evidence', 'label': 'S1'}])
        self.assertEqual(result['unknown_citations'], ['S2'])
        self.assertFalse(check_hallucination('', [])['supported'])
        self.assertFalse(check_hallucination('Evidence', None)['supported'])

    def test_validations(self):
        with self.assertRaises(ValueError):
            check_hallucination('x', ['x'], threshold=2)
        with self.assertRaises(TypeError):
            check_hallucination('x', [{'text': None}])


class ReadOnlyRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'wealth.db'
        connection = sqlite3.connect(self.path)
        connection.execute('CREATE TABLE clients (client_id TEXT, name TEXT)')
        connection.executemany('INSERT INTO clients VALUES (?, ?)',
                               [('CL002', 'Robert'), ('CL003', 'Other')])
        connection.commit()
        connection.close()

    def query(self, sql, params=(), **kwargs):
        return retrieve_structured_data(sql, params, database_path=self.path, **kwargs)

    def test_legacy_tuples_and_named_rows(self):
        self.assertEqual(self.query('SELECT name FROM clients WHERE client_id=?', ('CL002',)), [('Robert',)])
        self.assertEqual(self.query('SELECT name FROM clients WHERE client_id=?', ('CL002',), as_dict=True), [{'name': 'Robert'}])

    def test_rejects_mutations_and_keeps_database(self):
        for sql in ['DROP TABLE clients', 'DELETE FROM clients',
                    "UPDATE clients SET name='bad'", 'PRAGMA user_version=2',
                    "ATTACH DATABASE ':memory:' AS other"]:
            with self.subTest(sql=sql), self.assertRaises(sqlite3.DatabaseError):
                self.query(sql)
        self.assertEqual(self.query('SELECT COUNT(*) FROM clients'), [(2,)])

    def test_missing_database_is_not_created(self):
        missing = self.path.parent / 'missing.db'
        with self.assertRaises(FileNotFoundError):
            retrieve_structured_data('SELECT 1', database_path=missing)
        self.assertFalse(missing.exists())

    def test_bounds_and_ambiguous_names(self):
        with self.assertRaises(ValueError):
            self.query('SELECT * FROM clients', max_rows=1)
        with self.assertRaises(ValueError):
            self.query('SELECT name, name FROM clients', as_dict=True)
        with self.assertRaises(sqlite3.DatabaseError):
            self.query('WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM n) SELECT sum(x) FROM n')


if __name__ == '__main__':
    unittest.main()
