import unittest

from src.structured_data import StructuredDataStore


class StructuredDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = StructuredDataStore()

    def test_source_files_reconcile(self):
        report = self.store.validation_result()
        self.assertTrue(report.data["valid"])
        self.assertEqual(report.data["client_count"], 15)
        self.assertEqual(report.data["holding_count"], 70)
        self.assertEqual(report.data["transaction_count"], 55)
        self.assertEqual(report.data["pending_transaction_count"], 2)

    def test_client_can_be_resolved_by_name_or_id(self):
        by_id = self.store.get_client_profile("CL002")
        by_name = self.store.get_client_profile("Robert Chua")
        self.assertEqual(by_id.data["client_id"], "CL002")
        self.assertEqual(by_name.data["client_id"], "CL002")
        self.assertEqual(by_name.data["risk_score_1_to_10"], 2)

    def test_product_exposure_is_calculated_from_holding_rows(self):
        result = self.store.calculate_product_exposure("Robert Chua", "APEX")
        self.assertEqual(result.data["allocation_pct"], 15)
        self.assertEqual(result.data["value_sgd"], 127500)
        self.assertEqual(result.sources[0].source_path, "client_portfolio/clients_portfolio.csv")
        self.assertIsNotNone(result.sources[0].row)

    def test_pending_rebalancing_is_returned_for_james_sullivan(self):
        result = self.store.get_transactions("CL013", status="Pending")
        self.assertEqual(len(result.data), 1)
        self.assertEqual(result.data[0]["transaction_id"], "TXN-1047")
        self.assertIn("Portfolio Rebalancing", result.data[0]["product_name"])

    def test_complex_concentration_uses_explicit_asset_class_classification(self):
        result = self.store.find_complex_product_concentrations(20)
        self.assertEqual(
            [row["client_id"] for row in result.data], ["CL011", "CL001", "CL009"]
        )
        retail = self.store.find_complex_product_concentrations(20, retail_only=True)
        self.assertEqual(retail.data, [])

    def test_structured_result_can_be_added_to_rag_prompt(self):
        result = self.store.calculate_product_exposure("CL002", "APEX")
        block = result.to_prompt_block("D1")
        self.assertIn("[D1]", block)
        self.assertIn("structured calculation", block)
        self.assertIn("row", block)

    def test_correspondence_is_loaded_and_validated(self):
        self.assertEqual(len(self.store.email_threads), 7)
        self.assertEqual(len(self.store.email_messages), 18)
        thread = self.store.email_threads[0]
        self.assertEqual(thread["client_id"], "CL006")
        self.assertEqual(self.store.email_messages[0]["thread_id"], "EML-001")


if __name__ == "__main__":
    unittest.main()
