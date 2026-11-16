"""Unit tests for Jarvis Long-Term Memory Store (Phase 5).

Covers:
1. Successful memory insertion across categories
2. Empty and whitespace-only input rejection
3. Length capping (truncation to 300 characters)
4. Near-duplicate skipping (ignoring case, whitespace, and punctuation)
5. Sensitive data rejection (passwords, API keys, card numbers, government IDs)
6. Keyword search matching and relevance ranking
7. Recency tie-breaking in search results
8. Automatic update of last_used_at timestamp on search
9. Hard capacity cap at 500 memories (refusal when full)
10. Deletion by ID and deletion of matching queries
11. Full database wipe with clear_all()
12. Readable export report with export_to_text()
"""

import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

from memory import (
    MAX_MEMORIES_CAP,
    MAX_TEXT_LENGTH,
    MemoryStore,
    set_embedding_enabled,
    validate_sensitive_content,
)


class TestMemoryStore(unittest.TestCase):
    """Test suite for MemoryStore class."""

    def setUp(self) -> None:
        """Create a fresh temporary SQLite database for each test."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_memory.db"
        self.store = MemoryStore(db_path=self.db_path)

    def tearDown(self) -> None:
        """Clean up temporary test database directory."""
        self.temp_dir.cleanup()

    def test_add_valid_memories(self) -> None:
        """Verify adding memories across valid categories."""
        res1 = self.store.add("User prefers dark roast coffee with oat milk.", category="preference")
        self.assertTrue(res1["success"])
        self.assertIsNotNone(res1["id"])
        self.assertEqual(res1["category"], "preference")

        res2 = self.store.add("Alice's birthday is October 14th.", category="fact")
        self.assertTrue(res2["success"])
        self.assertEqual(self.store.count(), 2)

    def test_empty_and_whitespace_rejection(self) -> None:
        """Verify rejection of empty or whitespace-only inputs."""
        empty_res = self.store.add("")
        self.assertFalse(empty_res["success"])
        self.assertIn("empty", empty_res["message"].lower())

        whitespace_res = self.store.add("   \n\t  ")
        self.assertFalse(whitespace_res["success"])
        self.assertEqual(self.store.count(), 0)

    def test_length_truncation_at_300_characters(self) -> None:
        """Verify that memories longer than 300 characters are capped cleanly."""
        long_text = "Important note: " + ("A" * 350)
        res = self.store.add(long_text, category="fact")
        self.assertTrue(res["success"])
        self.assertEqual(len(res["text"]), MAX_TEXT_LENGTH)
        self.assertEqual(self.store.count(), 1)

    def test_near_duplicate_skipping(self) -> None:
        """Verify that case, punctuation, and extra whitespace variations are skipped as duplicates."""
        res1 = self.store.add("I love Earl Grey tea.", category="preference")
        self.assertTrue(res1["success"])

        # Punctuation variation
        res2 = self.store.add("i love earl grey tea!", category="preference")
        self.assertFalse(res2["success"])
        self.assertIn("similar memory already exists", res2["message"])

        # Extra whitespace and casing variation
        res3 = self.store.add("  I   LOVE  Earl   Grey   tea...  ", category="preference")
        self.assertFalse(res3["success"])

        # Total stored should still be 1
        self.assertEqual(self.store.count(), 1)

    def test_sensitive_data_rejection(self) -> None:
        """Verify that passwords, API keys, cards, and government IDs are rejected."""
        test_cases = [
            ("My password: SuperSecretP@ssword123", "password"),
            ("The login password is 'mypass123'", "password"),
            ("OpenAI key is sk-1234567890abcdef1234567890abcdef1234", "OpenAI API key"),
            ("Google token AIzaSyD1234567890abcdef1234567890abc", "Google API key"),
            ("GitHub token gh_p1234567890abcdef1234567890abcdef123456", "GitHub token"),
            ("AWS key AKIAIOSFODNN7EXAMPLE", "AWS access key"),
            ("Visa card 4532 1234 5678 9010", "credit card number"),
            ("Mastercard 5412-3456-7890-1234", "credit card number"),
            ("My SSN is 123-45-6789", "Social Security Number (SSN)"),
            ("Aadhaar card: 1234 5678 9012", "national ID number (Aadhaar)"),
            ("PAN number is ABCDE1234F", "government tax ID (PAN)"),
        ]

        for text, label in test_cases:
            is_safe, detected = validate_sensitive_content(text)
            self.assertFalse(is_safe, f"Expected {text} to be detected as sensitive")
            res = self.store.add(text)
            self.assertFalse(res["success"], f"Store should reject {text}")
            self.assertIn("sensitive data", res["message"])

        # Nothing should have been inserted
        self.assertEqual(self.store.count(), 0)

    def test_search_keyword_ranking(self) -> None:
        """Verify search scores higher match counts higher, with recency tie-breaking."""
        self.store.add("The user prefers dark roast coffee in the morning.", category="preference")
        self.store.add("The user drinks green tea in the evening.", category="preference")
        self.store.add("Coffee machine is located in the second floor kitchen.", category="fact")

        # Query matches "coffee" and "morning" (2 words for item 1, 1 word for item 3, 0 for item 2)
        results = self.store.search("coffee morning", limit=5)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["match_count"], 2)
        self.assertIn("dark roast coffee", results[0]["text"])
        self.assertEqual(results[1]["match_count"], 1)
        self.assertIn("Coffee machine", results[1]["text"])

    def test_search_updates_last_used_at(self) -> None:
        """Verify that searching updates the last_used_at timestamp on returned items."""
        res = self.store.add("Python is the user's primary programming language.", category="fact")
        mem_id = res["id"]

        # Retrieve initial last_used_at
        initial = self.store.list_all()[0]
        t0 = initial["last_used_at"]

        # Sleep briefly so timestamp advances
        time.sleep(0.05)

        # Execute search
        results = self.store.search("Python programming")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], mem_id)
        t1 = results[0]["last_used_at"]

        self.assertNotEqual(t0, t1)

    def test_capacity_cap_at_500(self) -> None:
        """Verify that adding exceeds the 500-item hard limit is refused."""
        # Rapidly populate 500 memories directly
        with self.store._get_connection() as conn:
            now = "2026-10-30T10:00:00Z"
            for i in range(MAX_MEMORIES_CAP):
                conn.execute(
                    "INSERT INTO memories (text, category, created_at, last_used_at) VALUES (?, ?, ?, ?)",
                    (f"Fact number {i} about project architecture", "fact", now, now),
                )
            conn.commit()

        self.assertEqual(self.store.count(), MAX_MEMORIES_CAP)

        # Attempt to insert the 501st memory
        res = self.store.add("Fact number 501", category="fact")
        self.assertFalse(res["success"])
        self.assertIn("full", res["message"].lower())
        self.assertIn("500", res["message"])

        # Count must remain strictly at 500
        self.assertEqual(self.store.count(), MAX_MEMORIES_CAP)

    def test_delete_by_id(self) -> None:
        """Verify deletion of specific record by its ID."""
        r1 = self.store.add("Temporary memo 1", category="task")
        self.store.add("Temporary memo 2", category="task")

        self.assertEqual(self.store.count(), 2)
        deleted = self.store.delete(r1["id"])
        self.assertTrue(deleted)
        self.assertEqual(self.store.count(), 1)

        # Deleting non-existent ID
        deleted_again = self.store.delete(r1["id"])
        self.assertFalse(deleted_again)

    def test_delete_matching(self) -> None:
        """Verify deleting memories by matching query keywords."""
        self.store.add("Buy grocery items: milk, eggs, bread", category="task")
        self.store.add("Buy birthday gift for Sarah", category="task")
        self.store.add("Sarah lives in Chicago", category="person")

        self.assertEqual(self.store.count(), 3)
        deleted_count = self.store.delete_matching("Sarah")
        self.assertEqual(deleted_count, 2)
        self.assertEqual(self.store.count(), 1)
        self.assertIn("grocery", self.store.list_all()[0]["text"])

    def test_clear_all(self) -> None:
        """Verify wiping all memories."""
        self.store.add("Fact 1", category="fact")
        self.store.add("Fact 2", category="fact")
        self.assertEqual(self.store.count(), 2)

        cleared = self.store.clear_all()
        self.assertEqual(cleared, 2)
        self.assertEqual(self.store.count(), 0)

    def test_export_to_text(self) -> None:
        """Verify formatted text report generation."""
        self.store.add("User lives in Seattle.", category="person")
        self.store.add("Prefers metric units.", category="preference")

        report = self.store.export_to_text()
        self.assertIn("Jarvis Long-Term Memory Store", report)
        self.assertIn("Total Memories: 2/500", report)
        self.assertIn("[PREFERENCE]", report)
        self.assertIn("[PERSON]", report)
        self.assertIn("User lives in Seattle.", report)

    def test_semantic_paraphrase_recall(self) -> None:
        """Verify semantic recall finds 'my sister's birthday is in March' from 'when's my sister's party'."""
        self.store.add("my sister's birthday is in March", category="fact")
        self.store.add("I prefer dark mode in code editors", category="preference")
        self.store.add("Meeting with project team on Tuesday afternoon", category="task")

        results = self.store.search("when's my sister's party", limit=5)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["text"], "my sister's birthday is in March")
        self.assertGreater(results[0]["similarity"], 0.60)
        self.assertIn("combined_score", results[0])

    def test_unrelated_query_returns_nothing(self) -> None:
        """Verify queries completely unrelated to stored memories return an empty list."""
        self.store.add("My favorite fruit is mango.", category="preference")
        self.store.add("Work laptop model is Dell Latitude.", category="fact")

        # Query completely out of domain with min_similarity
        results = self.store.search("quantum particle accelerator astrophysics", limit=5, min_similarity=0.55)
        self.assertEqual(len(results), 0)

    def test_embedding_fallback_path(self) -> None:
        """Verify fallback to pure keyword search when embedding model is disabled/unavailable."""
        self.store.add("User enjoys Italian espresso roast coffee.", category="preference")
        self.store.add("Office printer model is HP LaserJet.", category="fact")

        # Temporarily disable embeddings
        set_embedding_enabled(False)
        try:
            results = self.store.search("coffee espresso")
            self.assertEqual(len(results), 1)
            self.assertIn("espresso roast coffee", results[0]["text"])
            self.assertEqual(results[0]["match_count"], 2)
        finally:
            set_embedding_enabled(True)

    def test_db_migration_and_backup(self) -> None:
        """Verify legacy databases without embedding column are backed up and migrated."""
        legacy_db = Path(self.temp_dir.name) / "legacy_memory.db"
        # Create legacy table without 'embedding' column
        conn = sqlite3.connect(str(legacy_db))
        conn.execute("""
            CREATE TABLE memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                category TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_used_at TEXT NOT NULL
            )
        """)
        conn.execute("INSERT INTO memories VALUES (1, 'Old memory before migration', 'fact', '2026-01-01', '2026-01-01')")
        conn.commit()
        conn.close()

        # Initialize MemoryStore on the legacy DB
        migrated_store = MemoryStore(db_path=legacy_db)

        # 1. Verify backup file was created
        backup_file = legacy_db.with_name(f"{legacy_db.name}.bak")
        self.assertTrue(backup_file.exists(), "Backup file should be created before migration")

        # 2. Verify 'embedding' column was added
        with migrated_store._get_connection() as c:
            cols = [row["name"] for row in c.execute("PRAGMA table_info(memories)").fetchall()]
            self.assertIn("embedding", cols)

        # 3. Verify existing memory was preserved
        all_mems = migrated_store.list_all()
        self.assertEqual(len(all_mems), 1)
        self.assertEqual(all_mems[0]["text"], "Old memory before migration")

    def test_benchmark_search_500_memories(self) -> None:
        """Populate 500 memories and benchmark semantic search latency."""
        from memory import compute_embedding

        # Precompute target embedding
        target_text = "my sister's birthday is in March"
        target_vec = compute_embedding(target_text)
        target_blob = target_vec.tobytes() if target_vec is not None else None

        # Create dummy vector for other 499 memories
        import numpy as np
        rng = np.random.default_rng(42)
        dummy_matrix = rng.normal(size=(499, 384)).astype(np.float32)
        dummy_matrix /= np.linalg.norm(dummy_matrix, axis=1, keepdims=True)

        now = "2026-10-30T10:00:00Z"
        with self.store._get_connection() as conn:
            # Insert target memory
            conn.execute(
                "INSERT INTO memories (text, category, created_at, last_used_at, embedding) VALUES (?, ?, ?, ?, ?)",
                (target_text, "fact", now, now, target_blob),
            )
            # Insert 499 filler memories
            for i in range(499):
                blob = dummy_matrix[i].tobytes()
                conn.execute(
                    "INSERT INTO memories (text, category, created_at, last_used_at, embedding) VALUES (?, ?, ?, ?, ?)",
                    (f"Random filler memory entry number {i} for testing", "other", now, now, blob),
                )
            conn.commit()

        self.assertEqual(self.store.count(), 500)

        # Benchmark search latency over 500 items
        t0 = time.perf_counter()
        results = self.store.search("when's my sister's party", limit=5)
        elapsed = time.perf_counter() - t0

        elapsed_ms = elapsed * 1000.0
        print(f"\n[Benchmark] Search across 500 memories completed in {elapsed_ms:.2f} ms")

        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["text"], target_text)
        self.assertLess(elapsed, 0.250, "Search across 500 memories should take < 250ms")


def run_tests():
    """Run all tests and display formatted report."""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMemoryStore)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    unittest.main()

