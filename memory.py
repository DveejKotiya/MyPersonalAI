"""Long-term memory module for Jarvis (Phase 5).

Provides a local, privacy-first SQLite memory store for user facts,
preferences, contacts, and tasks with:
- Zero external dependencies (uses Python standard library: sqlite3, re, datetime)
- SQLite database stored at data/memory.db (auto-created folder)
- Automatic length truncation (300 characters max per memory)
- Near-duplicate detection (ignoring case and punctuation)
- Keyword matching search ranked by match count and recency
- Automatic last_used_at updates upon retrieval
- Strict 500-item hard capacity cap (refuses new additions when full)
- Proactive sensitive data filtering (rejects passwords, API keys, cards, government IDs)
- Export to formatted human-readable text

Decoupled from Brain until explicitly integrated.
"""

import contextlib
import re
import shutil
import sqlite3
import sys
import threading
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

import config

# Global embedding model singleton & status
_EMBEDDING_MODEL = None
_EMBEDDING_MODEL_LOCK = threading.Lock()
_EMBEDDING_DISABLED = False


def set_embedding_enabled(enabled: bool) -> None:
    """Enable or disable embedding model for testing fallback behavior."""
    global _EMBEDDING_DISABLED
    _EMBEDDING_DISABLED = not enabled


def get_embedding_model() -> Any:
    """Retrieve or lazily initialize the local FastEmbed TextEmbedding instance.

    Returns:
        fastembed.TextEmbedding instance or None if unavailable/disabled.
    """
    global _EMBEDDING_MODEL, _EMBEDDING_DISABLED
    if _EMBEDDING_DISABLED:
        return None
    if _EMBEDDING_MODEL is not None:
        return _EMBEDDING_MODEL

    with _EMBEDDING_MODEL_LOCK:
        if _EMBEDDING_MODEL is not None:
            return _EMBEDDING_MODEL
        try:
            from fastembed import TextEmbedding

            model_name = getattr(config, "EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
            _EMBEDDING_MODEL = TextEmbedding(model_name=model_name)
            return _EMBEDDING_MODEL
        except Exception as err:
            print(f"[Memory Notice] Local embedding model unavailable: {err}", file=sys.stderr)
            return None


def compute_embedding(text: str) -> np.ndarray | None:
    """Compute normalized 1D float32 embedding vector for text using local model.

    Returns:
        Normalized numpy float32 array or None if embedding failed/disabled.
    """
    model = get_embedding_model()
    if model is None:
        return None
    try:
        embeddings = list(model.embed([text]))
        if embeddings:
            vec = np.array(embeddings[0], dtype=np.float32)
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec = vec / norm
            return vec
    except Exception as err:
        print(f"[Memory Warning] Embedding computation failed: {err}", file=sys.stderr)
    return None


def _calculate_recency_boost(timestamp_iso: str | None) -> float:
    """Calculate a normalized recency score between 0.0 and 0.10."""
    if not timestamp_iso:
        return 0.0
    try:
        ts = datetime.fromisoformat(timestamp_iso)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        age_seconds = max(0.0, (datetime.now(UTC) - ts).total_seconds())
        days = age_seconds / 86400.0
        return float(0.10 * np.exp(-days / 14.0))
    except Exception:
        return 0.0


def compute_similarity(query_vec: np.ndarray, doc_vec: np.ndarray) -> float:
    """Compute calibrated cosine similarity between unit normalized vectors.

    Calibrates baseline BGE dot-product offset (~0.40) to standard [0.0, 1.0] scale.
    """
    raw_dot = float(np.dot(query_vec, doc_vec))
    if raw_dot > 0.40:
        return min(1.0, (raw_dot - 0.40) / 0.60)
    elif raw_dot > 0.0:
        return max(0.0, (raw_dot / 0.40) * 0.25)
    return 0.0



# Allowed memory taxonomy categories
VALID_CATEGORIES = {"preference", "fact", "person", "task", "other"}

# Maximum number of memories allowed in the store (from config.py)
MAX_MEMORIES_CAP = getattr(config, "MAX_MEMORIES_CAP", 500)

# Maximum character length per memory text (from config.py)
MAX_TEXT_LENGTH = getattr(config, "MAX_MEMORY_TEXT_LENGTH", 300)


# ==============================================================================
# Sensitive Data Sanitizer & Regex Filter
# ==============================================================================

# Regex patterns matching sensitive credentials, keys, card numbers, and IDs
SENSITIVE_PATTERNS = [
    # Passwords (e.g. "password: xyz", "password is xyz", "passwd = xyz")
    (re.compile(r"(?i)\b(?:password|passwd|pwd)\b\s*(?:is|[:=])\s*\S+"), "password"),
    (re.compile(r"(?i)\b(?:login|user|account|wifi)?\s*(?:password|passwd|pwd)\s*(?:is|[:=])\s*\S+"), "password"),

    # API Keys & Tokens
    (re.compile(r"sk-[a-zA-Z0-9_\-]{20,}"), "OpenAI API key"),
    (re.compile(r"AIza[0-9A-Za-z\-_]{30,}"), "Google API key"),
    (re.compile(r"(?:gh[pousr]?_|gh_[a-zA-Z0-9]|github_pat_)[A-Za-z0-9_]{20,}"), "GitHub token"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "AWS access key"),
    (re.compile(r"(?i)\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|secret[_-]?key)\b\s*[:=]\s*['\"]?[a-zA-Z0-9_\-]{16,}"), "API credential"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private key"),

    # Payment Cards (Visa, MasterCard, Amex, Discover, etc. with dashes, spaces, or raw 15-16 digits)
    (re.compile(r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b"), "credit card number"),
    (re.compile(r"\b(?:\d{4}[ -]){3}\d{4}\b"), "credit card number"),
    (re.compile(r"\b3[47]\d{2}[ -]\d{6}[ -]\d{5}\b"), "credit card number"),

    # Government Identification Numbers
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "Social Security Number (SSN)"),
    (re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b"), "national ID number (Aadhaar)"),
    (re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"), "government tax ID (PAN)"),
]


def validate_sensitive_content(text: str) -> tuple[bool, str | None]:
    """Inspect text for sensitive credentials, secrets, cards, or ID numbers.

    Args:
        text: Input string to validate.

    Returns:
        Tuple of (is_safe: bool, reason: str | None).
        If sensitive data is found, is_safe is False and reason identifies the type.
    """
    for pattern, label in SENSITIVE_PATTERNS:
        if pattern.search(text):
            return False, label
    return True, None


def normalize_for_comparison(text: str) -> str:
    """Normalize text by lowercasing, stripping punctuation, and collapsing whitespace.

    Used for near-duplicate memory detection.

    Args:
        text: Raw input string.

    Returns:
        Clean alphanumeric normalized string.
    """
    # Remove all non-alphanumeric characters (punctuation, symbols)
    cleaned = re.sub(r"[^\w\s]", "", text.lower())
    # Collapse multiple whitespaces into a single space
    return " ".join(cleaned.split())


# ==============================================================================
# Local SQLite Long-Term Memory Store
# ==============================================================================

# Module-level database lock ensuring synchronization across all MemoryStore instances
_MEMORY_DB_LOCK = threading.Lock()


class MemoryStore:
    """Manages long-term memories in a local SQLite database."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        """Initialize the memory database and create required schema.

        Args:
            db_path: Path to SQLite database file. Defaults to data/memory.db.
        """
        if db_path:
            self.db_path = Path(db_path) if db_path != ":memory:" else db_path
        else:
            base_dir = getattr(config, "BASE_DIR", Path(__file__).resolve().parent)
            self.db_path = base_dir / "data" / "memory.db"

        # Automatically create directory structure if using file path
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._lock = _MEMORY_DB_LOCK
        self._init_db()

    @contextlib.contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Open a SQLite connection configured with Row factory and ensure proper closure."""
        conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            timeout=10.0,
        )
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _migrate_db(self) -> None:
        """Ensure schema is up to date, adding 'embedding' BLOB column if missing.

        Safely creates a backup copy of data/memory.db before altering the database.
        """
        if self.db_path == ":memory:":
            return

        db_file = Path(self.db_path)
        if not db_file.exists():
            return

        # Check existing table schema
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memories'")
            if not cursor.fetchone():
                return

            cursor = conn.execute("PRAGMA table_info(memories)")
            cols = [row["name"] for row in cursor.fetchall()]
            if "embedding" in cols:
                return

        # Safe backup before schema change
        try:
            backup_file = db_file.with_name(f"{db_file.name}.bak")
            shutil.copy2(db_file, backup_file)
            print(f"[Memory Migration] Created backup of {db_file} at {backup_file}")
        except Exception as backup_err:
            print(f"[Memory Warning] Database backup failed: {backup_err}", file=sys.stderr)

        # Alter table
        with self._get_connection() as conn:
            conn.execute("ALTER TABLE memories ADD COLUMN embedding BLOB")
            conn.commit()
            print("[Memory Migration] Successfully added 'embedding' BLOB column to memories table.")

    def _lazy_compute_missing_embeddings(self) -> None:
        """Background thread to lazily compute embeddings for any existing rows missing them."""
        def worker() -> None:
            try:
                threading.Event().wait(0.3)
                if self.db_path == ":memory:":
                    return
                db_p = Path(self.db_path)
                if not db_p.exists():
                    return

                with self._lock, self._get_connection() as conn:
                    cursor = conn.execute("SELECT id, text FROM memories WHERE embedding IS NULL")
                    missing_rows = cursor.fetchall()

                if not missing_rows:
                    return

                model = get_embedding_model()
                if model is None:
                    return

                for row in missing_rows:
                    if not db_p.exists():
                        return
                    vec = compute_embedding(row["text"])
                    if vec is not None and db_p.exists():
                        with self._lock, self._get_connection() as conn:
                            conn.execute(
                                "UPDATE memories SET embedding = ? WHERE id = ?",
                                (vec.tobytes(), row["id"]),
                            )
                            conn.commit()
            except sqlite3.OperationalError:
                # Silently return if temporary test DB was closed/cleaned up
                return
            except Exception as err:
                print(f"[Memory Warning] Lazy embedding worker error: {err}", file=sys.stderr)

        if self.db_path != ":memory:":
            t = threading.Thread(target=worker, daemon=True, name="lazy-memory-embedder")
            t.start()

    def _init_db(self) -> None:
        """Create the memories table and indexes if they do not exist."""
        with self._lock:
            self._migrate_db()
            with self._get_connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS memories (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        text TEXT NOT NULL,
                        category TEXT NOT NULL CHECK(category IN ('preference', 'fact', 'person', 'task', 'other')),
                        created_at TEXT NOT NULL,
                        last_used_at TEXT NOT NULL,
                        embedding BLOB
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_last_used_at ON memories(last_used_at)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category)")
                conn.commit()

        # Launch lazy background embedder for any rows missing embeddings
        self._lazy_compute_missing_embeddings()

    def count(self) -> int:
        """Return the total number of memories currently stored."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM memories")
            return int(cursor.fetchone()[0])

    def add(self, text: str, category: str = "other") -> dict[str, Any]:
        """Store a new fact or preference in long-term memory with validation.

        Rules:
        - Rejects empty strings or whitespace-only inputs.
        - Trims text to 300 characters max.
        - Rejects sensitive data (passwords, API keys, card numbers, government IDs).
        - Enforces a 500-memory hard cap (refuses new additions when full).
        - Skips near-duplicates (matching text ignoring case and punctuation).

        Args:
            text: The information to remember.
            category: One of 'preference', 'fact', 'person', 'task', or 'other'.

        Returns:
            Dict containing:
                success: bool
                message: explanation of the outcome
                id: integer ID if created, or None
                text: stored text (if added)
                category: assigned category (if added)
        """
        # 1. Reject empty or whitespace-only input
        cleaned_text = (text or "").strip()
        if not cleaned_text:
            return {
                "success": False,
                "message": "Memory text cannot be empty.",
                "id": None,
            }

        # 2. Check for sensitive data
        is_safe, reason = validate_sensitive_content(cleaned_text)
        if not is_safe:
            return {
                "success": False,
                "message": f"Rejected: Content contains sensitive data ({reason}).",
                "id": None,
            }

        # 3. Trim to maximum 300 characters
        trimmed_text = cleaned_text[:MAX_TEXT_LENGTH].strip()

        # 4. Normalize category (defaults to 'other' if invalid)
        cat = category.strip().lower() if category else "other"
        if cat not in VALID_CATEGORIES:
            cat = "other"

        with self._lock:
            with self._get_connection() as conn:
                # 5. Enforce 500-memory hard capacity cap
                count_cur = conn.execute("SELECT COUNT(*) FROM memories")
                current_count = int(count_cur.fetchone()[0])

                if current_count >= MAX_MEMORIES_CAP:
                    return {
                        "success": False,
                        "message": (
                            f"Memory store is full (maximum {MAX_MEMORIES_CAP} memories reached). "
                            "Please delete unused memories before adding new ones."
                        ),
                        "id": None,
                    }

                # 6. Check for near-duplicates (ignoring case, punctuation, and whitespace)
                normalized_new = normalize_for_comparison(trimmed_text)
                cursor = conn.execute("SELECT id, text FROM memories")
                for row in cursor.fetchall():
                    if normalize_for_comparison(row["text"]) == normalized_new:
                        return {
                            "success": False,
                            "message": "Skipped: A similar memory already exists.",
                            "id": row["id"],
                        }

                # 7. Compute embedding vector
                vec = compute_embedding(trimmed_text)
                embedding_blob = vec.tobytes() if vec is not None else None

                # 8. Insert the new memory with UTC timestamps and embedding
                now_iso = datetime.now(UTC).isoformat()
                insert_cur = conn.execute(
                    """
                    INSERT INTO memories (text, category, created_at, last_used_at, embedding)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (trimmed_text, cat, now_iso, now_iso, embedding_blob),
                )
                conn.commit()
                new_id = insert_cur.lastrowid

                return {
                    "success": True,
                    "message": "Memory saved successfully.",
                    "id": new_id,
                    "text": trimmed_text,
                    "category": cat,
                }

    def _search_keyword_fallback(self, query_words: set[str], limit: int) -> list[dict[str, Any]]:
        """Fallback keyword-only search when local embedding model is unavailable."""
        if not query_words:
            return []

        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT id, text, category, created_at, last_used_at FROM memories")
                rows = cursor.fetchall()

                scored_matches: list[tuple[int, dict[str, Any]]] = []

                for row in rows:
                    mem_words = set(re.findall(r"\b\w+\b", row["text"].lower()))
                    matches = query_words.intersection(mem_words)
                    score = len(matches)

                    if score > 0:
                        scored_matches.append((
                            score,
                            {
                                "id": row["id"],
                                "text": row["text"],
                                "category": row["category"],
                                "created_at": row["created_at"],
                                "last_used_at": row["last_used_at"],
                                "match_count": score,
                                "similarity": 0.0,
                                "combined_score": float(score),
                            },
                        ))

                if not scored_matches:
                    return []

                scored_matches.sort(
                    key=lambda item: (
                        item[0],
                        item[1]["last_used_at"] or "",
                        item[1]["created_at"] or "",
                    ),
                    reverse=True,
                )

                selected = [item[1] for item in scored_matches[:limit]]

                now_iso = datetime.now(UTC).isoformat()
                ids_to_update = [item["id"] for item in selected]
                placeholders = ",".join("?" for _ in ids_to_update)

                conn.execute(
                    f"UPDATE memories SET last_used_at = ? WHERE id IN ({placeholders})",
                    [now_iso] + ids_to_update,
                )
                conn.commit()

                for item in selected:
                    item["last_used_at"] = now_iso

                return selected

    def search(
        self,
        query: str,
        limit: int = 5,
        min_similarity: float | None = None,
    ) -> list[dict[str, Any]]:
        """Search stored memories combining semantic similarity, keyword matching, and recency.

        Scoring:
        1. Semantic similarity (cosine similarity between query and memory embeddings) in [0.0, 1.0].
        2. Keyword overlap ratio between query keywords and memory text in [0.0, 1.0].
        3. Recency boost based on last retrieval / creation in [0.0, 0.10].
        Combined score = (0.65 * sim) + (0.25 * keyword_ratio) + recency_boost.

        Filters candidates below min_similarity (from config.MIN_SIMILARITY, default 0.35).
        If the embedding model is unavailable, gracefully falls back to keyword matching without crashing.

        Automatically updates last_used_at to the current timestamp for returned items.

        Args:
            query: User search keywords or natural meaning query.
            limit: Maximum number of results to return (default: 5).
            min_similarity: Minimum cosine similarity threshold (default: config.MIN_SIMILARITY or 0.35).

        Returns:
            List of matching memory dicts with 'id', 'text', 'category',
            'created_at', 'last_used_at', 'match_count', 'similarity', and 'combined_score'.
        """
        cleaned_query = (query or "").strip()
        if not cleaned_query:
            return []

        if min_similarity is None:
            min_similarity = float(getattr(config, "MIN_SIMILARITY", 0.35))

        query_words = set(re.findall(r"\b\w+\b", cleaned_query.lower()))

        # Compute query embedding (returns None if fastembed is unavailable or model error)
        query_vec = compute_embedding(cleaned_query)

        # Fallback to pure keyword search if embedding model is unavailable
        if query_vec is None:
            return self._search_keyword_fallback(query_words, limit)

        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT id, text, category, created_at, last_used_at, embedding FROM memories"
                )
                rows = cursor.fetchall()
                if not rows:
                    return []

                scored_matches: list[tuple[float, dict[str, Any]]] = []

                for row in rows:
                    mem_words = set(re.findall(r"\b\w+\b", row["text"].lower()))
                    matches = query_words.intersection(mem_words) if query_words else set()
                    keyword_ratio = len(matches) / len(query_words) if query_words else 0.0

                    # Semantic similarity
                    blob = row["embedding"]
                    if blob:
                        row_vec = np.frombuffer(blob, dtype=np.float32)
                        sim = compute_similarity(query_vec, row_vec)
                    else:
                        sim = keyword_ratio

                    # Recency boost
                    recency_boost = _calculate_recency_boost(row["last_used_at"] or row["created_at"])

                    # Combined weighted score
                    combined_score = (0.65 * sim) + (0.25 * keyword_ratio) + recency_boost

                    # Filter: discard unrelated memories below minimum semantic similarity threshold
                    if sim < min_similarity and keyword_ratio == 0:
                        continue

                    scored_matches.append((
                        combined_score,
                        {
                            "id": row["id"],
                            "text": row["text"],
                            "category": row["category"],
                            "created_at": row["created_at"],
                            "last_used_at": row["last_used_at"],
                            "match_count": len(matches),
                            "similarity": round(sim, 4),
                            "combined_score": round(combined_score, 4),
                        },
                    ))

                if not scored_matches:
                    return []

                # Rank by combined_score DESC, last_used_at DESC
                scored_matches.sort(
                    key=lambda item: (item[0], item[1]["last_used_at"] or ""),
                    reverse=True,
                )

                selected = [item[1] for item in scored_matches[:limit]]

                # Update last_used_at for all returned memories
                now_iso = datetime.now(UTC).isoformat()
                ids_to_update = [item["id"] for item in selected]
                placeholders = ",".join("?" for _ in ids_to_update)

                conn.execute(
                    f"UPDATE memories SET last_used_at = ? WHERE id IN ({placeholders})",
                    [now_iso] + ids_to_update,
                )
                conn.commit()

                for item in selected:
                    item["last_used_at"] = now_iso

                return selected

    def list_all(self, category: str | None = None) -> list[dict[str, Any]]:
        """Return all memories ordered chronologically (newest first).

        Args:
            category: Optional filter for category ('preference', 'fact', etc.).

        Returns:
            List of memory dictionaries.
        """
        with self._lock:
            with self._get_connection() as conn:
                if category and category.strip().lower() in VALID_CATEGORIES:
                    cursor = conn.execute(
                        "SELECT id, text, category, created_at, last_used_at FROM memories WHERE category = ? ORDER BY created_at DESC",
                        (category.strip().lower(),),
                    )
                else:
                    cursor = conn.execute(
                        "SELECT id, text, category, created_at, last_used_at FROM memories ORDER BY created_at DESC"
                    )

                return [
                    {
                        "id": row["id"],
                        "text": row["text"],
                        "category": row["category"],
                        "created_at": row["created_at"],
                        "last_used_at": row["last_used_at"],
                    }
                    for row in cursor.fetchall()
                ]

    def delete(self, memory_id: int) -> bool:
        """Delete a memory by its unique ID.

        Args:
            memory_id: The ID of the memory to remove.

        Returns:
            True if a record was removed, False if not found.
        """
        with self._lock, self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            conn.commit()
            return cursor.rowcount > 0

    def delete_matching(self, query: str) -> int:
        """Delete all memories matching keywords in the query.

        Args:
            query: Search query identifying memories to delete.

        Returns:
            Count of deleted memories.
        """
        cleaned_query = (query or "").strip()
        if not cleaned_query:
            return 0

        query_words = set(re.findall(r"\b\w+\b", cleaned_query.lower()))
        matches = self._search_keyword_fallback(query_words, limit=500)
        if not matches:
            return 0

        ids_to_delete = [m["id"] for m in matches]
        placeholders = ",".join("?" for _ in ids_to_delete)

        with self._lock, self._get_connection() as conn:
            cursor = conn.execute(
                f"DELETE FROM memories WHERE id IN ({placeholders})",
                ids_to_delete,
            )
            conn.commit()
            return cursor.rowcount

    def clear_all(self) -> int:
        """Delete all memories in the store.

        Returns:
            Count of deleted memories.
        """
        with self._lock, self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM memories")
            conn.commit()
            return cursor.rowcount

    def export_to_text(self) -> str:
        """Generate a readable text report of all memories stored in the database.

        Returns:
            Formatted plain text string.
        """
        memories = self.list_all()
        count = len(memories)

        lines = [
            "=" * 64,
            f"  {getattr(config, 'ASSISTANT_NAME', 'Jarvis')} Long-Term Memory Store",
            f"  Total Memories: {count}/{MAX_MEMORIES_CAP}",
            "=" * 64,
        ]

        if not memories:
            lines.append("\n(No memories stored yet.)\n")
            return "\n".join(lines)

        # Group memories by category
        by_category: dict[str, list[dict[str, Any]]] = {}
        for m in memories:
            by_category.setdefault(m["category"], []).append(m)

        for cat in sorted(by_category.keys()):
            items = by_category[cat]
            lines.append(f"\n[{cat.upper()}] ({len(items)})")
            lines.append("-" * 40)
            for item in items:
                created = item["created_at"][:19].replace("T", " ")
                lines.append(f"  • [#{item['id']}] {item['text']}")
                lines.append(f"    (Saved: {created})")

        lines.append("\n" + "=" * 64 + "\n")
        return "\n".join(lines)
