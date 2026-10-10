"""Exercise real Alembic commands against disposable databases."""

import os
from contextlib import closing
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
INITIAL = "8df23eeb178f"
HEAD = "03774587a3aa"


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "blog.db"
        self.env = {
            **os.environ,
            "DATABASE_URL": f"sqlite+aiosqlite:///{self.path}",
            "SECRET_KEY": "migration-test-secret-at-least-32-bytes",
        }

    def alembic(self, *args):
        result = subprocess.run(
            [sys.executable, "-m", "alembic", "-c", str(ROOT / "alembic.ini"), *args],
            cwd=self.temp.name,
            env=self.env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def test_fresh_database_upgrade_and_downgrade(self):
        self.alembic("upgrade", "head")
        self.alembic("check")
        with closing(sqlite3.connect(self.path)) as db:
            self.assertEqual(db.execute("SELECT version_num FROM alembic_version").fetchone()[0], HEAD)
            tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            self.assertTrue({"users", "posts", "password_reset_tokens"} <= tables)
        self.alembic("downgrade", INITIAL)
        with closing(sqlite3.connect(self.path)) as db:
            self.assertNotIn("likes", {row[1] for row in db.execute("PRAGMA table_info(posts)")})
        self.alembic("downgrade", "base")
        self.alembic("upgrade", "head")
        self.alembic("check")

    def test_existing_schema_preserves_records_and_defaults_likes(self):
        self.alembic("upgrade", INITIAL)
        with closing(sqlite3.connect(self.path)) as db:
            db.execute("INSERT INTO users (id, username, email, password_hash) VALUES (1, 'tester', 'tester@example.com', 'hash')")
            db.execute("INSERT INTO posts (id, title, content, user_id, date_posted) VALUES (1, 'Keep me', 'Existing content', 1, '2026-01-01 00:00:00')")
            db.execute("INSERT INTO password_reset_tokens (id, user_id, token_hash, expires_at, created_at) VALUES (1, 1, 'token-hash', '2026-01-02', '2026-01-01')")
            db.execute("DROP TABLE alembic_version")
            db.commit()
        self.alembic("stamp", INITIAL)
        self.alembic("upgrade", "head")
        self.alembic("check")
        with closing(sqlite3.connect(self.path)) as db:
            self.assertEqual(db.execute("SELECT title, content, likes FROM posts").fetchone(), ("Keep me", "Existing content", 0))
            self.assertEqual(db.execute("SELECT username FROM users").fetchone()[0], "tester")
            self.assertEqual(db.execute("SELECT token_hash FROM password_reset_tokens").fetchone()[0], "token-hash")

    def test_postgres_offline_sql_and_percent_encoded_credentials(self):
        self.env["DATABASE_URL"] = "postgresql+psycopg://tester:p%40ss@localhost/blog"
        sql = self.alembic("upgrade", "head", "--sql")
        self.assertIn("CREATE TABLE users", sql)
        self.assertIn("ALTER TABLE posts ADD COLUMN likes", sql)
        self.assertIn("DEFAULT '0' NOT NULL", sql)

    def test_adoption_of_older_database_creates_missing_table_and_backup(self):
        self.alembic("upgrade", INITIAL)
        with closing(sqlite3.connect(self.path)) as db:
            db.execute("DROP TABLE password_reset_tokens")
            db.execute("DROP TABLE alembic_version")
            db.execute("INSERT INTO users (id, username, email, password_hash) VALUES (1, 'tester', 'tester@example.com', 'hash')")
            db.commit()
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "adopt_sqlite.py")],
            cwd=self.temp.name, env=self.env, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.alembic("check")
        with closing(sqlite3.connect(self.path)) as db:
            self.assertEqual(db.execute("SELECT username FROM users").fetchone()[0], "tester")
            self.assertEqual(db.execute("SELECT count(*) FROM password_reset_tokens").fetchone()[0], 0)
        backups = list(Path(self.temp.name).glob("*.before-alembic-*.db"))
        self.assertEqual(len(backups), 1)
        with closing(sqlite3.connect(backups[0])) as db:
            self.assertEqual(db.execute("SELECT username FROM users").fetchone()[0], "tester")
            self.assertNotIn("likes", {r[1] for r in db.execute("PRAGMA table_info(posts)")})

    def test_adoption_rejects_schema_mismatch_without_changes(self):
        self.alembic("upgrade", INITIAL)
        with closing(sqlite3.connect(self.path)) as db:
            db.execute("DROP TABLE alembic_version")
            db.execute("ALTER TABLE posts ADD COLUMN unexpected TEXT")
            db.commit()
        before = self.path.read_bytes()
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "adopt_sqlite.py")],
            cwd=self.temp.name, env=self.env, capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Schema differs", result.stderr)
        self.assertEqual(self.path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
