"""Back up and adopt a pre-Alembic SQLite database without losing records."""

from datetime import UTC, datetime
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile

from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import MetaData, create_engine, inspect
from sqlalchemy.engine import make_url

from fastapi_blog.config import settings


ROOT = Path(__file__).resolve().parents[1]
INITIAL = "8df23eeb178f"


def migrate(url: str, *args: str) -> None:
    subprocess.run(
        [sys.executable, "-m", "alembic", "-c", str(ROOT / "alembic.ini"), *args],
        env={**os.environ, "DATABASE_URL": url},
        check=True,
    )


def main() -> None:
    url = make_url(settings.database_url)
    if url.get_backend_name() != "sqlite" or not url.database or url.database == ":memory:":
        raise SystemExit("This adoption script requires an existing SQLite database.")
    path = Path(url.database).resolve()
    if not path.is_file():
        raise SystemExit("Database does not exist. Use alembic upgrade head for a fresh database.")

    # Generate the baseline from the migration itself, independent of current models.
    with tempfile.TemporaryDirectory() as directory:
        baseline_path = Path(directory) / "baseline.db"
        migrate(f"sqlite+aiosqlite:///{baseline_path}", "upgrade", INITIAL)
        baseline_engine = create_engine(f"sqlite:///{baseline_path}")
        baseline = MetaData()
        try:
            baseline.reflect(baseline_engine, only=["users", "posts", "password_reset_tokens"])
        finally:
            baseline_engine.dispose()

        engine = create_engine(url.set(drivername="sqlite"))
        try:
            with engine.connect() as connection:
                inspector = inspect(connection)
                existing = set(inspector.get_table_names())
                if "alembic_version" in existing:
                    raise SystemExit("Database already has Alembic metadata. Use alembic upgrade head.")
                if not {"users", "posts"} <= existing:
                    raise SystemExit("Expected an existing blog database containing users and posts.")
                differences = compare_metadata(MigrationContext.configure(connection), baseline)
                missing = set(baseline.tables) - existing
                unexpected = [
                    difference for difference in differences
                    if not (
                        isinstance(difference, tuple)
                        and (
                            (difference[0] == "add_table" and difference[1].name in missing)
                            or (difference[0] == "add_index" and difference[1].table.name in missing)
                        )
                    )
                ]
                if unexpected:
                    raise SystemExit("Schema differs from the tutorial baseline; adoption stopped without changing the database.")

            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
            backup = path.with_name(f"{path.stem}.before-alembic-{timestamp}.db")
            source = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
            destination = sqlite3.connect(backup)
            try:
                source.backup(destination)
            finally:
                destination.close()
                source.close()
            print(f"Backup saved: {backup}", flush=True)
            # Only baseline tables are created here; later schema changes use Alembic.
            baseline.create_all(engine)
        finally:
            engine.dispose()

    migrate(settings.database_url, "stamp", INITIAL)
    migrate(settings.database_url, "upgrade", "head")
    migrate(settings.database_url, "check")
    print("Existing SQLite database adopted and upgraded successfully.")


if __name__ == "__main__":
    main()
