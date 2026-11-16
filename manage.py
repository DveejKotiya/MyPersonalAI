"""Management utility for Jarvis (Phase 5 final review).

Provides commands to create comprehensive backups and cleanly wipe all local data:
- Backup: Exports memories, logs, notes, and token usage into a timestamped zip.
- Wipe: Permanently removes all stored user data, logs, notes, and usage history.

Usage:
    python manage.py backup [--dest DEST_DIR]
    python manage.py wipe [--force]
"""

import argparse
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

import config
from memory import MemoryStore


def get_project_root() -> Path:
    """Return the base directory for the Jarvis project."""
    return getattr(config, "BASE_DIR", Path(__file__).resolve().parent)


def backup_data(dest_dir: str | Path | None = None) -> Path:
    """Export all stored memories, notes, logs, and usage data into a zip archive.

    Args:
        dest_dir: Target folder for the zip file (defaults to project 'backups/').

    Returns:
        The Path to the created backup zip file.
    """
    root = get_project_root()
    if dest_dir is None:
        target_dir = root / "backups"
    else:
        target_dir = Path(dest_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_zip_path = target_dir / f"jarvis_backup_{timestamp}.zip"

    files_to_backup: list[tuple[Path, str]] = []

    # 1. Memory Database and Text Export
    memory_db = root / "data" / "memory.db"
    if memory_db.is_file():
        files_to_backup.append((memory_db, "data/memory.db"))

    # Export human-readable text dump of memories
    try:
        store = MemoryStore(db_path=memory_db if memory_db.is_file() else None)
        text_export = store.export_to_text()
        temp_mem_txt = target_dir / f"memories_dump_{timestamp}.txt"
        with open(temp_mem_txt, "w", encoding="utf-8") as f:
            f.write(text_export)
        files_to_backup.append((temp_mem_txt, "memories_export.txt"))
    except Exception as mem_err:
        print(f"[Backup Warning] Could not export memories text: {mem_err}")
        temp_mem_txt = None

    # 2. Token Usage & State
    usage_csv = root / "data" / "usage.csv"
    if usage_csv.is_file():
        files_to_backup.append((usage_csv, "data/usage.csv"))

    state_json = root / "data" / "state.json"
    if state_json.is_file():
        files_to_backup.append((state_json, "data/state.json"))

    # 3. Notes
    notes_md = root / "notes.md"
    if notes_md.is_file():
        files_to_backup.append((notes_md, "notes.md"))

    # 4. Logs (actions.log, jarvis.log, and backups)
    logs_dir = root / "logs"
    if logs_dir.is_dir():
        for log_file in logs_dir.glob("*"):
            if log_file.is_file():
                files_to_backup.append((log_file, f"logs/{log_file.name}"))

    # 5. Screenshots
    screenshots_dir = root / "screenshots"
    if screenshots_dir.is_dir():
        for ss in screenshots_dir.glob("*"):
            if ss.is_file():
                files_to_backup.append((ss, f"screenshots/{ss.name}"))

    # Create Zip Archive
    with zipfile.ZipFile(backup_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        for file_path, arcname in files_to_backup:
            try:
                zipf.write(file_path, arcname=arcname)
            except Exception as err:
                print(f"[Backup Warning] Could not add '{file_path.name}' to archive: {err}")

    # Remove temporary text export file
    if temp_mem_txt and temp_mem_txt.is_file():
        try:
            temp_mem_txt.unlink()
        except OSError:
            pass

    size_kb = backup_zip_path.stat().st_size / 1024.0
    print(f"\n[Backup Complete] Created archive ({size_kb:.1f} KB):")
    print(f"  {backup_zip_path}\n")
    print(f"Backed up {len(files_to_backup)} item(s) including memories, usage, notes, and logs.")
    return backup_zip_path


def wipe_data(force: bool = False) -> bool:
    """Permanently delete all user memories, notes, logs, and token usage data.

    Args:
        force: If True, skips the interactive terminal confirmation.

    Returns:
        True if wipe completed, False if cancelled.
    """
    if not force:
        print("\n" + "!" * 64)
        print("  WARNING: This will permanently erase ALL Jarvis data:")
        print("  - Long-term memories (data/memory.db)")
        print("  - Token usage history (data/usage.csv)")
        print("  - Application state (data/state.json)")
        print("  - OAuth credentials and tokens (data/secrets/)")
        print("  - User notes (notes.md)")
        print("  - Action & system logs (logs/)")
        print("  - Captured screenshots (screenshots/)")
        print("!" * 64 + "\n")

        try:
            confirm_input = input("Are you absolutely sure you want to proceed? Type 'yes' to wipe: ").strip().lower()
            if confirm_input != "yes":
                print("\nWipe cancelled. No data was deleted.\n")
                return False
        except (KeyboardInterrupt, EOFError):
            print("\nWipe cancelled.\n")
            return False

    root = get_project_root()
    deleted_items: list[str] = []

    # 1. Clear Long-Term Memory SQLite
    data_dir = root / "data"
    if data_dir.is_dir():
        for item in data_dir.glob("*"):
            try:
                if item.is_file():
                    item.unlink()
                    deleted_items.append(f"data/{item.name}")
                elif item.is_dir():
                    shutil.rmtree(item)
                    deleted_items.append(f"data/{item.name}/")
            except Exception as err:
                print(f"[Wipe Warning] Could not delete '{item}': {err}")

    # 2. Reset Notes File
    notes_md = root / "notes.md"
    if notes_md.is_file():
        try:
            notes_md.unlink()
            deleted_items.append("notes.md")
        except Exception as err:
            print(f"[Wipe Warning] Could not remove notes.md: {err}")

    # 3. Clean Logs Directory
    logs_dir = root / "logs"
    if logs_dir.is_dir():
        for item in logs_dir.glob("*"):
            try:
                if item.is_file():
                    item.unlink()
                    deleted_items.append(f"logs/{item.name}")
                elif item.is_dir():
                    shutil.rmtree(item)
                    deleted_items.append(f"logs/{item.name}/")
            except Exception as err:
                print(f"[Wipe Warning] Could not delete '{item}': {err}")

    # 4. Clean Screenshots Directory
    screenshots_dir = root / "screenshots"
    if screenshots_dir.is_dir():
        for item in screenshots_dir.glob("*"):
            try:
                if item.is_file():
                    item.unlink()
                    deleted_items.append(f"screenshots/{item.name}")
            except Exception as err:
                print(f"[Wipe Warning] Could not delete screenshot '{item}': {err}")

    # Re-initialize clean directory structure
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "data" / "secrets").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "screenshots").mkdir(parents=True, exist_ok=True)

    print("\n[Wipe Complete] All local user data and logs have been wiped.")
    print(f"Cleaned {len(deleted_items)} file(s)/directory(ies).")
    return True


def main() -> None:
    """Parse CLI arguments for backup and wipe actions."""
    parser = argparse.ArgumentParser(
        description="Jarvis Maintenance & Privacy Manager (backup / wipe)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Action to execute")

    # backup subcommand
    backup_parser = subparsers.add_parser("backup", help="Export memory, usage, notes, and logs to a zip archive")
    backup_parser.add_argument("--dest", type=str, default=None, help="Destination directory for the backup archive")

    # wipe subcommand
    wipe_parser = subparsers.add_parser("wipe", help="Permanently erase all memory, usage, notes, and logs")
    wipe_parser.add_argument("--force", action="store_true", help="Bypass interactive confirmation prompt")

    args = parser.parse_args()

    if args.command == "backup":
        backup_data(dest_dir=args.dest)
    elif args.command == "wipe":
        wipe_data(force=args.force)


if __name__ == "__main__":
    main()
