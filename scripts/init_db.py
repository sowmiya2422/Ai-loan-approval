"""Create the database and the first admin user.  Run:  python scripts/init_db.py"""
import sys
from getpass import getpass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import auth, db  # noqa: E402


def main():
    db.init_db()
    print("Database ready.")
    username = input("Admin username: ").strip()
    password = getpass("Admin password (min 8 chars): ")
    if password != getpass("Confirm password: "):
        sys.exit("Passwords do not match.")
    try:
        auth.create_user(username, password, role="admin")
    except ValueError as e:
        sys.exit(f"Error: {e}")
    print(f"Admin '{username}' created.")


if __name__ == "__main__":
    main()