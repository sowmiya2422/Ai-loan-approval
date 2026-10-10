"""User registration and login with salted PBKDF2 password hashing."""
import hashlib
import hmac
import os
import re
import sqlite3

from src import db

ALGORITHM = "pbkdf2_sha256"
ITERATIONS = 600_000
ROLES = ("admin", "officer")
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.]{3,30}$")
MIN_PASSWORD_LEN = 8


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, ITERATIONS)
    return f"{ALGORITHM}${ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, iterations, salt_hex, hash_hex = stored.split("$")
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), bytes.fromhex(salt_hex), int(iterations)
        )
        return hmac.compare_digest(digest.hex(), hash_hex)
    except (ValueError, TypeError):
        return False


# Used so a wrong username takes about as long as a wrong password.
_DUMMY_HASH = hash_password("dummy-password")


def validate_credentials(username: str, password: str) -> None:
    if not USERNAME_RE.match(username):
        raise ValueError("Username must be 3-30 characters: letters, numbers, _ or . only.")
    if len(password) < MIN_PASSWORD_LEN:
        raise ValueError(f"Password must be at least {MIN_PASSWORD_LEN} characters.")


def create_user(username: str, password: str, role: str = "officer", db_path=None) -> int:
    username = username.strip()
    validate_credentials(username, password)
    if role not in ROLES:
        raise ValueError(f"Role must be one of {ROLES}.")
    try:
        with db.get_connection(db_path) as conn:
            cur = conn.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                (username, hash_password(password), role),
            )
            user_id = cur.lastrowid
            assert user_id is not None
            return user_id
    except sqlite3.IntegrityError:
        raise ValueError("That username is already taken.") from None


def authenticate(username: str, password: str, db_path=None):
    """Return {'id', 'username', 'role'} on success, else None."""
    with db.get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT id, username, password_hash, role, is_active FROM users WHERE username = ?",
            (username.strip(),),
        ).fetchone()
    if row is None:
        verify_password(password, _DUMMY_HASH)
        return None
    if not row["is_active"] or not verify_password(password, row["password_hash"]):
        return None
    return {"id": row["id"], "username": row["username"], "role": row["role"]}