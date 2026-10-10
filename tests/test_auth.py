import pytest

from src import auth, db


@pytest.fixture()
def db_path(tmp_path):
    path = tmp_path / "test.db"
    db.init_db(path)
    return path


def test_create_and_login(db_path):
    auth.create_user("alice", "secret123", db_path=db_path)
    user = auth.authenticate("alice", "secret123", db_path=db_path)
    assert user is not None
    assert user["username"] == "alice" and user["role"] == "officer"


def test_wrong_password_and_unknown_user(db_path):
    auth.create_user("alice", "secret123", db_path=db_path)
    assert auth.authenticate("alice", "wrongpass", db_path=db_path) is None
    assert auth.authenticate("nobody", "secret123", db_path=db_path) is None


def test_username_case_insensitive_and_unique(db_path):
    auth.create_user("alice", "secret123", db_path=db_path)
    with pytest.raises(ValueError):
        auth.create_user("ALICE", "another123", db_path=db_path)


def test_weak_inputs_rejected(db_path):
    with pytest.raises(ValueError):
        auth.create_user("bob", "short", db_path=db_path)
    with pytest.raises(ValueError):
        auth.create_user("b!", "longenough1", db_path=db_path)


def test_hash_is_salted_and_not_plaintext():
    h1, h2 = auth.hash_password("secret123"), auth.hash_password("secret123")
    assert h1 != h2 and "secret123" not in h1 