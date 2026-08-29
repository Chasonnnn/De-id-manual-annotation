from dataclasses import replace
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from uuid import uuid4

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, create_engine

from hosted_app.auth import (
    AuthenticationRequired,
    AuthManager,
    BootstrapNotAllowed,
    InvalidCredentials,
    UserRecord,
    WeakPassword,
)
from hosted_app.database import create_schema
from hosted_app.domain import UserState
from hosted_app.repository import HostedRepository


def test_first_admin_can_be_bootstrapped_with_argon2id_password() -> None:
    repository = InMemoryAuthRepository()
    auth = AuthManager(repository, now=lambda: datetime(2026, 8, 28, tzinfo=UTC))

    principal = auth.bootstrap_admin(
        " Admin@Example.COM ",
        "correct horse battery staple",
        display_name="Project Admin",
    )

    stored = repository.users_by_email["admin@example.com"]
    assert principal.email == "admin@example.com"
    assert principal.display_name == "Project Admin"
    assert principal.role == "admin"
    assert stored.password_hash.startswith("$argon2id$")
    assert "correct horse battery staple" not in stored.password_hash


def test_password_hash_creation_enforces_the_twelve_character_minimum() -> None:
    repository = InMemoryAuthRepository()
    auth = AuthManager(repository, now=lambda: datetime(2026, 8, 28, tzinfo=UTC))

    with pytest.raises(WeakPassword, match="at least 12 characters"):
        auth.create_password_hash("too-short")

    assert auth.create_password_hash("twelve-chars").startswith("$argon2id$")


def test_bootstrap_is_rejected_after_the_first_user_exists() -> None:
    repository = InMemoryAuthRepository()
    auth = AuthManager(repository, now=lambda: datetime(2026, 8, 28, tzinfo=UTC))
    auth.bootstrap_admin("first@example.com", "first password", display_name="First")

    with pytest.raises(BootstrapNotAllowed, match="already initialized"):
        auth.bootstrap_admin(
            "second@example.com",
            "second password",
            display_name="Second",
        )


def test_login_returns_an_opaque_token_and_stores_only_its_sha256_hash() -> None:
    now = datetime(2026, 8, 28, 12, tzinfo=UTC)
    repository = InMemoryAuthRepository()
    auth = AuthManager(repository, now=lambda: now, session_ttl=timedelta(hours=8))
    auth.bootstrap_admin("admin@example.com", "correct password", display_name="Admin")

    login = auth.login(" ADMIN@example.com ", "correct password")

    expected_hash = sha256(login.token.encode()).hexdigest()
    assert login.principal.email == "admin@example.com"
    assert login.expires_at == datetime(2026, 8, 28, 20, tzinfo=UTC)
    assert login.token not in repository.sessions_by_hash
    assert repository.sessions_by_hash[expected_hash].token_hash == expected_hash


def test_login_issues_a_session_bound_csrf_token() -> None:
    repository = InMemoryAuthRepository()
    auth = AuthManager(repository, now=lambda: datetime(2026, 8, 28, tzinfo=UTC))
    auth.bootstrap_admin("admin@example.com", "correct password", display_name="Admin")

    login = auth.login("admin@example.com", "correct password")

    assert login.csrf_token != login.token
    assert auth.validate_csrf(login.token, login.csrf_token) is True
    assert auth.validate_csrf(login.token, "wrong-token") is False


@pytest.mark.parametrize(
    ("email", "password"),
    [
        ("missing@example.com", "any password"),
        ("admin@example.com", "wrong password"),
        ("inactive@example.com", "correct password"),
    ],
)
def test_login_rejects_unknown_wrong_password_and_inactive_users_identically(
    email: str,
    password: str,
) -> None:
    repository = InMemoryAuthRepository()
    auth = AuthManager(repository, now=lambda: datetime(2026, 8, 28, tzinfo=UTC))
    auth.bootstrap_admin("admin@example.com", "correct password", display_name="Admin")
    inactive = replace(
        repository.users_by_email["admin@example.com"],
        id=str(uuid4()),
        email="inactive@example.com",
        state=UserState.DEACTIVATED,
    )
    repository.users_by_email[inactive.email] = inactive

    with pytest.raises(InvalidCredentials) as error:
        auth.login(email, password)

    assert str(error.value) == "Invalid email or password"
    assert repository.sessions_by_hash == {}


def test_authenticate_resolves_the_principal_from_an_opaque_token() -> None:
    repository = InMemoryAuthRepository()
    auth = AuthManager(repository, now=lambda: datetime(2026, 8, 28, tzinfo=UTC))
    auth.bootstrap_admin("admin@example.com", "correct password", display_name="Admin")
    login = auth.login("admin@example.com", "correct password")

    principal = auth.authenticate(login.token)

    assert principal == login.principal


def test_expired_session_is_rejected_and_removed() -> None:
    clock = [datetime(2026, 8, 28, 12, tzinfo=UTC)]
    repository = InMemoryAuthRepository()
    auth = AuthManager(
        repository,
        now=lambda: clock[0],
        session_ttl=timedelta(hours=1),
    )
    auth.bootstrap_admin("admin@example.com", "correct password", display_name="Admin")
    login = auth.login("admin@example.com", "correct password")
    clock[0] = login.expires_at

    with pytest.raises(AuthenticationRequired, match="Authentication required"):
        auth.authenticate(login.token)

    assert repository.sessions_by_hash == {}


def test_existing_session_is_rejected_after_user_is_deactivated() -> None:
    repository = InMemoryAuthRepository()
    auth = AuthManager(repository, now=lambda: datetime(2026, 8, 28, tzinfo=UTC))
    auth.bootstrap_admin("admin@example.com", "correct password", display_name="Admin")
    login = auth.login("admin@example.com", "correct password")
    repository.users_by_email["admin@example.com"] = replace(
        repository.users_by_email["admin@example.com"],
        state=UserState.DEACTIVATED,
    )

    with pytest.raises(AuthenticationRequired, match="Authentication required"):
        auth.authenticate(login.token)

    assert repository.sessions_by_hash == {}


def test_logout_invalidates_the_presented_session() -> None:
    repository = InMemoryAuthRepository()
    auth = AuthManager(repository, now=lambda: datetime(2026, 8, 28, tzinfo=UTC))
    auth.bootstrap_admin("admin@example.com", "correct password", display_name="Admin")
    login = auth.login("admin@example.com", "correct password")

    auth.logout(login.token)

    with pytest.raises(AuthenticationRequired, match="Authentication required"):
        auth.authenticate(login.token)


def test_password_change_invalidates_all_existing_sessions() -> None:
    repository = InMemoryAuthRepository()
    auth = AuthManager(repository, now=lambda: datetime(2026, 8, 28, tzinfo=UTC))
    auth.bootstrap_admin("admin@example.com", "correct password", display_name="Admin")
    first_login = auth.login("admin@example.com", "correct password")
    second_login = auth.login("admin@example.com", "correct password")

    auth.change_password(
        first_login.token,
        current_password="correct password",
        new_password="new secure password",
    )

    for old_token in (first_login.token, second_login.token):
        with pytest.raises(AuthenticationRequired):
            auth.authenticate(old_token)
    with pytest.raises(InvalidCredentials):
        auth.login("admin@example.com", "correct password")
    assert auth.login("admin@example.com", "new secure password").principal.email == (
        "admin@example.com"
    )


def test_rejected_password_change_preserves_the_current_session() -> None:
    repository = InMemoryAuthRepository()
    auth = AuthManager(repository, now=lambda: datetime(2026, 8, 28, tzinfo=UTC))
    auth.bootstrap_admin("admin@example.com", "correct password", display_name="Admin")
    login = auth.login("admin@example.com", "correct password")

    with pytest.raises(InvalidCredentials, match="Current password is incorrect"):
        auth.change_password(
            login.token,
            current_password="wrong password",
            new_password="new secure password",
        )
    with pytest.raises(WeakPassword, match="at least 12 characters"):
        auth.change_password(
            login.token,
            current_password="correct password",
            new_password="too-short",
        )

    assert auth.authenticate(login.token) == login.principal


def test_auth_manager_integrates_with_the_hosted_repository() -> None:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    create_schema(engine)
    repository = HostedRepository(lambda: Session(engine))
    auth = AuthManager(repository, now=lambda: datetime(2026, 8, 28, tzinfo=UTC))

    admin = auth.bootstrap_admin(
        "admin@example.com",
        "correct password",
        display_name="Project Admin",
    )
    login = auth.login("admin@example.com", "correct password")

    assert auth.authenticate(login.token) == admin


class InMemoryAuthRepository:
    def __init__(self) -> None:
        self.users_by_email = {}
        self.sessions_by_hash = {}

    def bootstrap_admin(
        self, *, email: str, display_name: str, password_hash: str
    ) -> UserRecord | None:
        if self.users_by_email:
            return None
        user = UserRecord(
            id=str(uuid4()),
            email=email,
            display_name=display_name,
            password_hash=password_hash,
            role="admin",
        )
        self.users_by_email[email] = user
        return user

    def get_user_by_email(self, email: str) -> UserRecord | None:
        return self.users_by_email.get(email)

    def get_user_by_id(self, user_id: str) -> UserRecord | None:
        return next(
            (user for user in self.users_by_email.values() if user.id == user_id),
            None,
        )

    def create_login_session(self, session) -> None:
        self.sessions_by_hash[session.token_hash] = session

    def get_login_session(self, token_hash: str):
        return self.sessions_by_hash.get(token_hash)

    def delete_login_session(self, token_hash: str) -> None:
        self.sessions_by_hash.pop(token_hash, None)

    def replace_password_and_delete_sessions(
        self, *, user_id: str, password_hash: str
    ) -> None:
        user = self.get_user_by_id(user_id)
        assert user is not None
        self.users_by_email[user.email] = replace(user, password_hash=password_hash)
        self.sessions_by_hash = {
            token_hash: session
            for token_hash, session in self.sessions_by_hash.items()
            if session.user_id != user_id
        }
