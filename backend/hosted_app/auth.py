from __future__ import annotations

import secrets
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from hmac import compare_digest
from hmac import new as new_hmac
from typing import Protocol

from pwdlib import PasswordHash

from .domain import ActivationTokenRecord, LoginSessionRecord, UserState


class BootstrapNotAllowed(RuntimeError):
    pass


class InvalidCredentials(RuntimeError):
    pass


class AuthenticationRequired(RuntimeError):
    pass


class WeakPassword(ValueError):
    pass


class EmailNotAllowed(ValueError):
    pass


class InvalidActivationToken(RuntimeError):
    pass


@dataclass(frozen=True)
class UserRecord:
    id: str
    email: str
    display_name: str
    password_hash: str | None
    role: str
    state: UserState = UserState.ACTIVE


@dataclass(frozen=True)
class AuthenticatedPrincipal:
    id: str
    email: str
    display_name: str
    role: str
    state: UserState


@dataclass(frozen=True)
class LoginResult:
    token: str
    csrf_token: str
    expires_at: datetime
    principal: AuthenticatedPrincipal


@dataclass(frozen=True)
class ActivationResult:
    token: str
    expires_at: datetime
    principal: AuthenticatedPrincipal


class AuthRepository(Protocol):
    def bootstrap_admin(
        self, *, email: str, display_name: str, password_hash: str
    ) -> UserRecord | None: ...

    def get_user_by_email(self, email: str) -> UserRecord | None: ...

    def get_user_by_id(self, user_id: str) -> UserRecord | None: ...

    def create_login_session(self, session: LoginSessionRecord) -> None: ...

    def get_login_session(self, token_hash: str) -> LoginSessionRecord | None: ...

    def delete_login_session(self, token_hash: str) -> None: ...

    def replace_password_and_delete_sessions(
        self, *, user_id: str, password_hash: str
    ) -> None: ...

    def create_pending_user_with_activation(
        self,
        *,
        email: str,
        display_name: str,
        role: str,
        activation: ActivationTokenRecord,
        admin_id: str,
    ) -> UserRecord: ...

    def activate_user(
        self,
        *,
        token_hash: str,
        password_hash: str,
        now: datetime,
    ) -> UserRecord | None: ...

    def reset_user_password(
        self,
        *,
        user_id: str,
        activation: ActivationTokenRecord,
        admin_id: str,
    ) -> UserRecord: ...


class AuthManager:
    def __init__(
        self,
        repository: AuthRepository,
        *,
        now: Callable[[], datetime] | None = None,
        session_ttl: timedelta = timedelta(hours=12),
        activation_ttl: timedelta = timedelta(hours=24),
    ) -> None:
        self._repository = repository
        self._now = now or (lambda: datetime.now(UTC))
        self._session_ttl = session_ttl
        self._activation_ttl = activation_ttl
        self._password_hash = PasswordHash.recommended()
        self._unknown_user_hash = self._password_hash.hash(secrets.token_urlsafe(32))

    def bootstrap_admin(
        self,
        email: str,
        password: str,
        *,
        display_name: str,
    ) -> AuthenticatedPrincipal:
        normalized_email = self.normalize_account_email(email)
        password_hash = self.create_password_hash(password)
        user = self._repository.bootstrap_admin(
            email=normalized_email,
            display_name=display_name.strip(),
            password_hash=password_hash,
        )
        if user is None:
            raise BootstrapNotAllowed("Application is already initialized")
        return AuthenticatedPrincipal(
            id=user.id,
            email=user.email,
            display_name=user.display_name,
            role=user.role,
            state=user.state,
        )

    def create_password_hash(self, password: str) -> str:
        if len(password) < 12:
            raise WeakPassword("Password must be at least 12 characters")
        return self._password_hash.hash(password)

    def normalize_account_email(self, email: str) -> str:
        normalized_email = email.strip().lower()
        local_part, separator, domain = normalized_email.rpartition("@")
        if not separator or not local_part or not domain:
            raise EmailNotAllowed("valid email is required")
        return normalized_email

    def login(self, email: str, password: str) -> LoginResult:
        try:
            normalized_email = self.normalize_account_email(email)
        except EmailNotAllowed:
            normalized_email = ""
        user = (
            self._repository.get_user_by_email(normalized_email)
            if normalized_email
            else None
        )
        candidate_hash = (
            user.password_hash
            if user and user.password_hash
            else self._unknown_user_hash
        )
        valid_password = self._password_hash.verify(password, candidate_hash)
        if user is None or user.state != UserState.ACTIVE or not valid_password:
            raise InvalidCredentials("Invalid email or password")

        token = secrets.token_urlsafe(32)
        expires_at = self._now() + self._session_ttl
        session = LoginSessionRecord(
            token_hash=sha256(token.encode()).hexdigest(),
            user_id=user.id,
            expires_at=expires_at,
        )
        self._repository.create_login_session(session)
        return LoginResult(
            token=token,
            csrf_token=self.csrf_token(token),
            expires_at=expires_at,
            principal=AuthenticatedPrincipal(
                id=user.id,
                email=user.email,
                display_name=user.display_name,
                role=user.role,
                state=user.state,
            ),
        )

    def validate_csrf(self, token: str, csrf_token: str) -> bool:
        return compare_digest(self.csrf_token(token), csrf_token)

    @staticmethod
    def csrf_token(token: str) -> str:
        return new_hmac(
            token.encode(),
            b"annotation-csrf-v1",
            sha256,
        ).hexdigest()

    def authenticate(self, token: str) -> AuthenticatedPrincipal:
        token_hash = sha256(token.encode()).hexdigest()
        session = self._repository.get_login_session(token_hash)
        if session is None:
            raise AuthenticationRequired("Authentication required")
        if session.expires_at <= self._now():
            self._repository.delete_login_session(token_hash)
            raise AuthenticationRequired("Authentication required")
        user = self._repository.get_user_by_id(session.user_id)
        if user is None or user.state != UserState.ACTIVE:
            self._repository.delete_login_session(token_hash)
            raise AuthenticationRequired("Authentication required")
        return AuthenticatedPrincipal(
            id=user.id,
            email=user.email,
            display_name=user.display_name,
            role=user.role,
            state=user.state,
        )

    def invite_user(
        self,
        *,
        email: str,
        display_name: str,
        role: str,
        admin_id: str,
    ) -> ActivationResult:
        normalized_email = self.normalize_account_email(email)
        token = secrets.token_urlsafe(32)
        expires_at = self._now() + self._activation_ttl
        user = self._repository.create_pending_user_with_activation(
            email=normalized_email,
            display_name=display_name.strip(),
            role=role,
            activation=ActivationTokenRecord(
                token_hash=sha256(token.encode()).hexdigest(),
                expires_at=expires_at,
            ),
            admin_id=admin_id,
        )
        return ActivationResult(
            token=token,
            expires_at=expires_at,
            principal=self._principal(user),
        )

    def activate(self, token: str, password: str) -> AuthenticatedPrincipal:
        password_hash = self.create_password_hash(password)
        user = self._repository.activate_user(
            token_hash=sha256(token.encode()).hexdigest(),
            password_hash=password_hash,
            now=self._now(),
        )
        if user is None:
            raise InvalidActivationToken("Activation link is invalid or expired")
        return self._principal(user)

    def reset_password(self, user_id: str, *, admin_id: str) -> ActivationResult:
        token = secrets.token_urlsafe(32)
        expires_at = self._now() + self._activation_ttl
        user = self._repository.reset_user_password(
            user_id=user_id,
            activation=ActivationTokenRecord(
                token_hash=sha256(token.encode()).hexdigest(),
                expires_at=expires_at,
            ),
            admin_id=admin_id,
        )
        return ActivationResult(
            token=token,
            expires_at=expires_at,
            principal=self._principal(user),
        )

    @staticmethod
    def _principal(user: UserRecord) -> AuthenticatedPrincipal:
        return AuthenticatedPrincipal(
            id=user.id,
            email=user.email,
            display_name=user.display_name,
            role=user.role,
            state=user.state,
        )

    def logout(self, token: str) -> None:
        self._repository.delete_login_session(sha256(token.encode()).hexdigest())

    def change_password(
        self,
        token: str,
        *,
        current_password: str,
        new_password: str,
    ) -> None:
        principal = self.authenticate(token)
        user = self._repository.get_user_by_id(principal.id)
        if user is None or not self._password_hash.verify(
            current_password,
            user.password_hash,
        ):
            raise InvalidCredentials("Current password is incorrect")
        new_password_hash = self.create_password_hash(new_password)
        self._repository.replace_password_and_delete_sessions(
            user_id=user.id,
            password_hash=new_password_hash,
        )
