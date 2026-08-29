from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Protocol

SESSION_SERVICE = "edu.cornell.annotationctl.session"
ACTIVE_SERVICE = "edu.cornell.annotationctl.active"
ACTIVE_ACCOUNT = "active"


class CredentialStoreError(RuntimeError):
    pass


@dataclass(frozen=True)
class Credential:
    base_url: str
    email: str
    session_token: str
    csrf_token: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> Credential:
        if not isinstance(value, dict):
            raise CredentialStoreError("stored Keychain credential is invalid")
        expected = {"base_url", "email", "session_token", "csrf_token"}
        if set(value) != expected or not all(
            isinstance(value[key], str) and value[key] for key in expected
        ):
            raise CredentialStoreError("stored Keychain credential is invalid")
        return cls(**value)


class CredentialStore(Protocol):
    def save(self, credential: Credential) -> None: ...

    def load(self) -> Credential | None: ...

    def delete(self) -> None: ...


class InMemoryCredentialStore:
    def __init__(self) -> None:
        self._credential: Credential | None = None

    def save(self, credential: Credential) -> None:
        self._credential = credential

    def load(self) -> Credential | None:
        return self._credential

    def delete(self) -> None:
        self._credential = None


Runner = Callable[..., subprocess.CompletedProcess[str]]


class MacOSKeychainCredentialStore:
    def __init__(
        self,
        *,
        platform: str | None = None,
        runner: Runner = subprocess.run,
    ) -> None:
        self._platform = platform or sys.platform
        self._runner = runner

    def save(self, credential: Credential) -> None:
        self._require_macos()
        profile_key = self._profile_key(credential.base_url, credential.email)
        self._write(
            service=SESSION_SERVICE,
            account=profile_key,
            secret=json.dumps(credential.to_dict(), separators=(",", ":")),
        )
        self._write(
            service=ACTIVE_SERVICE,
            account=ACTIVE_ACCOUNT,
            secret=profile_key,
        )

    def load(self) -> Credential | None:
        self._require_macos()
        profile_key = self._read(service=ACTIVE_SERVICE, account=ACTIVE_ACCOUNT)
        if profile_key is None:
            return None
        encoded = self._read(service=SESSION_SERVICE, account=profile_key)
        if encoded is None:
            raise CredentialStoreError("active Keychain credential is missing")
        try:
            value = json.loads(encoded)
        except json.JSONDecodeError as error:
            raise CredentialStoreError(
                "stored Keychain credential is invalid"
            ) from error
        return Credential.from_dict(value)

    def delete(self) -> None:
        self._require_macos()
        profile_key = self._read(service=ACTIVE_SERVICE, account=ACTIVE_ACCOUNT)
        if profile_key is None:
            return
        self._delete(service=SESSION_SERVICE, account=profile_key)
        self._delete(service=ACTIVE_SERVICE, account=ACTIVE_ACCOUNT)

    def _require_macos(self) -> None:
        if self._platform != "darwin":
            raise CredentialStoreError(
                "macOS Keychain is required; this platform is unsupported"
            )

    def _run(self, args: list[str], *, secret_input: str | None = None):
        try:
            return self._runner(
                args,
                input=secret_input,
                text=True,
                capture_output=True,
                check=True,
            )
        except FileNotFoundError as error:
            raise CredentialStoreError(
                "macOS Keychain command is unavailable"
            ) from error
        except subprocess.CalledProcessError as error:
            raise CredentialStoreError("macOS Keychain command failed") from error

    def _read(self, *, service: str, account: str) -> str | None:
        args = [
            "/usr/bin/security",
            "find-generic-password",
            "-s",
            service,
            "-a",
            account,
            "-w",
        ]
        try:
            result = self._runner(
                args,
                input=None,
                text=True,
                capture_output=True,
                check=True,
            )
        except FileNotFoundError as error:
            raise CredentialStoreError(
                "macOS Keychain command is unavailable"
            ) from error
        except subprocess.CalledProcessError as error:
            if error.returncode == 44:
                return None
            raise CredentialStoreError("macOS Keychain command failed") from error
        return result.stdout.rstrip("\n")

    def _write(self, *, service: str, account: str, secret: str) -> None:
        self._run(
            [
                "/usr/bin/security",
                "add-generic-password",
                "-U",
                "-s",
                service,
                "-a",
                account,
                "-w",
            ],
            secret_input=secret + "\n",
        )

    def _delete(self, *, service: str, account: str) -> None:
        self._run(
            [
                "/usr/bin/security",
                "delete-generic-password",
                "-s",
                service,
                "-a",
                account,
            ]
        )

    @staticmethod
    def _profile_key(base_url: str, email: str) -> str:
        return sha256(f"{base_url}\0{email}".encode()).hexdigest()
