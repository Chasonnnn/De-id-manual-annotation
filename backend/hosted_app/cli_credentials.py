from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Protocol

SESSION_SERVICE = "edu.cornell.annotationctl.session"
ACTIVE_SERVICE = "edu.cornell.annotationctl.active"
ACTIVE_ACCOUNT = "active"
SESSION_FIELDS = ("base_url", "email", "session_token", "csrf_token")
KEYCHAIN_PROMPT_BYTE_LIMIT = 128


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
SecretWriter = Callable[[list[str], str], None]


def _write_secret_with_expect(args: list[str], secret: str) -> None:
    script = """
        log_user 0
        set timeout 10
        gets stdin secret
        set command {}
        for {set index 0} {$index < $env(ANNOTATIONCTL_KEYCHAIN_ARG_COUNT)} {incr index} {
            set key "ANNOTATIONCTL_KEYCHAIN_ARG_$index"
            lappend command [set env($key)]
        }
        spawn {*}$command
        expect {
            -nocase -glob "*password data*" { send -- "$secret\\r" }
            timeout { exit 124 }
            eof { catch wait result; exit [lindex $result 3] }
        }
        expect {
            -nocase -glob "*retype password*" {
                send -- "$secret\\r"
                exp_continue
            }
            timeout { exit 124 }
            eof { catch wait result; exit [lindex $result 3] }
        }
    """
    environment = os.environ.copy()
    environment["ANNOTATIONCTL_KEYCHAIN_ARG_COUNT"] = str(len(args))
    for index, argument in enumerate(args):
        environment[f"ANNOTATIONCTL_KEYCHAIN_ARG_{index}"] = argument
    try:
        subprocess.run(
            ["/usr/bin/expect", "-c", script],
            input=(secret + "\n").encode(),
            env=environment,
            capture_output=True,
            check=True,
        )
    except FileNotFoundError as error:
        raise CredentialStoreError("macOS Keychain command is unavailable") from error
    except subprocess.CalledProcessError as error:
        details = error.stderr.decode(errors="replace").replace(
            secret, "[REDACTED_SECRET]"
        )
        detail_suffix = f": {details.strip()}" if details.strip() else ""
        raise CredentialStoreError(
            f"macOS Keychain command failed (exit {error.returncode}){detail_suffix}"
        ) from error


class MacOSKeychainCredentialStore:
    def __init__(
        self,
        *,
        platform: str | None = None,
        runner: Runner = subprocess.run,
        secret_writer: SecretWriter = _write_secret_with_expect,
    ) -> None:
        self._platform = platform or sys.platform
        self._runner = runner
        self._secret_writer = secret_writer

    def save(self, credential: Credential) -> None:
        self._require_macos()
        profile_key = self._profile_key(credential.base_url, credential.email)
        values = credential.to_dict()
        for field in SESSION_FIELDS:
            self._write(
                service=f"{SESSION_SERVICE}.{field}",
                account=profile_key,
                secret=values[field],
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
        value: dict[str, str] = {}
        for field in SESSION_FIELDS:
            field_value = self._read(
                service=f"{SESSION_SERVICE}.{field}", account=profile_key
            )
            if field_value is None:
                raise CredentialStoreError("active Keychain credential is missing")
            value[field] = field_value
        return Credential.from_dict(value)

    def delete(self) -> None:
        self._require_macos()
        profile_key = self._read(service=ACTIVE_SERVICE, account=ACTIVE_ACCOUNT)
        if profile_key is None:
            return
        for field in SESSION_FIELDS:
            self._delete(service=f"{SESSION_SERVICE}.{field}", account=profile_key)
        self._delete(service=ACTIVE_SERVICE, account=ACTIVE_ACCOUNT)

    def _require_macos(self) -> None:
        if self._platform != "darwin":
            raise CredentialStoreError(
                "macOS Keychain is required; this platform is unsupported"
            )

    def _run(self, args: list[str]):
        try:
            return self._runner(
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
        if len(secret.encode()) >= KEYCHAIN_PROMPT_BYTE_LIMIT:
            raise CredentialStoreError(
                "Keychain credential field exceeds the secure prompt limit"
            )
        self._secret_writer(
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
            secret,
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
