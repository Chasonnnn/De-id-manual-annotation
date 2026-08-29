import json
import subprocess

import pytest

from hosted_app.cli_credentials import (
    Credential,
    CredentialStoreError,
    MacOSKeychainCredentialStore,
)


def test_keychain_rejects_unsupported_platform_without_plaintext_fallback() -> None:
    store = MacOSKeychainCredentialStore(platform="linux")

    with pytest.raises(
        CredentialStoreError,
        match="macOS Keychain is required; this platform is unsupported",
    ):
        store.load()


def test_keychain_passes_secrets_over_stdin_never_process_arguments() -> None:
    calls: list[tuple[list[str], str | None]] = []
    active_key = "profile-key"
    credential = Credential(
        base_url="https://annotation.example.com",
        email="admin@cornell.edu",
        session_token="session-secret",
        csrf_token="csrf-secret",
    )

    def runner(args, *, input=None, text, capture_output, check):
        calls.append((list(args), input))
        if args[1] == "find-generic-password":
            value = (
                active_key
                if any(part.endswith("annotationctl.active") for part in args)
                else json.dumps(credential.to_dict())
            )
            return subprocess.CompletedProcess(args, 0, stdout=value + "\n", stderr="")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    store = MacOSKeychainCredentialStore(platform="darwin", runner=runner)
    store.save(credential)
    assert store.load() == credential
    store.delete()

    flattened_args = " ".join(part for args, _ in calls for part in args)
    assert "session-secret" not in flattened_args
    assert "csrf-secret" not in flattened_args
    assert any(
        input_value and "session-secret" in input_value for _, input_value in calls
    )
    writes = [args for args, _ in calls if args[1] == "add-generic-password"]
    assert all(args[-1] == "-w" for args in writes)


def test_keychain_command_failures_are_explicit() -> None:
    def runner(args, **kwargs):
        raise subprocess.CalledProcessError(
            returncode=44,
            cmd=args,
            stderr="The specified item could not be found in the keychain.",
        )

    store = MacOSKeychainCredentialStore(platform="darwin", runner=runner)
    assert store.load() is None

    with pytest.raises(CredentialStoreError, match="macOS Keychain command failed"):
        store.save(
            Credential(
                base_url="https://annotation.example.com",
                email="admin@cornell.edu",
                session_token="session-secret",
                csrf_token="csrf-secret",
            )
        )
