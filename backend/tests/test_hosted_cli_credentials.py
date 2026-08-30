import subprocess

import pytest

from hosted_app.cli_credentials import (
    Credential,
    CredentialStoreError,
    MacOSKeychainCredentialStore,
    _write_secret_with_expect,
)


def test_keychain_rejects_unsupported_platform_without_plaintext_fallback() -> None:
    store = MacOSKeychainCredentialStore(platform="linux")

    with pytest.raises(
        CredentialStoreError,
        match="macOS Keychain is required; this platform is unsupported",
    ):
        store.load()


def test_keychain_passes_secrets_to_prompt_writer_never_process_arguments() -> None:
    calls: list[tuple[list[str], str | None]] = []
    secret_writes: list[tuple[list[str], str]] = []
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
            service = args[args.index("-s") + 1]
            field_values = {
                "base_url": credential.base_url,
                "email": credential.email,
                "session_token": credential.session_token,
                "csrf_token": credential.csrf_token,
            }
            value = active_key
            for field, field_value in field_values.items():
                if service.endswith(f"annotationctl.session.{field}"):
                    value = field_value
                    break
            return subprocess.CompletedProcess(args, 0, stdout=value + "\n", stderr="")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    def secret_writer(args: list[str], secret: str) -> None:
        secret_writes.append((list(args), secret))

    store = MacOSKeychainCredentialStore(
        platform="darwin",
        runner=runner,
        secret_writer=secret_writer,
    )
    store.save(credential)
    assert store.load() == credential
    store.delete()

    flattened_args = " ".join(part for args, _ in calls for part in args)
    assert "session-secret" not in flattened_args
    assert "csrf-secret" not in flattened_args
    assert not any(input_value for _, input_value in calls)
    flattened_secrets = " ".join(secret for _, secret in secret_writes)
    assert "session-secret" in flattened_secrets
    assert "csrf-secret" in flattened_secrets
    assert len(secret_writes) == 5
    assert all(len(secret.encode()) < 128 for _, secret in secret_writes)
    writes = [args for args, _ in secret_writes]
    assert all(args[-1] == "-w" for args in writes)


def test_keychain_command_failures_are_explicit() -> None:
    def runner(args, **kwargs):
        raise subprocess.CalledProcessError(
            returncode=44,
            cmd=args,
            stderr="The specified item could not be found in the keychain.",
        )

    def secret_writer(args: list[str], secret: str) -> None:
        raise CredentialStoreError("macOS Keychain command failed")

    store = MacOSKeychainCredentialStore(
        platform="darwin",
        runner=runner,
        secret_writer=secret_writer,
    )
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


def test_expect_writer_uses_binary_io_and_keeps_secret_out_of_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def runner(args, **kwargs):
        captured["args"] = args
        captured.update(kwargs)
        return subprocess.CompletedProcess(args, 0, stdout=b"", stderr=b"\xc0")

    monkeypatch.setattr(subprocess, "run", runner)

    _write_secret_with_expect(["/usr/bin/security", "-w"], "session-secret")

    assert "session-secret" not in " ".join(captured["args"])
    assert captured["input"] == b"session-secret\n"
    assert "text" not in captured
    assert "retype password" in captured["args"][2]
    assert captured["args"][:2] == ["/usr/bin/expect", "-c"]
    assert len(captured["args"]) == 3
    assert captured["env"]["ANNOTATIONCTL_KEYCHAIN_ARG_0"] == "/usr/bin/security"
    assert "session-secret" not in " ".join(captured["env"].values())
