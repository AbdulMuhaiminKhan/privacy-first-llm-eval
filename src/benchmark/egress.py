"""Egress guard: block every outbound connection from this Python process except to loopback.

Used by `python -m benchmark run --no-egress` (default ON) and by the privacy test. Scope of the claim:
it proves THIS process (document loading, retrieval, prompting, analysis) sends nothing off-machine.
It does not police the Ollama server process itself; see docs/PRIVACY.md for the OS-level check.
"""

from __future__ import annotations

import ipaddress
import socket
from contextlib import contextmanager

_original_connect = socket.socket.connect
_original_create_connection = socket.create_connection


class EgressBlocked(ConnectionError):
    pass


def _is_loopback(host: str) -> bool:
    if host in ("localhost",):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False  # unresolved hostnames are not allowed


def _guarded_connect(self: socket.socket, address, *args, **kwargs):
    if self.family in (socket.AF_INET, socket.AF_INET6):
        host = address[0]
        if not _is_loopback(str(host)):
            raise EgressBlocked(f"Blocked outbound connection to {address!r} (offline mode)")
    return _original_connect(self, address, *args, **kwargs)


def _guarded_create_connection(address, *args, **kwargs):
    if not _is_loopback(str(address[0])):
        raise EgressBlocked(f"Blocked outbound connection to {address!r} (offline mode)")
    return _original_create_connection(address, *args, **kwargs)


def install() -> None:
    socket.socket.connect = _guarded_connect  # type: ignore[method-assign]
    socket.create_connection = _guarded_create_connection  # type: ignore[assignment]


def uninstall() -> None:
    socket.socket.connect = _original_connect  # type: ignore[method-assign]
    socket.create_connection = _original_create_connection  # type: ignore[assignment]


@contextmanager
def no_egress():
    install()
    try:
        yield
    finally:
        uninstall()
