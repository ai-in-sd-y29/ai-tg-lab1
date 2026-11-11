"""Изолированная временная PostgreSQL для проверок, без изменения учебной БД."""

import secrets
import socket
import subprocess
import time
from contextlib import contextmanager
from dataclasses import replace


@contextmanager
def temporary_database(settings):
    name = "itmo-test-lab2-" + secrets.token_hex(6)
    password = secrets.token_urlsafe(32)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    def docker(*args):
        r = subprocess.run(["docker", *args], capture_output=True, timeout=180)
        if r.returncode:
            raise RuntimeError("Temporary Docker database command failed")
        return r.stdout

    try:
        docker(
            "run",
            "-d",
            "--name",
            name,
            "-p",
            f"127.0.0.1:{port}:5432",
            "-e",
            "POSTGRES_DB=bot",
            "-e",
            "POSTGRES_USER=bot",
            "-e",
            f"POSTGRES_PASSWORD={password}",
            "postgres:16-bookworm",
        )
        deadline = time.monotonic() + 60
        while True:
            try:
                docker("exec", name, "pg_isready", "-h", "127.0.0.1", "-U", "bot", "-d", "bot")
                break
            except RuntimeError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.5)
        yield replace(
            settings,
            postgres_host="127.0.0.1",
            postgres_port=port,
            postgres_db="bot",
            postgres_user="bot",
            postgres_password=password,
        )
    finally:
        # Удаляется только контейнер со случайным именем этого вызова.
        subprocess.run(["docker", "rm", "-f", "-v", name], capture_output=True, timeout=60)
