import socket

import pytest

from pictor import cli
from pictor.config import Config


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_the_preferred_port_is_used_when_free():
    port = free_port()
    assert cli.find_port("127.0.0.1", port) == port


def test_a_busy_port_moves_to_the_next_one():
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        taken = busy.getsockname()[1]
        assert cli.find_port("127.0.0.1", taken) > taken


def test_no_free_port_is_a_clear_error():
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        with pytest.raises(SystemExit, match="No free port"):
            cli.find_port("127.0.0.1", busy.getsockname()[1], attempts=1)


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as stop:
        cli.main(["--version"])
    assert stop.value.code == 0
    assert capsys.readouterr().out.startswith("pictor ")


def test_doctor_reports_hardware_and_models_without_ollama(monkeypatch, capsys):
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:9")  # nothing listens on the discard port
    code = cli.main(["doctor", "--model", "gemma4:12b"])
    output = capsys.readouterr().out
    assert code == 1  # Ollama is not answering
    assert "Hardware" in output and "Model budget" in output
    assert "RECOMMENDED" in output
    assert "Chosen model: gemma4:12b" in output
    assert Config.load().model == "gemma4:12b"  # --model is remembered
