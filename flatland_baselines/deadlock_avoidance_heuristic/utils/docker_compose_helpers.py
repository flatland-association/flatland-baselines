from testcontainers.compose import DockerCompose


def _print_output(stdout: str, stderr: str) -> None:
    print(f"stdout: {stdout}")
    print(f"stderr: {stderr}")


def _dump_compose_logs(basic: DockerCompose) -> None:
    try:
        stdout, stderr = basic.get_logs()
        _print_output(stdout, stderr)
    except Exception as e:
        print(f"Could not get logs from docker compose: {e}")
