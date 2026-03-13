import logging
import os
import shutil
import tempfile
import time
from pathlib import Path

import pandas as pd
import pytest
from testcontainers.compose import DockerCompose

from flatland_baselines.deadlock_avoidance_heuristic.offline_demo.test_offline import verify_online_offline_calibration

logger = logging.getLogger(__name__)


@pytest.fixture
def _containers_fixture(environments) -> Path:
    newpath = tempfile.mkdtemp()
    try:
        # set env var ATTENDED to True if docker-compose.yml is already up and running
        temp = Path(newpath)

        (temp / "analysis_data_dir").mkdir(parents=True, exist_ok=True)
        (temp / "results").mkdir(parents=True, exist_ok=True)
        (temp / "actions").mkdir(parents=True, exist_ok=True)
        (temp / "episodes").mkdir(parents=True, exist_ok=True)
        (temp / "visualizations").mkdir(parents=True, exist_ok=True)

        if os.environ.get("ATTENDED", "False").lower() == "true":
            yield
            return temp

        global basic

        start_time = time.time()

        env_file = temp / ".env"
        with(env_file).open("w") as f:
            f.write(
                f"ENVIRONMENTS={environments}\n"
                f"DATA_DIR={newpath}"
            )
        basic = DockerCompose(context=".", env_file=str(env_file), compose_file_name=str(Path(__file__).parent.resolve() / "docker-compose.yml"), wait=False)

        logger.info("/ start docker compose down")
        basic.stop()
        duration = time.time() - start_time
        logger.info(f"\\ end docker compose down. Took {duration:.2f} seconds.")
        start_time = time.time()
        logger.info("/ start docker compose up")
        try:
            basic.start()
            duration = time.time() - start_time
            logger.info(f"\\ end docker compose up. Took {duration:.2f} seconds.")

            yield temp

            # TODO workaround for testcontainers not supporting streaming to logger
            start_time = time.time()
            logger.info("/ start get docker compose logs")
            stdout, stderr = basic.get_logs()
            logger.info("stdout from docker compose")
            logger.info(stdout)
            logger.warning("stderr from docker compose")
            logger.warning(stderr)
            duration = time.time() - start_time
            logger.info(f"\\ end get docker compose logs. Took {duration:.2f} seconds.")

            start_time = time.time()
            logger.info("/ start docker compose down")
            basic.stop()
            duration = time.time() - start_time
            logger.info(f"\\ end docker down. Took {duration:.2f} seconds.")
        except BaseException as e:
            print("An exception occurred during running docker compose:")
            print(e)
            stdout, stderr = basic.get_logs()
            print(stdout)
            print(stderr)
            raise e
    finally:
        shutil.rmtree(newpath, ignore_errors=True)


# https://docs.pytest.org/en/7.1.x/how-to/fixtures.html#override-a-fixture-with-direct-test-parametrization
@pytest.mark.parametrize('environments', ['environments_v2'])
@pytest.mark.slow
def test_online_calibrated_against_offline_legacy_way(_containers_fixture):
    """
    Verify online evaluation yields the same result as offline evaluation in legacy way with current code basis.
    """

    root_data_dir = _containers_fixture
    print(root_data_dir)
    print(list(root_data_dir.rglob("**/*")))

    df = pd.read_csv(root_data_dir / "results" / "results.csv")

    sum_normalized_reward = df["normalized_reward"].sum()
    mean_normalized_reward = df["normalized_reward"].mean()
    mean_percentage_complete = df["percentage_complete"].mean()
    mean_reward = df['reward'].mean()

    verify_online_offline_calibration(mean_normalized_reward, mean_percentage_complete, mean_reward, sum_normalized_reward)
