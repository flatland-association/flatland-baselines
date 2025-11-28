import logging
import os
import shutil
import tempfile
import time
from pathlib import Path

import pytest
from testcontainers.compose import DockerCompose

from flatland.evaluators.trajectory_analysis import data_frame_for_trajectories, persist_data_frame_for_trajectories

logger = logging.getLogger(__name__)


@pytest.fixture
def _containers_fixture(environments) -> Path:
    newpath = tempfile.mkdtemp()
    try:
        # set env var ATTENDED to True if docker-compose.yml is already up and running
        if os.environ.get("ATTENDED", "False").lower() == "true":
            yield
            return Path(newpath)

        global basic

        start_time = time.time()

        env_file = Path(newpath) / ".env"
        with(env_file).open("w") as f:
            f.write(
                f"ENVIRONMENTS={environments}\n"
                f"DATA_DIR={newpath}"
            )
        basic = DockerCompose(context=".", env_file=str(env_file), compose_file_name=str(Path(__file__).parent.resolve() / "docker-compose.yml"))

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

            yield Path(newpath)

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


def _run(root_data_dir: Path):
    print(root_data_dir)
    data = data_frame_for_trajectories(root_data_dir)
    all_actions, all_trains_positions, all_trains_arrived, all_trains_rewards_dones_infos, env_stats, agent_stats = data
    analysis_dir = Path("../analysis-meta")
    persist_data_frame_for_trajectories(*data, output_dir=analysis_dir)
    # TODO assertions for environments_v2
    assert (all_trains_arrived["mean_normalized_reward"] == 1.0).all()
    assert (all_trains_arrived["success_rate"] == 1.0).all()


# https://docs.pytest.org/en/7.1.x/how-to/fixtures.html#override-a-fixture-with-direct-test-parametrization
@pytest.mark.parametrize('environments', ['debug-environments'])
def test_debug_environments(_containers_fixture):
    _run(_containers_fixture)


@pytest.mark.skip("run manually")
@pytest.mark.parametrize('environments', ['environments_v2'])
def test_environments_v2(_containers_fixture):
    _run(_containers_fixture)
