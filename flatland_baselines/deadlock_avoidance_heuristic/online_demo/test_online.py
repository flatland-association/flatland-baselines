import logging
import os
import shutil
import tempfile
import time
from pathlib import Path

import pandas as pd
import pytest
from testcontainers.compose import DockerCompose

from flatland_baselines.deadlock_avoidance_heuristic.offline_demo.test_offline import verify_online_offline_calibration_envs_v2, \
    verify_online_offline_calibration_envs_v3_trunc

logger = logging.getLogger(__name__)


@pytest.fixture(scope="function")
def _containers_fixture(environments, seed, baselines_ref, rl_ref) -> Path:
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
                f"DATA_DIR={newpath}\n"
                f"FLATLAND_EVALUATION_RANDOM_SEED={seed}\n"
                f"FLATLAND_BASELINES_REF={baselines_ref}\n"
                f"FLATLAND_RL_REF={rl_ref}\n"
            )
        basic = DockerCompose(context=".", env_file=str(env_file), compose_file_name=str(Path(__file__).parent.resolve() / "docker-compose.yml"), wait=False, )

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

            yield temp, seed

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
@pytest.mark.parametrize('environments,seed,baselines_ref,rl_ref', [
    ('environments_v2', "1001", "", ""),
    # -> same result as (3)

    #############################################################
    # (3) one commit after new get_k_shortest_path (https://github.com/flatland-association/flatland-baselines/pull/38/ and after last get_k_shortest_path fix https://github.com/flatland-association/flatland-rl/pull/345 and needing also https://github.com/flatland-association/flatland-rl/pull/317 because of interface changes)
    # ('environments_v2', "1001", "b79af610cc4d1b5aa331cabc75e4812b504e18cd","3be36d8f0e0aff1a05ad6b3caacc72386f8e9b74"),
    # Mean Reward : -3427.52
    # Sum Normalized Reward : 43.141500749639434 (primary score)
    # Mean Percentage Complete : 0.671 (secondary score)
    # Mean Normalized Reward : 0.86283
    # -> breaking change due to new get_k_shortest_path in flatland-baselines

    #############################################################
    # (2b) one commit before new get_k_shortest_path (https://github.com/flatland-association/flatland-baselines/pull/38/ and one commit before first get_k_shortest_path fix https://github.com/flatland-association/flatland-rl/pull/335)
    # ('environments_v2', "1001", "0241520c062095823b442c2cad2f6b6386e6aec2","c3c9c9db0df804a90cb22d4d0c22c96b21f4fba8"),
    # Mean Reward : -3545.04
    # Sum Normalized Reward : 43.01006646037365 (primary score)
    # Mean Percentage Complete : 0.678 (secondary score)
    # Mean Normalized Reward : 0.8602
    # -> same result as (2)

    #############################################################
    # (2) just after breaking change no rewards after (single) agent is done https://github.com/flatland-association/flatland-rl/pull/302/files
    # ('environments_v2', "1001", "0241520c062095823b442c2cad2f6b6386e6aec2","db80be21a4b79c1bd4f32bcca3a0e50448f711cc"),
    # Mean Reward : -3545.04
    # Sum Normalized Reward : 43.01006646037365 (primary score)
    # Mean Percentage Complete : 0.678 (secondary score)
    # Mean Normalized Reward : 0.8602
    # -> breaking change due to rewards fix in flatland-rl

    #############################################################
    # (1) flatland-rl just before breaking change no rewards after (single) agent is done https://github.com/flatland-association/flatland-rl/pull/302/files
    # ('environments_v2', "1001", "0241520c062095823b442c2cad2f6b6386e6aec2","166a9f1183d86cbf8726ddb09359197dba629209"),
    # Mean Reward : -3541.52
    # Sum Normalized Reward : 43.08898598301832 (primary score)
    # Mean Percentage Complete : 0.678 (secondary score)
    # Mean Normalized Reward : 0.86178
    # -> same result as with 4.2.1 in https://github.com/flatland-association/flatland-baselines/pull/40/changes#diff-ddbd72c00fdf2bf68dc455453bc81a1ad80be9018af263c28b367985a40dc98c

])
@pytest.mark.slow
def test_online_calibrated_against_offline_envs_v2(_containers_fixture):
    """
    Verify online evaluation yields the same result as offline evaluation in legacy way with current code basis on envs v2 (old statefull rail generator).
    """

    root_data_dir, post_seed = _containers_fixture
    print(root_data_dir)
    print(list(root_data_dir.rglob("**/*")))

    df = pd.read_csv(root_data_dir / "results" / "results.csv")

    sum_normalized_reward = df["normalized_reward"].sum()
    mean_normalized_reward = df["normalized_reward"].mean()
    mean_percentage_complete = df["percentage_complete"].mean()
    mean_reward = df['reward'].mean()

    verify_online_offline_calibration_envs_v2(mean_normalized_reward, mean_percentage_complete, mean_reward, sum_normalized_reward)


# https://docs.pytest.org/en/7.1.x/how-to/fixtures.html#override-a-fixture-with-direct-test-parametrization
@pytest.mark.parametrize('environments,seed,baselines_ref,rl_ref', [
    ('environments_v3_trunc', "1001", "", ""),
    ('environments_v3_trunc', "NONE", "", ""),
])
@pytest.mark.slow
def test_online_calibrated_against_offline_envs_v3_trunc(_containers_fixture):
    """
    Verify online evaluation yields the same result as offline evaluation in legacy way with current code basis on first 20 envs of v3 (new stateless rail generator).
    """

    root_data_dir, post_seed = _containers_fixture
    print(root_data_dir)
    print(list(root_data_dir.rglob("**/*")))

    df = pd.read_csv(root_data_dir / "results" / "results.csv")

    sum_normalized_reward = df["normalized_reward"].sum()
    mean_normalized_reward = df["normalized_reward"].mean()
    mean_percentage_complete = df["percentage_complete"].mean()
    mean_reward = df['reward'].mean()

    verify_online_offline_calibration_envs_v3_trunc(mean_normalized_reward, mean_percentage_complete, mean_reward, sum_normalized_reward, post_seed)
