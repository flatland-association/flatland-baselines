import os
import pstats
import tempfile
import uuid
from pathlib import Path
from typing import List

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

FLATLAND_RL_VERSIONS = [
    # {
    #     "sha": "f37c7f7947d823317651521994aaaf464e6e8dfa",
    #     "date": "Sat Nov 19 17:55:28 2022 +0000",
    #     "message": "introduce find_swaps2 - faster version of find_swaps",
    #     "name": "before lru"
    # },
    # {
    #     "sha": "45768358",
    #     "date": "Fri Oct 27 15:19:24 2023 +0200",
    #     "name": "v4.0.0",
    #     "message": "Release version 4.0.0"
    # },
    # {
    #     "sha": "5a97ccb6aec2e7c6227aba8a3b33de54f567ee3a",
    #     "date": "Tue Apr 23 15:17:36 2024 +0200",
    #     "name": "v4.0.2"
    # },
    # {
    #     "sha": "9115580bf7c602ca3c524ad392489bd712f355da",
    #     "date": "Tue Feb 18 17:03:18 2025 +0100",
    #     "name": "v4.0.4"
    # },
    # {
    #     "sha": "01d4c7ae8179c7a716059552eb31865772e5a549",
    #     "date": "Tue Feb 18 17:11:28 2025 +0100",
    #     "name": "118-fix-lru-cache-in-env-loading"
    # },
    # {
    #     "sha": "3f905a2bc37a0cd69047513d43df1576e7ba7634",
    #     "date": "Mon Mar 31 11:22:49 2025 +0200",
    #     "name": "179-simplify-step"
    # },
    # {
    #     "sha": "4fecd60e49dfb144b452f100ce916af2ed2a58fd",
    #     "date": "Mon Mar 31 18:16:23 2025 +0200",
    #     "name": "v4.1.0"
    # },
    # {
    #     "sha": "04911f88f50e30188b7d671291e0c2bbe1ee5ad1",
    #     "date": "Fri May 16 16:57:14 2025 +0200",
    #     "name": "v4.1.1"
    # },
    # {
    #     "sha": "8f607149e29590a5baa5211d0efd32d1858091a3",
    #     "date": "Fri May 16 16:57:14 2025 +0200",
    #     "name": "v4.2.0"
    # },
    # {
    #     "sha": "79c1031f4cb80e671fc6e58e9a590fd78865a4af",
    #     "date": "Fri May 16 16:57:14 2025 +0200",
    #     "name": "v4.2.1"
    # },
    # {
    #     "sha": "d6a69ca00bde635f78b6b45032bdcfe6d2e480aa",
    #     "date": "Fri May 16 16:57:14 2025 +0200",
    #     "name": "v4.2.2"
    # },
    # {
    #     "sha": "189c259a8fe19266329a6826a137e1db29a12996",
    #     "date": "Fri May 16 16:57:14 2025 +0200",
    #     "name": "v4.2.3"
    # },
    # {
    #     "sha": "dfcea58bfed32c9b549fba04a86e57a532280dc2",
    #     "date": "Fri May 16 16:57:14 2025 +0200",
    #     "name": "v4.2.4"
    # },
    # {
    #     "sha": "172f9f4f7e3ee1df5ab52bf38e8e88d32d85af33",
    #     "date": "Fri May 16 16:57:14 2025 +0200",
    #     "name": "pr-400"
    # },
    # {
    #     "sha": "3c22c4c16946afa3625af78b67cf7076c299b014",
    #     "date": "Fri May 16 16:57:14 2025 +0200",
    #     "name": "pr-401"
    # },
    # {
    #     "sha": "2156f46ffd8c707c8737fe00da89376f9cc5c4e4",
    #     "date": "Fri May 16 16:57:14 2025 +0200",
    #     "name": "pr-402"
    # },
    # {
    #     "sha": "6de179e336d6d7fe40a385a90ca9d1ac328fc786",
    #     "date": "Fri May 16 16:57:14 2025 +0200",
    #     "name": "v4.2.5"
    # },
    {
        "sha": "f5943d7e23f706742f335ec260b54d4588b3ab88",
        "date": "Fri May 16 16:57:14 2025 +0200",
        "name": "pr-430-click"
    },
    # {
    #     "sha": "LOCAL",
    #     "date": "--",
    #     "name": "LOCAL"
    # }
]
FLATLAND_BASELINES_VERSIONS = [
    {
        "sha": "9bf2f693456daabb8b99bc6dd0cd246c52590343",
        "date": "Fri May 1 15:35:13 2026 +0200",
        "name": "v4.2.5"
    }
]


# https://stackoverflow.com/questions/44302726/pandas-how-to-store-cprofile-output-in-a-pandas-dataframe
def prof_to_df(st):
    keys_from_k = ['file', 'line', 'fn']
    keys_from_v = ['cc', 'ncalls', 'tottime', 'cumtime', 'callers']
    data = {k: [] for k in keys_from_k + keys_from_v}

    s = st.stats

    for k in s.keys():
        for i, kk in enumerate(keys_from_k):
            data[kk].append(k[i])

        for i, kk in enumerate(keys_from_v):
            data[kk].append(s[k][i])
    return pd.DataFrame(data)


def filter_df(df, conditions):
    cond = False
    for fn, file in conditions:
        cond = cond | (df["fn"] == fn) & (df["file"].str.contains(file))
    return df[cond]


def analyse_df(df, fn, file, agg, sort_by="cumtime"):
    df_ = df[(df["fn"] == fn) & (df["file"].str.contains(file))].groupby("name").agg(agg).sort_values((sort_by, "median"), ascending=True)
    df_["diff_median"] = df_[(sort_by, "median")].diff().cumsum()
    df_["diff%_median"] = df_["diff_median"] / (df_[("cumtime", "median")] + df_["diff_median"]) * 100
    df_["diff_mean"] = df_[(sort_by, "mean")].diff().cumsum()
    df_["diff%_mean"] = df_["diff_mean"] / (df_[("cumtime", "mean")] + df_["diff_mean"]) * 100
    return df_


def aggregate(output_dir: Path, labels: List[str], example: str, NUM_RUNS: int):
    dfs = []
    for label, l_flatland, l_baselines in labels:
        for i in range(NUM_RUNS):
            fn = f'{example}_{label}_{i}.prof'
            ps = pstats.Stats(str(output_dir / fn))
            df = prof_to_df(ps)
            df["name"] = label
            df["sha"] = label
            df["sha_flatland"] = l_flatland["sha"]
            df["sha_baselines"] = l_baselines["sha"]
            df["name_flatland"] = l_flatland["name"]
            df["name_baselines"] = l_baselines["name"]
            dfs.append(df)
    df = pd.concat(dfs)
    return df


def main(env_path, num_runs, agg, output_dir):
    SCENARIOS_VOLUME_MOUNTPATH = os.getenv("SCENARIOS_VOLUME_MOUNTPATH", None)

    scenario_id = env_path.replace("/", "_")

    with tempfile.TemporaryDirectory() as tmpdirname:
        os.system(f"git clone https://github.com/flatland-association/flatland-rl.git {tmpdirname}/flatland-rl")
        os.system(f"cd {tmpdirname}/flatland-rl && git clean -f && git reset --hard")

        os.system(f"git clone https://github.com/flatland-association/flatland-baselines.git {tmpdirname}/flatland-baselines")
        os.system(f"cd {tmpdirname}flatland-baselines && git clean -f && git reset --hard")
        labels = []
        example = "flatland_performance_profiling.py"
        for l_flatland in FLATLAND_RL_VERSIONS:
            os.system(f"cd {tmpdirname}/flatland-rl && git checkout {l_flatland["sha"]} && git log -1")
            for l_baselines in FLATLAND_BASELINES_VERSIONS:
                os.system(f"cd {tmpdirname}/flatland-baselines && git checkout {l_baselines["sha"]} && git log -1")
                label = f"{l_flatland["name"]}_{l_baselines["name"]}"
                labels.append((label, l_flatland, l_baselines))
                for i in range(num_runs):
                    data_dir = f"/tmp/{uuid.uuid4()}"
                    Path(data_dir).mkdir()
                    args = ["--data-dir", data_dir,
                            "--ep-id", scenario_id,
                            "--env-path", f"{SCENARIOS_VOLUME_MOUNTPATH}/{env_path}",
                            "--policy", "flatland_baselines.deadlock_avoidance_heuristic.policy.deadlock_avoidance_policy.DeadlockAvoidanceHeuristics",
                            "--obs-builder", "flatland_baselines.deadlock_avoidance_heuristic.observation.full_env_observation.FullEnvObservation",
                            "--snapshot-interval", "0", ]
                    python_path = []
                    if l_flatland["name"] != 'LOCAL':
                        python_path.append(f"{tmpdirname}/flatland-rl")
                    if l_baselines["name"] != 'LOCAL':
                        python_path.append(f"{tmpdirname}/flatland-baselines")
                    python_path = ':'.join(python_path)
                    if python_path != '':
                        python_path = f"PYTHONPATH={python_path}"
                    os.system(
                        f"cd {tmpdirname}/flatland-rl && {python_path} python -m cProfile -o {tmpdirname}/{example}_{label}_{i}.prof -m flatland.trajectories.policy_runner {" ".join(args)}")
        df_flatland_performance_profiling = aggregate(Path(tmpdirname), labels, example, num_runs)
        print(df_flatland_performance_profiling)

        plt.figure(figsize=(15, 8))
        ax = sns.barplot(filter_df(df_flatland_performance_profiling, [
            ("step", "rail_env.py"),
            ("get_k_shortest_paths", "rail_env_shortest_paths.py"),
            ("create_from_policy", "policy_runner.py")
        ]), x="name", y="cumtime", hue="fn", legend=True, estimator="median")
        ax.bar_label(ax.containers[0], fontsize=10);
        ax.bar_label(ax.containers[1], fontsize=10);
        ax.bar_label(ax.containers[2], fontsize=10);
        plt.savefig(output_dir / "performance_overall.png")

        analyse_df(df_flatland_performance_profiling, "run_simulation", example, agg)

        plt.figure(figsize=(15, 8))
        ax = sns.barplot(filter_df(df_flatland_performance_profiling, [
            ("step", "rail_env.py"),
            ("a_star", "star"),
            ("addAgent", "agent_chains.py"),
            ("find_conflicts", "agent_chains.py"),
            ("check_motion", "agent_chains.py"),
        ]), x="name", y="cumtime", hue="fn", legend=True, estimator="mean")
        plt.savefig(output_dir / "performance_a_star_motion_check.png")
        plt.figure(figsize=(15, 8))
        ax = sns.barplot(filter_df(df_flatland_performance_profiling, [
            ("is_dead_end", "map"),
            ("get_transition", "map"),
        ]), x="name", y="cumtime", hue="fn", legend=True, estimator="median")
        ax.bar_label(ax.containers[0], fontsize=10);
        plt.savefig(output_dir / "performance_lru.png")


if __name__ == '__main__':
    level = 0
    scenario = 1
    main(
        env_path=f"level_{level}/level_{level}_scenario_{scenario}.pkl",
        num_runs=2,
        agg={"fn": ["first"], "sha": ["first"], "cumtime": ['mean', 'median', 'min', 'max', 'std'], "tottime": ['mean', 'median', 'min', 'max', 'std']},
        output_dir=Path(".")
    )
