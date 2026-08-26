import os
import pstats
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional

import click
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame

FLATLAND_RL_VERSIONS = [
    {
        "sha": "1e90404bd50d66a7e9ab74131a1cd73e823e42c2",
        "date": "Fri May 16 16:57:14 2025 +0200",
        "name": "pr-430-click"
    },
    {
        "sha": "ac36ea87d7fbce3d6fa22e603c58b67d1bfeac8d",
        "date": "Fri May 16 16:57:14 2025 +0200",
        "name": "pr-427-cache-fraction-comparisons"
    },
    {
        "sha": "3fb0fdc013079d626d182edec6a6700556f5aaab",
        "date": "Fri May 16 16:57:14 2025 +0200",
        "name": "pr-428-cache-fraction-speed-counter-rewards"
    },
    {
        "sha": "203d90ad8ad7616785370918bf9088a5348a7eb6",
        "date": "Fri May 16 16:57:14 2025 +0200",
        "name": "pr-426-waypoint-use-cache-hash"
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
    },
    # {
    #     "sha": "LOCAL",
    #     "date": "--",
    #     "name": "LOCAL"
    # }
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


def aggregate(output_dir: Path, labels: List[str], example: str, num_runs: int):
    dfs = []
    for label, l_flatland, l_baselines in labels:
        for i in range(num_runs):
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


def _plot_figures(df_flatland_performance_profiling: DataFrame, output_dir):
    plt.figure(figsize=(15, 8))
    ax = sns.barplot(filter_df(df_flatland_performance_profiling, [
        ("step", "rail_env.py"),
        ("get_k_shortest_paths", "rail_env_shortest_paths.py"),
        ("create_from_policy", "policy_runner.py")
    ]), x="name", y="cumtime", hue="fn", legend=True, estimator="median")
    ax.bar_label(ax.containers[0], fontsize=10)
    ax.bar_label(ax.containers[1], fontsize=10)
    ax.bar_label(ax.containers[2], fontsize=10)
    plt.savefig(output_dir / "performance_overall.png")

    plt.figure(figsize=(15, 8))
    sns.barplot(filter_df(df_flatland_performance_profiling, [
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
    ax.bar_label(ax.containers[0], fontsize=10)
    plt.savefig(output_dir / "performance_lru.png")


_DEFAULT_AGG = {
    "fn": ["first"],
    "sha": ["first"],
    "cumtime": ["mean", "median", "min", "max", "std"],
    "tottime": ["mean", "median", "min", "max", "std"],
}


@click.command()
@click.option("--env-path", default="level_0_scenario_1.pkl", help="Relative path to the scenario pkl file, e.g. level_0/level_0_scenario_1.pkl", )
@click.option("--num-runs", default=5, show_default=True, type=int, help="Number of profiling runs per version combination")
@click.option("--output-dir", default=".", show_default=True, type=click.Path(path_type=Path), help="Directory to write plot PNGs into")
@click.option("--scenarios-dir", required=False, type=click.Path(path_type=Path),
              help="Base dir for scenarios. If not set, env var SCENARIOS_DIR needs to be set.")
def performance_dla(env_path: str, num_runs: int, output_dir: Path, agg: Optional[dict] = None, scenarios_dir: Path = None):
    if agg is None:
        agg = _DEFAULT_AGG
    if scenarios_dir is None:
        scenarios_dir = os.getenv("SCENARIOS_DIR")
    if scenarios_dir is None:
        raise ValueError("Environment variable SCENARIOS_DIR must be set or --scenarios-dir <DIR> must be passed")

    with tempfile.TemporaryDirectory() as tmpdirname:
        subprocess.run(f"git clone https://github.com/flatland-association/flatland-rl.git {tmpdirname}/flatland-rl", check=True, shell=True)
        subprocess.run(f"cd {tmpdirname}/flatland-rl && git clean -f && git reset --hard", check=True, shell=True)

        subprocess.run(f"git clone https://github.com/flatland-association/flatland-baselines.git {tmpdirname}/flatland-baselines", check=True, shell=True)
        subprocess.run(f"cd {tmpdirname}/flatland-baselines && git clean -f && git reset --hard", check=True, shell=True)
        labels = []
        example = "performance_dla.py"
        for l_flatland in FLATLAND_RL_VERSIONS:
            subprocess.run(f"cd {tmpdirname}/flatland-rl && git checkout {l_flatland['sha']} && git log -1", check=True, shell=True)
            for l_baselines in FLATLAND_BASELINES_VERSIONS:
                subprocess.run(f"cd {tmpdirname}/flatland-baselines && git checkout {l_baselines['sha']} && git log -1", check=True, shell=True)
                label = f"{l_flatland['name']}_{l_baselines['name']}"
                labels.append((label, l_flatland, l_baselines))
                for i in range(num_runs):
                    scenario_id = str(uuid.uuid4())
                    data_dir = f"{tmpdirname}/{scenario_id}"
                    Path(data_dir).mkdir()
                    args = ["--data-dir", data_dir,
                            "--ep-id", scenario_id,
                            "--env-path", f"{scenarios_dir}/{env_path}",
                            "--policy", "flatland_baselines.deadlock_avoidance_heuristic.policy.deadlock_avoidance_policy.DeadlockAvoidanceHeuristics",
                            "--obs-builder", "flatland_baselines.deadlock_avoidance_heuristic.observation.full_env_observation.FullEnvObservation",
                            "--snapshot-interval", "0"]
                    pythonpath_dirs = []
                    if l_flatland["name"] != 'LOCAL':
                        pythonpath_dirs.append(f"{tmpdirname}/flatland-rl")
                    if l_baselines["name"] != 'LOCAL':
                        pythonpath_dirs.append(f"{tmpdirname}/flatland-baselines")
                    env = {**os.environ}
                    if pythonpath_dirs:
                        env["PYTHONPATH"] = ':'.join(pythonpath_dirs)
                    subprocess.run(
                        ["python", "-m", "cProfile",
                         "-o", f"{tmpdirname}/{example}_{label}_{i}.prof",
                         "-m", "flatland.trajectories.policy_runner"] + args,
                        check=True,
                        cwd=f"{tmpdirname}/flatland-rl",
                        env=env,
                    )
        df_flatland_performance_profiling = aggregate(Path(tmpdirname), labels, example, num_runs)

        print(df_flatland_performance_profiling)
        print(analyse_df(df_flatland_performance_profiling, "create_from_policy", "policy_runner.py", agg))
        _plot_figures(df_flatland_performance_profiling, output_dir)


if __name__ == '__main__':
    performance_dla()
