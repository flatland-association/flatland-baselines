from pathlib import Path

import pandas as pd


def print_stats(df, col):
    print(f"{col}: {df[col].mean():.2f}+-{df[col].std():.2f}, [{df[col].min():.2f},{df[col].max():.2f}] from {len(df)}")


if __name__ == '__main__':
    paths = [p for p in Path("../../../../flatland-scenarios/scenario_generator/results/").rglob("**/TrainMovementEvents.trains_arrived.tsv")]
    print(paths)
    df = pd.concat([pd.read_csv(p, sep="\t") for p in paths])
    print(df)

    print_stats(df, "success_rate")
    print_stats(df, "normalized_reward")
