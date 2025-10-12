"""
Plot AWS DeepRacer training performance from deepracer-for-cloud logs.

This script:
 - Loads all training-simtrace CSVs from your local deepracer-for-cloud folder
 - Combines them into a single DataFrame
 - Plots key performance indicators:
     1. Average reward per training iteration
     2. Car path (x vs y positions)
     3. Reward per step for a specific iteration
     4. Steering vs speed scatter plot
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# USER CONFIGURATION
# Change this to match system path
BASE_DIR = "deepracer-for-cloud/data/minio/bucket/rl-deepracer-sagemaker/training-simtrace"

# LOAD AND MERGE FILES
print(f"Searching for training-simtrace files in:\n{BASE_DIR}\n")

csv_files = glob.glob(os.path.join(BASE_DIR, "*-iteration.csv", "*"))

if not csv_files:
    raise FileNotFoundError("No training-simtrace CSV files found. Check your BASE_DIR path.")

dfs = []
for file in sorted(csv_files):
    iteration_str = os.path.basename(os.path.dirname(file))
    try:
        iteration = int(iteration_str.split('-')[0])
    except ValueError:
        iteration = None
    temp_df = pd.read_csv(file)
    temp_df["iteration"] = iteration
    dfs.append(temp_df)

df_all = pd.concat(dfs, ignore_index=True)
print(f"Loaded {len(csv_files)} files with {len(df_all):,} total rows.\n")

# CLEANUP AND BASIC CHECK
expected_columns = ["episode", "steps", "x", "y", "heading", "steering_angle",
                    "speed", "reward", "progress"]
for col in expected_columns:
    if col not in df_all.columns:
        print(f"Warning: column '{col}' not found in CSVs")

# Drop rows with missing values just in case
df_all.dropna(subset=["reward"], inplace=True)

# 1. AVERAGE REWARD PER ITERATION
reward_by_iter = df_all.groupby("iteration")["reward"].mean()

plt.figure(figsize=(10,5))
plt.plot(reward_by_iter.index, reward_by_iter.values, marker="o", color="tab:blue")
plt.xlabel("Training Iteration")
plt.ylabel("Average Reward per Step")
plt.title("DeepRacer – Average Reward Progression per Iteration")
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. CAR PATH (XY TRAJECTORY)
# Last iteration for best model visualization
latest_iter = df_all["iteration"].max()
df_latest = df_all[df_all["iteration"] == latest_iter]

plt.figure(figsize=(8,8))
sc = plt.scatter(df_latest["x"], df_latest["y"],
                 c=df_latest["progress"], cmap="viridis", s=10)
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.title(f"Car Path During Training – Iteration {latest_iter}")
plt.colorbar(sc, label="Progress (%)")
plt.grid(True)
plt.axis("equal")
plt.tight_layout()
plt.show()

# 3. REWARD PER STEP FOR LATEST ITERATION
plt.figure(figsize=(10,5))
plt.plot(df_latest["steps"], df_latest["reward"], color="tab:orange")
plt.xlabel("Step")
plt.ylabel("Reward")
plt.title(f"Reward per Step – Iteration {latest_iter}")
plt.grid(True)
plt.tight_layout()
plt.show()

# 4. STEERING VS SPEED SCATTER
plt.figure(figsize=(10,5))
sc2 = plt.scatter(df_latest["steering_angle"], df_latest["speed"],
                  c=df_latest["progress"], cmap="plasma", s=8)
plt.xlabel("Steering Angle (deg)")
plt.ylabel("Speed (m/s)")
plt.title(f"Control Behavior (Steering vs Speed) – Iteration {latest_iter}")
plt.colorbar(sc2, label="Progress (%)")
plt.grid(True)
plt.tight_layout()
plt.show()

print("\n Plot generation complete.")
