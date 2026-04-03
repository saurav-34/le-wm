"""
Collect a dataset from any Gymnasium / MiniGrid environment and save it in
the HDF5 format expected by LeWM's HDF5Dataset.

HDF5 layout:
  pixels      (N, H, W, C)  uint8   -- rendered frames, every step
  action      (N, act_dim)  float32 -- actions, every step
  observation (N, obs_dim)  float32 -- flattened env obs, every step
  ep_len      (E,)          int64   -- length of each episode
  ep_offset   (E,)          int64   -- start index of each episode

Policy modes:
  random   -- uniform random actions (default)
  train    -- train a PPO agent with SB3, then collect with it
  load     -- load a saved SB3 policy from --policy-path and collect

Usage examples:
  # MiniGrid with trained PPO (recommended)
  python scripts/collect_dataset.py \\
      --env MiniGrid-FourRooms-v0 --policy train --train-steps 500000

  # MiniGrid random (quick sanity check)
  python scripts/collect_dataset.py \\
      --env MiniGrid-FourRooms-v0 --policy random --episodes 500

  # Load a previously saved policy
  python scripts/collect_dataset.py \\
      --env MiniGrid-FourRooms-v0 --policy load --policy-path ppo_fourrooms.zip

  # LunarLander with trained PPO
  python scripts/collect_dataset.py \\
      --env LunarLanderContinuous-v3 --policy train --train-steps 300000
"""

import argparse
import os
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
from tqdm import tqdm

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ---------------------------------------------------------------------------
# MiniGrid helpers
# ---------------------------------------------------------------------------

def is_minigrid(env_id: str) -> bool:
    return "MiniGrid" in env_id or "BabyAI" in env_id


# ---------------------------------------------------------------------------
# Resize helper
# ---------------------------------------------------------------------------

def resize_frame(frame: np.ndarray, img_size: int) -> np.ndarray:
    if frame.shape[0] == img_size and frame.shape[1] == img_size:
        return frame
    if HAS_CV2:
        return cv2.resize(frame, (img_size, img_size), interpolation=cv2.INTER_AREA)
    # Nearest-neighbour fallback
    h, w = frame.shape[:2]
    ys = (np.arange(img_size) * h / img_size).astype(int)
    xs = (np.arange(img_size) * w / img_size).astype(int)
    return frame[np.ix_(ys, xs)]


# ---------------------------------------------------------------------------
# Policy builders
# ---------------------------------------------------------------------------

def build_random_policy(env):
    def policy(obs):
        return env.action_space.sample()
    return policy


def build_trained_policy(env_id: str, train_steps: int, seed: int, policy_save_path: str | None, is_mg: bool):
    """Train a PPO agent with SB3 and return a policy callable."""
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
    except ImportError:
        raise ImportError("stable-baselines3 not found. Run: pip install stable-baselines3")

    print(f"\nTraining PPO for {train_steps:,} steps on {env_id} ...")

    if is_mg:
        import minigrid  # noqa: F401 — registers MiniGrid envs with gymnasium
        from minigrid.wrappers import FlatObsWrapper

        def mg_factory():
            e = gym.make(env_id)
            e = FlatObsWrapper(e)
            return e

        vec_env = make_vec_env(mg_factory, n_envs=8, seed=seed)
    else:
        vec_env = make_vec_env(env_id, n_envs=8, seed=seed)

    model = PPO("MlpPolicy", vec_env, verbose=1, seed=seed)
    model.learn(total_timesteps=train_steps, progress_bar=True)

    if policy_save_path:
        model.save(policy_save_path)
        print(f"Policy saved to {policy_save_path}")

    def policy(obs):
        obs_arr = np.array(obs, dtype=np.float32).flatten()[None]  # (1, obs_dim)
        action, _ = model.predict(obs_arr, deterministic=True)
        return action[0]

    return policy


def build_loaded_policy(policy_path: str):
    """Load a saved SB3 PPO policy."""
    try:
        from stable_baselines3 import PPO
    except ImportError:
        raise ImportError("stable-baselines3 not found. Run: pip install stable-baselines3")

    model = PPO.load(policy_path)
    print(f"Loaded policy from {policy_path}")

    def policy(obs):
        obs_arr = np.array(obs, dtype=np.float32).flatten()[None]
        action, _ = model.predict(obs_arr, deterministic=True)
        return action[0]

    return policy


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def collect_episodes(
    env_id: str,
    num_episodes: int,
    img_size: int,
    max_steps: int,
    policy,
    seed: int,
    is_mg: bool,
):
    if is_mg:
        import minigrid  # noqa: F401 — registers MiniGrid envs with gymnasium
        from minigrid.wrappers import FlatObsWrapper
        env = gym.make(env_id, render_mode="rgb_array")
        flat_env = FlatObsWrapper(env)   # for flattened obs fed to policy
    else:
        env = gym.make(env_id, render_mode="rgb_array")
        flat_env = env

    all_pixels, all_actions, all_obs = [], [], []
    ep_lengths = []

    for ep in tqdm(range(num_episodes), desc="Collecting episodes"):
        flat_obs, _ = flat_env.reset(seed=seed + ep)
        ep_pixels, ep_actions, ep_obs = [], [], []

        for _ in range(max_steps):
            frame = env.render()  # (H, W, C) uint8
            frame = resize_frame(frame, img_size)

            action = policy(flat_obs)
            flat_obs, _, terminated, truncated, _ = flat_env.step(action)

            act = np.array(action, dtype=np.float32).flatten()
            obs_flat = np.array(flat_obs, dtype=np.float32).flatten()

            ep_pixels.append(frame.astype(np.uint8))
            ep_actions.append(act)
            ep_obs.append(obs_flat)

            if terminated or truncated:
                break

        ep_lengths.append(len(ep_pixels))
        all_pixels.extend(ep_pixels)
        all_actions.extend(ep_actions)
        all_obs.extend(ep_obs)

    env.close()
    if flat_env is not env:
        flat_env.close()

    return (
        np.stack(all_pixels),
        np.stack(all_actions),
        np.stack(all_obs),
        np.array(ep_lengths, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_hdf5(out_path: Path, pixels, actions, observations, ep_lengths):
    ep_offsets = np.concatenate([[0], np.cumsum(ep_lengths)[:-1]]).astype(np.int64)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("pixels",      data=pixels,       compression="lzf", chunks=(1, *pixels.shape[1:]))
        f.create_dataset("action",      data=actions,      compression="lzf")
        f.create_dataset("observation", data=observations, compression="lzf")
        f.create_dataset("ep_len",      data=ep_lengths)
        f.create_dataset("ep_offset",   data=ep_offsets)

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\nSaved {len(ep_lengths)} episodes  |  {pixels.shape[0]} total steps  |  {out_path}  [{size_mb:.1f} MB]")
    print(f"  pixels:      {pixels.shape}  {pixels.dtype}")
    print(f"  action:      {actions.shape}  {actions.dtype}")
    print(f"  observation: {observations.shape}  {observations.dtype}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Collect LeWM-compatible HDF5 dataset")
    parser.add_argument("--env",         default="MiniGrid-FourRooms-v0", help="Gymnasium env ID")
    parser.add_argument("--policy",      default="train",  choices=["random", "train", "load"],
                        help="Policy: random | train (PPO via SB3) | load (from --policy-path)")
    parser.add_argument("--policy-path", default=None,
                        help="Path to save/load SB3 policy (.zip). For --policy train: saves here. For --policy load: loads from here.")
    parser.add_argument("--train-steps", type=int, default=500_000,  help="PPO training steps (only for --policy train)")
    parser.add_argument("--episodes",   type=int, default=300,       help="Episodes to collect")
    parser.add_argument("--max-steps",  type=int, default=500,       help="Max steps per episode")
    parser.add_argument("--img-size",   type=int, default=224,       help="Output image size (square) — keep at 224 to match LeWM architecture")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--out-name",   default=None,                help="HDF5 name (no extension). Defaults to env slug.")
    parser.add_argument("--out-dir",    default=None,                help="Output dir. Defaults to $STABLEWM_HOME or ~/.stable-wm/")
    args = parser.parse_args()

    is_mg = is_minigrid(args.env)
    if is_mg:
        try:
            import minigrid  # noqa: F401 — registers MiniGrid envs with gymnasium
        except ImportError:
            raise ImportError("MiniGrid not found. Run: pip install minigrid")

    out_dir  = Path(args.out_dir) if args.out_dir else Path(os.environ.get("STABLEWM_HOME", Path.home() / ".stable-wm"))
    out_name = args.out_name or args.env.lower().replace("-", "_").replace("/", "_")
    out_path = out_dir / f"{out_name}.h5"

    print(f"Environment  : {args.env}  {'[MiniGrid]' if is_mg else ''}")
    print(f"Policy       : {args.policy}")
    print(f"Episodes     : {args.episodes}")
    print(f"Max steps    : {args.max_steps}")
    print(f"Image size   : {args.img_size}x{args.img_size}")
    print(f"Output       : {out_path}")

    # Build policy
    if args.policy == "random":
        # build_random_policy needs an env instance — pass a dummy wrapper
        dummy_env = gym.make(args.env)
        policy = build_random_policy(dummy_env)
        dummy_env.close()
    elif args.policy == "train":
        policy = build_trained_policy(
            env_id=args.env,
            train_steps=args.train_steps,
            seed=args.seed,
            policy_save_path=args.policy_path,
            is_mg=is_mg,
        )
    else:  # load
        if not args.policy_path:
            raise ValueError("--policy load requires --policy-path")
        policy = build_loaded_policy(args.policy_path)

    # Collect
    pixels, actions, observations, ep_lengths = collect_episodes(
        env_id=args.env,
        num_episodes=args.episodes,
        img_size=args.img_size,
        max_steps=args.max_steps,
        policy=policy,
        seed=args.seed,
        is_mg=is_mg,
    )

    save_hdf5(out_path, pixels, actions, observations, ep_lengths)

    print(f"""
To train LeWM on this dataset:

  1. Create config/train/data/{out_name}.yaml  (or use --out-name to match an existing config)
  2. Run: python train.py data={out_name}

Example config/train/data/{out_name}.yaml:
  dataset:
    num_steps: ${{eval:'${{wm.num_preds}} + ${{wm.history_size}}'}}
    frameskip: 5
    name: {out_name}
    keys_to_load:
      - pixels
      - action
      - observation
    keys_to_cache:
      - action
      - observation
""")


if __name__ == "__main__":
    main()
