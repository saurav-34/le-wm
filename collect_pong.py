"""
Collect CALE-Pong episodes and save to HDF5 format for le-wm training.

HDF5 layout expected by HDF5Dataset:
  ep_len    : (num_episodes,)     int64  — length of each episode in raw steps
  ep_offset : (num_episodes,)     int64  — cumulative start offset of each episode
  pixels    : (total_steps, 84, 84, 3) uint8  — resized from 210×160 at collection time
  action    : (total_steps, 1)    float32  — stored at every raw step
  observation:(total_steps, 128)  float32  — ALE RAM state

The dataset uses `frameskip` at READ time (not write time), so we store
every raw environment step here and let HDF5Dataset handle subsampling.

Episodes are written to HDF5 incrementally — constant RAM usage regardless
of dataset size.
"""

import argparse
import logging
from pathlib import Path

import ale_py
import gymnasium as gym
import h5py
import hdf5plugin  # noqa: F401 — registers Blosc compressor
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

IMG_SIZE = (84, 84)
OBS_DIM = 128


def collect_episodes(
    num_episodes: int,
    max_steps_per_episode: int,
    seed: int,
    output_path: Path,
):
    gym.register_envs(ale_py)
    env = gym.make(
        "ALE/Pong-v5",
        render_mode="rgb_array",
        obs_type="ram",
        frameskip=1,
        repeat_action_probability=0.0,
        full_action_space=False,
    )

    rng = np.random.default_rng(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    blosc_opts = dict(
        chunks=(256, *IMG_SIZE, 3),
        **hdf5plugin.Blosc(cname="lz4", clevel=5, shuffle=hdf5plugin.Blosc.BITSHUFFLE),
    )

    with h5py.File(output_path, "w", swmr=False) as f:
        px_ds = f.create_dataset(
            "pixels", shape=(0, *IMG_SIZE, 3), maxshape=(None, *IMG_SIZE, 3),
            dtype=np.uint8, **blosc_opts,
        )
        act_ds = f.create_dataset(
            "action", shape=(0, 1), maxshape=(None, 1),
            dtype=np.float32, chunks=(4096, 1),
        )
        obs_ds = f.create_dataset(
            "observation", shape=(0, OBS_DIM), maxshape=(None, OBS_DIM),
            dtype=np.float32, chunks=(4096, OBS_DIM),
        )
        ep_len_ds = f.create_dataset(
            "ep_len", shape=(0,), maxshape=(None,), dtype=np.int64,
        )
        ep_off_ds = f.create_dataset(
            "ep_offset", shape=(0,), maxshape=(None,), dtype=np.int64,
        )

        total_steps = 0
        for ep in range(num_episodes):
            obs_ram, _ = env.reset(seed=int(rng.integers(0, 2**31)))
            ep_pixels, ep_actions, ep_obs = [], [], []

            for _ in range(max_steps_per_episode):
                frame = np.array(Image.fromarray(env.render()).resize(IMG_SIZE))
                action = int(rng.integers(0, env.action_space.n))
                obs_ram, _, terminated, truncated, _ = env.step(action)

                ep_pixels.append(frame)
                ep_actions.append([action])
                ep_obs.append(obs_ram.astype(np.float32))

                if terminated or truncated:
                    break

            n = len(ep_pixels)

            # append this episode to each dataset
            px_ds.resize(total_steps + n, axis=0)
            act_ds.resize(total_steps + n, axis=0)
            obs_ds.resize(total_steps + n, axis=0)

            px_ds[total_steps: total_steps + n] = np.stack(ep_pixels)
            act_ds[total_steps: total_steps + n] = np.array(ep_actions, dtype=np.float32)
            obs_ds[total_steps: total_steps + n] = np.stack(ep_obs)

            ep_len_ds.resize(ep + 1, axis=0)
            ep_off_ds.resize(ep + 1, axis=0)
            ep_len_ds[ep] = n
            ep_off_ds[ep] = total_steps

            total_steps += n
            log.info(f"Episode {ep + 1}/{num_episodes}: {n} steps (total {total_steps})")

    env.close()
    log.info(f"Done — {num_episodes} episodes, {total_steps} steps → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Collect CALE-Pong data for le-wm")
    parser.add_argument("--num-episodes", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=27000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/saurav/.stable_worldmodel/pong_random.h5"),
    )
    args = parser.parse_args()

    collect_episodes(
        num_episodes=args.num_episodes,
        max_steps_per_episode=args.max_steps,
        seed=args.seed,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
