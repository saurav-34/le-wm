"""
Evaluate a trained LeWM checkpoint on the collected dataset.

What this script measures:
  1. Prediction accuracy  -- MSE between predicted and true embeddings on val set
  2. Rollout accuracy     -- how error grows as we predict further into the future
  3. Nearest-neighbor viz -- saves side-by-side GIFs: ground truth vs model prediction

Usage:
  python scripts/evaluate.py \
      --checkpoint ~/.stable_worldmodel/lewm_epoch_81_object.ckpt \
      --dataset minigrid_fourrooms_v0 \
      --out-dir eval_results/
"""

import argparse
import os
import sys
from pathlib import Path

# make le-wm root importable (needed for torch.load of model object)
sys.path.insert(0, str(Path(__file__).parent.parent))

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


def load_model(ckpt_path: str, device: torch.device):
    model = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model


def load_dataset(dataset_name: str, img_size: int = 224):
    import stable_worldmodel as swm
    import stable_pretraining as spt
    from stable_pretraining import data as dt

    dataset = swm.data.HDF5Dataset(
        name=dataset_name,
        frameskip=5,
        num_steps=4,   # history_size(3) + num_preds(1)
        keys_to_load=["pixels", "action", "observation"],
        keys_to_cache=["action", "observation"],
    )

    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source="pixels", target="pixels")
    resize   = dt.transforms.Resize(img_size, source="pixels", target="pixels")
    transform = dt.transforms.Compose(to_image, resize)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(42)
    _, val_set = spt.data.random_split(dataset, lengths=[0.9, 0.1], generator=rnd_gen)
    return dataset, val_set


def denorm(img_chw: np.ndarray) -> np.ndarray:
    """Reverse ImageNet normalisation → uint8 HWC."""
    img = img_chw.transpose(1, 2, 0)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return (img.clip(0, 1) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 1. Prediction accuracy
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_prediction_accuracy(model, val_loader, device, ctx_len=3):
    pred_errors, step_errors = [], []

    for batch in tqdm(val_loader, desc="Prediction accuracy"):
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        batch["action"] = torch.nan_to_num(batch["action"], 0.0)

        out     = model.encode(batch)
        emb     = out["emb"]          # (B, T, D)
        act_emb = out["act_emb"]

        ctx_emb = emb[:, :ctx_len]
        ctx_act = act_emb[:, :ctx_len]
        tgt_emb = emb[:, 1:]          # (B, T-1, D)
        pred_emb = model.predict(ctx_emb, ctx_act)

        err = (pred_emb - tgt_emb).pow(2).mean(dim=-1)  # (B, T-1)
        pred_errors.append(err.mean().item())
        step_errors.append(err.mean(0).cpu().numpy())   # (T-1,)

    mean_pred_mse  = float(np.mean(pred_errors))
    per_step_mse   = np.stack(step_errors).mean(0)
    return mean_pred_mse, per_step_mse


# ---------------------------------------------------------------------------
# 2. Rollout accuracy (multi-step)
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_rollout(model, val_loader, device, ctx_len=3, rollout_steps=10):
    """How does prediction error grow with rollout horizon?"""
    all_step_errors = []

    for batch in tqdm(val_loader, desc="Rollout accuracy"):
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        batch["action"] = torch.nan_to_num(batch["action"], 0.0)

        out     = model.encode(batch)
        emb     = out["emb"]      # (B, T, D)
        act_emb = out["act_emb"]

        B = emb.size(0)
        cur_emb = emb[:, :ctx_len].clone()
        cur_act = act_emb[:, :ctx_len].clone()

        step_errors = []
        for t in range(min(rollout_steps, emb.size(1) - ctx_len)):
            next_emb = model.predict(cur_emb, cur_act)[:, -1:]   # (B,1,D)
            tgt      = emb[:, ctx_len + t]                        # (B,D)
            err      = (next_emb.squeeze(1) - tgt).pow(2).mean(dim=-1)  # (B,)
            step_errors.append(err.mean().item())

            # shift context
            cur_emb = torch.cat([cur_emb[:, 1:], next_emb], dim=1)
            if ctx_len + t + 1 < act_emb.size(1):
                cur_act = torch.cat([cur_act[:, 1:],
                                     act_emb[:, ctx_len + t:ctx_len + t + 1]], dim=1)

        all_step_errors.append(step_errors)
        break  # one batch is enough for rollout viz

    return np.array(all_step_errors[0]) if all_step_errors else np.array([])


# ---------------------------------------------------------------------------
# 3. Nearest-neighbor visualisation
# ---------------------------------------------------------------------------

@torch.no_grad()
def make_nn_gif(model, val_loader, full_dataset, device,
                ctx_len=3, n_seqs=4, fps=4, out_dir: Path = Path(".")):
    """Save side-by-side GIFs: ground truth | predicted (nearest neighbour)."""
    try:
        from PIL import Image
    except ImportError:
        print("Pillow not found — skipping GIF generation. Run: pip install Pillow")
        return

    # build reference bank from full dataset
    print("Building reference embedding bank...")
    ref_loader = DataLoader(full_dataset, batch_size=64, shuffle=False,
                            num_workers=2, drop_last=False)
    ref_embs, ref_pixels = [], []

    for i, batch in enumerate(tqdm(ref_loader, desc="Ref bank")):
        if i > 15:  # cap at ~1000 frames
            break
        px = batch["pixels"].to(device)       # (B, T, C, H, W)
        B, T = px.shape[:2]
        px_flat = px.view(B * T, *px.shape[2:]).float()
        out  = model.encoder(px_flat, interpolate_pos_encoding=True)
        emb  = model.projector(out.last_hidden_state[:, 0])
        ref_embs.append(emb.cpu())

        px_np = px_flat.permute(0, 2, 3, 1).cpu().numpy()
        px_np = (px_np * IMAGENET_STD + IMAGENET_MEAN).clip(0, 1)
        ref_pixels.append((px_np * 255).astype(np.uint8))

    ref_embs   = torch.cat(ref_embs, dim=0)       # (N, D)
    ref_pixels = np.concatenate(ref_pixels, axis=0) # (N, H, W, C)

    ref_norm = F.normalize(ref_embs, dim=-1).to(device)

    collected = 0
    for batch in val_loader:
        if collected >= n_seqs:
            break

        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        batch["action"] = torch.nan_to_num(batch["action"], 0.0)

        out     = model.encode(batch)
        emb     = out["emb"]
        act_emb = out["act_emb"]
        px      = batch["pixels"]   # (B, T, C, H, W)

        B = emb.size(0)
        for b in range(min(B, n_seqs - collected)):
            ctx_emb = emb[b:b+1, :ctx_len]
            cur_emb = ctx_emb.clone()
            cur_act = act_emb[b:b+1, :ctx_len]

            n_future = emb.size(1) - ctx_len
            pred_frames, gt_frames = [], []

            for t in range(n_future):
                next_emb = model.predict(cur_emb, cur_act)[:, -1:]
                p_norm   = F.normalize(next_emb.squeeze(1), dim=-1)  # (1, D)
                sim      = (p_norm @ ref_norm.T).squeeze(0)           # (N,)
                nn_idx   = sim.argmax().item()
                pred_frames.append(ref_pixels[nn_idx])               # (H, W, C)

                gt_px = px[b, ctx_len + t].permute(1, 2, 0).cpu().numpy()
                gt_px = (gt_px * IMAGENET_STD + IMAGENET_MEAN).clip(0, 1)
                gt_frames.append((gt_px * 255).astype(np.uint8))

                cur_emb = torch.cat([cur_emb[:, 1:], next_emb], dim=1)
                if ctx_len + t + 1 < act_emb.size(1):
                    cur_act = torch.cat([cur_act[:, 1:],
                                         act_emb[b:b+1, ctx_len + t:ctx_len + t + 1]], dim=1)

            if not pred_frames:
                continue

            h, w = gt_frames[0].shape[:2]
            divider = np.ones((h, 4, 3), dtype=np.uint8) * 180
            frames_pil = []
            for gf, pf in zip(gt_frames, pred_frames):
                row = np.concatenate([gf, divider, pf], axis=1)
                frames_pil.append(Image.fromarray(row))

            out_path = out_dir / f"seq_{collected}.gif"
            frames_pil[0].save(
                out_path, save_all=True, append_images=frames_pil[1:],
                loop=0, duration=int(1000 / fps)
            )
            print(f"Saved {out_path}")
            collected += 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to _object.ckpt")
    parser.add_argument("--dataset",    required=True, help="Dataset name (no .h5)")
    parser.add_argument("--img-size",   type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out-dir",    default="eval_results")
    parser.add_argument("--n-seqs",     type=int, default=4, help="GIFs to generate")
    args = parser.parse_args()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device     : {device}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Dataset    : {args.dataset}\n")

    model = load_model(args.checkpoint, device)
    full_dataset, val_set = load_dataset(args.dataset, args.img_size)

    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=2, drop_last=False)

    # 1. Prediction accuracy
    mean_mse, per_step = eval_prediction_accuracy(model, val_loader, device)
    print(f"\n=== Prediction Accuracy ===")
    print(f"Mean val pred MSE : {mean_mse:.6f}")
    for i, e in enumerate(per_step):
        print(f"  step {i+1}: {e:.6f}")

    # 2. Rollout accuracy
    rollout_errors = eval_rollout(model, val_loader, device)
    print(f"\n=== Rollout Error (autoregressive) ===")
    for i, e in enumerate(rollout_errors):
        print(f"  t+{i+1}: {e:.6f}")

    # 3. GIF visualisation
    print(f"\n=== Generating Prediction GIFs ===")
    make_nn_gif(model, val_loader, full_dataset, device,
                n_seqs=args.n_seqs, out_dir=out_dir)

    print(f"\nDone. Results in {out_dir}/")


if __name__ == "__main__":
    main()
