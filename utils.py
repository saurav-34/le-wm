import numpy as np
import torch
from pathlib import Path
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()

    def norm_fn(x):
        return ((x - mean) / std).float()

    normalizer = dt.transforms.WrapTorchTransform(norm_fn, source=source, target=target)
    return normalizer

class ModelObjectCallBack(Callback):
    """Callback to pickle model object after each epoch."""

    def __init__(self, dirpath, filename="model_object", epoch_interval: int = 1):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        output_path = (
            self.dirpath
            / f"{self.filename}_epoch_{trainer.current_epoch + 1}_object.ckpt"
        )

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._dump_model(pl_module.model, output_path)

            # save final epoch
            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._dump_model(pl_module.model, output_path)

    def _dump_model(self, model, path):
        try:
            torch.save(model, path)
        except Exception as e:
            print(f"Error saving model object: {e}")


class WandbPredictionVizCallback(Callback):
    """At the end of each epoch, picks a val sequence, runs the world model
    autoregressively to predict future embeddings, retrieves the nearest
    neighbor frame from the dataset for each predicted embedding, and logs
    a side-by-side video to WandB:
        Left  — ground truth frames
        Right — model's predicted frames (nearest neighbor retrieval)
    """

    def __init__(self, val_loader, full_dataset, ctx_len: int,
                 num_sequences: int = 4, epoch_interval: int = 5, fps: int = 4):
        super().__init__()
        self.val_loader = val_loader
        self.ctx_len = ctx_len
        self.num_sequences = num_sequences
        self.epoch_interval = epoch_interval
        self.fps = fps

        # pre-build reference bank: encode all frames for nearest-neighbor lookup
        # we do this lazily on first call to avoid blocking startup
        self._ref_pixels = None   # (N, H, W, C) uint8 numpy
        self._ref_embs = None     # (N, D) torch
        self._full_dataset = full_dataset

    @torch.no_grad()
    def _build_ref_bank(self, model, device, img_size):
        """Encode a representative sample of frames from the dataset."""
        from stable_pretraining import data as dt

        MAX_REF = 2000  # cap to keep memory reasonable
        pixels_list, embs_list = [], []

        for i, batch in enumerate(self.val_loader):
            if len(pixels_list) * batch["pixels"].shape[0] >= MAX_REF:
                break
            px = batch["pixels"].to(device)           # (B, T, C, H, W)
            B, T = px.shape[:2]
            px_flat = px.view(B * T, *px.shape[2:])   # (B*T, C, H, W)
            out = model.encoder(px_flat.float(), interpolate_pos_encoding=True)
            emb = model.projector(out.last_hidden_state[:, 0])  # (B*T, D)
            embs_list.append(emb.cpu())

            # store raw uint8 pixels for display (undo imagenet norm approx)
            px_np = px_flat.permute(0, 2, 3, 1).cpu().numpy()  # (B*T, H, W, C)
            # reverse imagenet normalisation to get ~uint8
            mean = np.array([0.485, 0.456, 0.406])
            std  = np.array([0.229, 0.224, 0.225])
            px_np = (px_np * std + mean).clip(0, 1)
            px_np = (px_np * 255).astype(np.uint8)
            pixels_list.append(px_np)

        self._ref_embs   = torch.cat(embs_list, dim=0)    # (N, D)
        self._ref_pixels = np.concatenate(pixels_list, axis=0)  # (N, H, W, C)

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        if not HAS_WANDB:
            return
        if not trainer.is_global_zero:
            return
        if (trainer.current_epoch + 1) % self.epoch_interval != 0:
            return
        if trainer.logger is None:
            return

        model = pl_module.model
        model.eval()
        device = next(model.parameters()).device

        # build reference bank once
        if self._ref_embs is None:
            self._build_ref_bank(model, device, img_size=None)

        ref_embs = self._ref_embs.to(device)  # (N, D)

        videos = []
        collected = 0

        for batch in self.val_loader:
            if collected >= self.num_sequences:
                break

            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            batch["action"] = torch.nan_to_num(batch["action"], 0.0)

            output = model.encode(batch)
            emb     = output["emb"]      # (B, T, D)
            act_emb = output["act_emb"]  # (B, T, A)
            px      = batch["pixels"]    # (B, T, C, H, W)
            B, T    = emb.shape[:2]

            for b in range(min(B, self.num_sequences - collected)):
                ctx_emb = emb[b:b+1, :self.ctx_len]          # (1, ctx, D)
                ctx_act = act_emb[b:b+1, :self.ctx_len]      # (1, ctx, A)

                # autoregressively predict future embeddings
                pred_embs = []
                cur_emb = ctx_emb.clone()
                cur_act = ctx_act.clone()
                n_future = T - self.ctx_len

                for _ in range(n_future):
                    next_emb = model.predict(cur_emb, cur_act)[:, -1:]  # (1,1,D)
                    pred_embs.append(next_emb)
                    cur_emb = torch.cat([cur_emb[:, 1:], next_emb], dim=1)
                    cur_act = torch.cat([cur_act[:, 1:],
                                         act_emb[b:b+1, cur_act.shape[1]:cur_act.shape[1]+1]], dim=1)

                if not pred_embs:
                    continue

                pred_embs = torch.cat(pred_embs, dim=1).squeeze(0)  # (n_future, D)

                # nearest-neighbor retrieval for each predicted embedding
                # cosine similarity for robustness
                pred_norm = pred_embs / (pred_embs.norm(dim=-1, keepdim=True) + 1e-8)
                ref_norm  = ref_embs  / (ref_embs.norm(dim=-1, keepdim=True) + 1e-8)
                sim = pred_norm @ ref_norm.T          # (n_future, N)
                nn_idx = sim.argmax(dim=-1).cpu().numpy()  # (n_future,)

                retrieved_frames = self._ref_pixels[nn_idx]  # (n_future, H, W, C)

                # ground truth future frames
                gt_px = px[b, self.ctx_len:].permute(0, 2, 3, 1).cpu().numpy()  # (n_future, H, W, C)
                mean = np.array([0.485, 0.456, 0.406])
                std  = np.array([0.229, 0.224, 0.225])
                gt_px = (gt_px * std + mean).clip(0, 1)
                gt_px = (gt_px * 255).astype(np.uint8)

                # side-by-side: GT (left) | predicted (right)
                h, w = gt_px.shape[1], gt_px.shape[2]
                divider = np.ones((n_future, h, 4, 3), dtype=np.uint8) * 128
                side_by_side = np.concatenate([gt_px, divider, retrieved_frames], axis=2)
                # wandb video wants (T, H, W, C)
                videos.append(side_by_side)
                collected += 1

        if videos:
            # log each sequence as a separate wandb.Video
            log_dict = {"epoch": trainer.current_epoch + 1}
            for i, vid in enumerate(videos):
                # wandb.Video expects (T, C, H, W)
                vid_chw = vid.transpose(0, 3, 1, 2)
                log_dict[f"val/prediction_video_{i}"] = wandb.Video(
                    vid_chw, fps=self.fps, format="gif"
                )
            trainer.logger.experiment.log(log_dict)