import torch
import os, shutil, time, stat

def _clear_readonly(p: str):
    try:
        os.chmod(p, stat.S_IWRITE | stat.S_IREAD)
    except OSError:
        pass

def save_if_best(model, train_loss, path="pca_net_best.pt", meta=None):
    prev = float('inf')
    if os.path.exists(path):
        ckpt = torch.load(path, map_location="cpu")
        prev = ckpt.get("best_train_loss", float('inf'))

    if train_loss < prev:
        payload = {"model_state": model.state_dict(),
                   "best_train_loss": float(train_loss),
                   "meta": meta or {}}

        tmp = f"{path}.tmp.{os.getpid()}"
        torch.save(payload, tmp)

        # retry replace to dodge transient locks
        for _ in range(10):
            try:
                if os.path.exists(path):
                    _clear_readonly(path)         # in case the file is read-only
                os.replace(tmp, path)            # atomic on same volume
                print(f"Saved new best ({train_loss:.3e} < {prev:.3e}) → {path}")
                break
            except PermissionError:
                time.sleep(0.2)
        else:
            # last resort: try move (non-atomic) or raise the last error
            try:
                if os.path.exists(path):
                    _clear_readonly(path)
                    os.remove(path)
                shutil.move(tmp, path)
                print(f"Saved new best (non-atomic move) → {path}")
            finally:
                if os.path.exists(tmp):
                    try: os.remove(tmp)
                    except OSError: pass