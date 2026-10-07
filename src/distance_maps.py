"""Online 2D signed distance maps in pixel units."""
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from torch.utils.data import Dataset


def one_hot_to_sdf(gt: torch.Tensor, classes: list[int]) -> torch.Tensor:
    """Accept (B,K,H,W); negative inside, positive outside, zero for absent classes."""
    if gt.ndim != 4:
        raise ValueError("Expected ground truth with shape (B,K,H,W)")
    if not classes or any(c < 0 or c >= gt.shape[1] for c in classes):
        raise ValueError("Distance-map classes must be nonempty and within the class range")
    masks = gt.detach().cpu().numpy()
    sdf = np.zeros(masks.shape, dtype=np.float32)
    for b in range(len(masks)):
        for c in classes:
            mask = masks[b, c].astype(bool)
            if mask.any():
                # Explicit exterior for full-frame objects, whose complement is empty.
                inside = (distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
                          if mask.all() else distance_transform_edt(mask))
                sdf[b, c] = distance_transform_edt(~mask) - inside
    return torch.from_numpy(sdf).to(gt.device)


class WithDistanceMaps(Dataset):
    """Wrap after augmentation so maps describe the actual training target."""
    def __init__(self, base, classes):
        self.base, self.classes = base, classes

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = dict(self.base[index])
        item["dist_maps"] = one_hot_to_sdf(item["gts"].unsqueeze(0), self.classes)[0]
        return item
