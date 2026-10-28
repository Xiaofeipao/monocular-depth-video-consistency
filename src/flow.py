"""Backward sampling uses current-to-previous flow, in pixel units."""
import cv2
import numpy as np


def warp(source, backward_flow):
    h, w = backward_flow.shape[:2]
    if source.shape[:2] != (h, w) or backward_flow.shape != (h, w, 2):
        raise ValueError("Flow/source dimensions do not agree")
    y, x = np.indices((h, w), dtype=np.float32)
    mx, my = x + backward_flow[..., 0], y + backward_flow[..., 1]
    valid = np.isfinite(mx) & np.isfinite(my) & (mx >= 0) & (mx <= w-1) & (my >= 0) & (my <= h-1)
    sampled = cv2.remap(np.asarray(source, np.float32), np.nan_to_num(mx).astype(np.float32),
                        np.nan_to_num(my).astype(np.float32), cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return sampled, valid


def consistency(forward, backward, threshold=1.0):
    sampled, valid = warp(forward, backward)
    error = np.linalg.norm(backward + sampled, axis=-1)
    return valid & np.isfinite(error) & (error <= threshold), error


class FarnebackFlow:
    name = "farneback"

    def __call__(self, previous, current):
        a, b = (cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in (previous, current))
        args = (None, .5, 3, 21, 3, 5, 1.2, 0)
        return cv2.calcOpticalFlowFarneback(a, b, *args), cv2.calcOpticalFlowFarneback(b, a, *args)


class RAFTFlow:
    name = "raft-small"

    def __init__(self, device="cuda", checkpoint=None):
        import torch
        from torchvision.models.optical_flow import raft_small
        if checkpoint is None:
            raise ValueError("Pass a local torchvision RAFT-Small state dict; no implicit download")
        self.torch, self.device = torch, device
        self.model = raft_small(weights=None).to(device).eval()
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))

    def __call__(self, previous, current):
        torch = self.torch
        h, w = previous.shape[:2]
        # RAFT's correlation pyramid needs at least 128 pixels on both axes.
        hh, ww = max(128, (h+7)//8*8), max(128, (w+7)//8*8)
        def tensor(image):
            image = cv2.resize(image[..., ::-1], (ww, hh))
            return torch.from_numpy(image.copy()).permute(2,0,1)[None].to(self.device).float()/127.5-1
        a, b = tensor(previous), tensor(current)
        def predict(a, b):
            with torch.inference_mode():
                f = self.model(a, b)[-1][0].permute(1,2,0).cpu().numpy()
            f = cv2.resize(f, (w, h))
            return f * np.array([w/ww, h/hh], np.float32)
        return predict(a, b), predict(b, a)
