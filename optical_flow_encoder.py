import cv2
import numpy as np
import torch

def compute_dense_optical_flow(frames, stride=1, max_len=256):
    """
    frames: list of np.array frames (BGR)
    stride: compute flow between frame i and i + stride
    returns: flow_features: (T, H, W, 2) → mean pooled to (T, 2)
    """
    flows = []
    num_frames = len(frames)
    
    for i in range(0, num_frames - stride, stride):
        prev = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        next = cv2.cvtColor(frames[i + stride], cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prev, next, None,
                                            pyr_scale=0.5,
                                            levels=3,
                                            winsize=15,
                                            iterations=3,
                                            poly_n=5,
                                            poly_sigma=1.2,
                                            flags=0)
        # flow: (H, W, 2), compute average displacement
        avg_flow = np.mean(flow, axis=(0, 1))  # (2,)
        flows.append(avg_flow)

    # Pad or truncate to max_len
    flows = np.array(flows)
    if len(flows) < max_len:
        pad_len = max_len - len(flows)
        pad = np.zeros((pad_len, 2))
        flows = np.concatenate([flows, pad], axis=0)
    else:
        flows = flows[:max_len]

    return flows  # shape: (T, 2)
