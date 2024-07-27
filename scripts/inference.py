import torch
from mamba_model import VideoMamba

def run_inference(segments):
    model = VideoMamba()
    results = []
    for segment in segments:
        result = model.infer(segment)
        results.append(result)
    return results