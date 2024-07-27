import cv2
import numpy as np
import torch
from torchvision import transforms
from .video_transforms import Compose, Resize, CenterCrop, Normalize, ClipToTensor

class VideoProcessor:
    def __init__(self, clip_len=8, frame_sample_rate=2, crop_size=224, short_side_size=256, 
                 new_height=256, new_width=340, keep_aspect_ratio=True):
        self.clip_len = clip_len
        self.frame_sample_rate = frame_sample_rate
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.new_height = new_height
        self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio

        self.data_transform = Compose([
            Resize(self.short_side_size, interpolation='bilinear'),
            CenterCrop(size=(self.crop_size, self.crop_size)),
            ClipToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess_video(self, video_path):
        buffer = self.load_video(video_path)
        if len(buffer) == 0:
            raise RuntimeError(f"Video {video_path} could not be loaded correctly")
        
        buffer = self.data_transform(buffer)
        return buffer

    def load_video(self, video_path, chunk_nb=0):
        try:
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_indices = self._get_seq_frames(frame_count, self.clip_len, clip_idx=chunk_nb)

            buffer = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    raise RuntimeError(f"Error reading frame {idx} from video {video_path}")
                if not self.keep_aspect_ratio:
                    frame = cv2.resize(frame, (self.new_width, self.new_height))
                buffer.append(frame)

            cap.release()
            buffer = np.array(buffer)
            return buffer
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            return []

    def _get_seq_frames(self, video_size, num_frames, clip_idx=-1):
        seg_size = max(0., float(video_size - 1) / num_frames)
        max_frame = int(video_size) - 1
        seq = []
        if clip_idx == -1:
            for i in range(num_frames):
                start = int(np.round(seg_size * i))
                end = int(np.round(seg_size * (i + 1)))
                idx = min(np.random.randint(start, end), max_frame)
                seq.append(idx)
        else:
            duration = seg_size / 2
            for i in range(num_frames):
                start = int(np.round(seg_size * i))
                frame_index = start + int(duration * (clip_idx + 1))
                idx = min(frame_index, max_frame)
                seq.append(idx)
        return seq

# Example usage
if __name__ == "__main__":
    video_path = "path/to/your/video.mp4"
    processor = VideoProcessor()
    processed_video = processor.preprocess_video(video_path)
    print("Processed video shape:", processed_video.shape)