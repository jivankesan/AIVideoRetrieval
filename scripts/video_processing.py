from decord import VideoReader, cpu
import numpy as np
from huggingface_hub import hf_hub_download

def process_video(video_path):
    
    def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
        '''
        Sample a given number of frame indices from the video.
        Args:
            clip_len (`int`): Total number of frames to sample.
            frame_sample_rate (`int`): Sample every n-th frame.
            seg_len (`int`): Maximum allowed index of sample's last frame.
        Returns:
            indices (`List[int]`): List of sampled frame indices
        '''
        converted_len = int(clip_len * frame_sample_rate)
        end_idx = np.random.randint(converted_len, seg_len)
        start_idx = end_idx - converted_len
        indices = np.linspace(start_idx, end_idx, num=clip_len)
        indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
        return indices
    
    
    video_reader = VideoReader(video_path, num_threads=1, ctx=cpu(0))
    video_reader.seek(0)
    indices = sample_frame_indices(clip_len=8, frame_sample_rate=1, seg_len=len(video_reader))
    video = video_reader.get_batch(indices).asnumpy()
    
    return video

if __name__ == "__main__":
    video_path = "test.mp4"
    res = process_video(video_path)
    print(res.shape)