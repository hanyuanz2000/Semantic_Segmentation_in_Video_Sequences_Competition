import os
import h5py
import torch
import numpy as np
from PIL import Image

def convert_video_to_hdf5(video_path, hdf5_path, seq_len=11):
    """
    Convert the first 'seq_len' frames of a video into a single HDF5 file.

    Args:
        video_path (string): Path to the video directory containing the frames.
        hdf5_path (string): Path to save the HDF5 file.
        seq_len (int): Number of frames to include.
    """
    # Read and stack the frames
    frames = [Image.open(os.path.join(video_path, f'image_{i}.png')).convert('RGB') for i in range(seq_len)]
    frames = np.stack([np.array(frame) for frame in frames])  # Convert to numpy array and stack
    frames = frames.transpose((0, 3, 1, 2))  # Rearrange dimensions to seq_len, C, H, W

    # Save to HDF5
    with h5py.File(hdf5_path, 'w') as h5f:
        h5f.create_dataset('video', data=frames)

if __name__ == '__main__':
    root_dir = '/Users/zhanghanyuan/Document/Git/Semantic_Segmentation_in_Video_Sequences_Competition/Data'  
    # video_folders = os.listdir(os.path.join(root_dir, 'train'))

    # for video_folder in video_folders:
    #     video_path = os.path.join(root_dir, 'train', video_folder)

    # test 1 video with convert_video_to_hdf5
    video_path = os.path.join(root_dir, 'train', 'video_0')
    hdf5_path = '/Users/zhanghanyuan/Downloads/video_0.hdf5'

    convert_video_to_hdf5(video_path, hdf5_path)
