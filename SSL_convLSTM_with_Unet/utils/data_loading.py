import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt

class SSL_Reconstruction_Dataset(Dataset):
    """
    return X, Y where
    X is the first 11 frames, Y is the 22nd frame.
    here we will transform both X and Y
    
    Args:
        root_dir (string): Directory with all the video folders.
        transform (callable, optional): Optional transform to be applied on a sample.
    """
    def __init__(self, root_dir, subset, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.subset = subset

        # Accumulate all video folder paths from both train and unlabeled directories
        subset_dir = os.path.join(root_dir, subset)
        self.video_folders = [f for f in sorted(os.listdir(subset_dir)) if os.path.isdir(os.path.join(subset_dir, f))]

    def __len__(self):
        return len(self.video_folders)

    def __getitem__(self, idx):
        video_folder = self.video_folders[idx]
        video_path = os.path.join(self.root_dir, self.subset, video_folder)
        
        # Load the first 11 frames
        frames = [Image.open(os.path.join(video_path, f'image_{i}.png')).convert('RGB') for i in range(11)]
        
        # Load the 22nd frame
        frame_22 = Image.open(os.path.join(video_path, 'image_21.png')).convert('RGB')

        if self.transform:

            # Apply the transform to each frame individually
            frames, frame_22 = self.transform(frames, frame_22)

            # ensure the frame is in C x H x W format
            frame_22 = frame_22.permute(2, 0, 1) if frame_22.shape[-1] == 3 else frame_22

            # stack the frames into a single tensor, in C x T x H x W format
            frames = torch.stack(frames)

        else:
            # If no transform is specified, convert images to tensor
            frames = torch.stack([transforms.ToTensor()(frame) for frame in frames])
            frame_22 = transforms.ToTensor()(frame_22)
        
        # Return the first 11 frames and the 22nd frame
        return frames, frame_22

class VideoFrameDataPt(Dataset):
    def __init__(self, root_dir ,subset='unlabeled_partition', frame_target=False, first=0, last = 3, transform=None):
        
        """
        Args:
            root_dir (string): Directory with all the videos (subfolders).
            subset (string): 'train', 'val', or 'unlabeled' to specify the dataset type.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.subset = subset
        self.subset_dir = os.path.join(root_dir, subset)
    
        first = max(0, first)
        last = min(last, len(os.listdir(self.subset_dir)))
        print(f'Start to call VideoFrameDataPt dataset with first {first} and last {last} files')
        
        files = sorted(os.listdir(os.path.join(root_dir, subset)))[first: last]
        self.frames = []
        self.target = []
        for f in files:
            f_name = '/'.join([root_dir, subset, f])
            tensor = torch.load(f_name)
            # first 11 frames in each partition as frames, last as target
            frames = tensor['frames']
            target = tensor['future_frames']
            self.frames.append(frames)
            self.target.append(target)

        self.frames = torch.cat(self.frames, dim=0)
        self.target = torch.cat(self.target, dim=0)

        print('concatenated frames shape: ', self.frames.shape)
        print('concatenated target shape: ', self.target.shape)
    
    def __len__(self):
        return self.frames.shape[0]

    def __getitem__(self, idx):
        frames = self.frames[idx] # single input dp
        target = self.target[idx]
        if self.transform:
            image_frames = [transforms.ToPILImage()(f) for f in frames]
            target = transforms.ToPILImage()(target)
            frames, target = self.transform(image_frames, target)
            frames = torch.stack(frames)

        return frames, target # convert input to list of tensors


def plot_frame(ax, frame, title):
    """ Plot a single frame with title. """
    frame_image = frame.cpu().numpy()
    frame_image = np.transpose(frame_image, (1, 2, 0))

    # Normalize from [-1, 1] to [0, 1]
    frame_image = (frame_image + 1) / 2
    frame_image = np.clip(frame_image, 0, 1)  # Ensuring the values are within [0, 1]

    ax.imshow(frame_image)
    ax.set_title(title)
    
def test_SSL(subset, transform=None):
    # Test the dataset class by loading the first sample
    print(f'Testing SSL {subset} dataset with transform: {transform}')
    dataset = SSL_Reconstruction_Dataset(root_dir, subset, transform=transform)
    print('Dataset length: ', len(dataset))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    for i_batch, sample_batched in enumerate(dataloader):
        frames, frame_22 = sample_batched

        print(f'frames type: {type(frames)}')
        print(f'frame_22 type: {type(frame_22)}')
        
        print(f'frames shape: {frames.shape}')
        print(f'frame_22 shape: {frame_22.shape}')
        
        # Plot the first and last frames
        first_frame = frames[0, 0, :, :, :]
        print(f'First frame shape: {first_frame.shape}')

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))  # Adjust figsize as needed
        
        plot_frame(axes[0], first_frame, 'First frame')
        plot_frame(axes[1], frame_22.squeeze(0), 'Frame 22')

        plt.show()

        break

def test_pt(subset, transform=None):
    # Test the dataset class by loading the first sample
    print(f'Testing pt {subset} dataset with transform: {transform}')
    dataset = VideoFrameDataPt(root_dir, subset, transform=transform)
    print('Dataset length: ', len(dataset))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    for i_batch, sample_batched in enumerate(dataloader):
        frames, frame_22 = sample_batched

        print(f'frames type: {type(frames)}')
        print(f'frames shape: {frames.shape}')
        print(f'frames: {frames[0, 0, :, :, :]}')
        frames_flatten = frames.view(-1)
        print(min(frames_flatten), max(frames_flatten))
        
        print(f'frame_22 type: {type(frame_22)}')
        print(f'frame_22 shape: {frame_22.shape}')
        print(f'frame_22: {frame_22.squeeze(0)}')
        frame_22_flatten = frame_22.view(-1)
        print(min(frame_22_flatten), max(frame_22_flatten))

        # Plot the first and last frames
        first_frame = frames[0, 0, :, :, :]
        print(f'First frame shape: {first_frame.shape}')

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))  # Adjust figsize as needed
        
        plot_frame(axes[0], first_frame, 'First frame')
        plot_frame(axes[1], frame_22.squeeze(0), 'Frame 22')

        plt.show()
        break

if __name__ == '__main__':
    # Example usage and Test
    root_dir = '/Users/zhanghanyuan/Document/Git/Semantic_Segmentation_in_Video_Sequences_Competition/Data'

    # Test transforms
    from customized_transform import SSLTrainingTransform
    from customized_transform import SegmentationTrainingTransform

    transform1 = SSLTrainingTransform()
    transform2 = SegmentationTrainingTransform()
    # test_SSL('train', transform=transform1)
    # test_SSL('train', transform=transform2)
    test_pt('train_partition', transform=transform1)
    
    
