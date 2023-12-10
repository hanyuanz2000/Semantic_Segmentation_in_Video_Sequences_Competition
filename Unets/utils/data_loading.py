import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt

class one_to_one_Segmentation_Dataset(Dataset):
    """
    return X, Y where
    X is a single frame, Y is the mask of the same frame.
    """
    def __init__(self, root_dir, subset='train', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        assert subset in ['train', 'val'], 'subset must be either train or val'
        self.subset = subset

        subset_dir = os.path.join(root_dir, subset)
        video_folders = [os.path.join(subset_dir, f) for f in sorted(os.listdir(subset_dir)) if os.path.isdir(os.path.join(subset_dir, f))]

        # Flatten the list of images and corresponding masks
        self.samples = []
        for folder in video_folders:
            for i in range(22):  # 22 frames per video
                image_path = os.path.join(folder, f'image_{i}.png')
                mask_path = os.path.join(folder, 'mask.npy')
                self.samples.append((image_path, mask_path, i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, mask_path, frame_idx = self.samples[idx]

        # Load image
        image = [Image.open(image_path).convert('RGB')]

        # Load mask
        mask = np.load(mask_path)[frame_idx][np.newaxis, :, :]
        mask = torch.from_numpy(mask).float()

        if self.transform:
            image, mask = self.transform(image, mask)
            image = image[0]

        else:
            image = transforms.ToTensor()(image[0])

        return image, mask

class LastFrame_and_Mask_Dataset(Dataset):
    """
        return X, Y where
        X is the last frame, Y is the mask of the last frame.
        Args:
            root_dir (string): Directory with all the videos (subfolders).
            subset (string): 'train', 'val' to specify the dataset type.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
    def __init__(self, root_dir, subset='train', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        assert subset in ['train', 'val'], 'subset must be either train or val'
        self.subset = subset
        subset_dir = os.path.join(root_dir, subset)
        self.video_folders = [f for f in sorted(os.listdir(subset_dir)) if os.path.isdir(os.path.join(subset_dir, f))]

    def __len__(self):
        return len(self.video_folders)

    def __getitem__(self, idx):
        video_folder = self.video_folders[idx]
        video_path = os.path.join(self.root_dir, self.subset, video_folder)

        # Load the last frame
        frame = [Image.open(os.path.join(video_path, 'image_21.png')).convert('RGB')]
        
        # Load mask corresponding to the last frame
        mask = np.load(os.path.join(video_path, 'mask.npy'))[-1][np.newaxis, :, :]
        mask = torch.from_numpy(mask)

        if self.transform:
            frame, mask = self.transform(frame, mask)
            frame = frame[0]
        else:
            frame = transforms.ToTensor()(frame[0])

        return frame, mask

class Labeled_Segementation_Dataset(Dataset):
    def __init__(self, root_dir, subset='train', transform=None):
        """
        return X, Y where
        X is the first 11 frames, Y is the mask of the 22nd frame.
        Args:
            root_dir (string): Directory with all the videos (subfolders).
            subset (string): 'train', 'val' to specify the dataset type.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        # assert subset in ['train', 'val']
        assert subset in ['train', 'val'], 'subset must be either train or val'
        self.subset = subset
        subset_dir = os.path.join(root_dir, subset)
        self.video_folders = [f for f in sorted(os.listdir(subset_dir)) if os.path.isdir(os.path.join(subset_dir, f))]

    def __len__(self):
        return len(self.video_folders)

    def __getitem__(self, idx):
        video_folder = self.video_folders[idx]
        video_path = os.path.join(self.root_dir, self.subset, video_folder)

        # Load the first 11 frames
        frames = [Image.open(os.path.join(video_path, f'image_{i}.png')).convert('RGB') for i in range(11)]
        # Load mask
        mask = np.load(os.path.join(video_path, 'mask.npy'))[-1][np.newaxis, :, :]
        mask = torch.from_numpy(mask)

        if self.transform:
            frames, mask = self.transform(frames, mask)
            frames = torch.stack(frames)

        else:
            frames = torch.stack([transforms.ToTensor()(frame) for frame in frames])
            mask = mask.float()

        return frames, mask

class Hidden_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the videos (subfolders).
            subset (string): 'unlabeled' or 'hidden' to specify the dataset type.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.subset = 'hidden'
        subset_dir = os.path.join(root_dir, self.subset)
        self.video_folders = [f for f in sorted(os.listdir(subset_dir)) if os.path.isdir(os.path.join(subset_dir, f))]

    def __len__(self):
        return len(self.video_folders)

    def __getitem__(self, idx):
        video_folder = self.video_folders[idx]
        video_path = os.path.join(self.root_dir, self.subset, video_folder)

        # Load the first 11 frames
        frames = [Image.open(os.path.join(video_path, f'image_{i}.png')).convert('RGB') for i in range(11)]
        frames = torch.stack([transforms.ToTensor()(frame) for frame in frames])

        return frames, torch.zeros(1)

def plot_frame(ax, frame, title):
    """ Plot a single frame with title. """
    frame_image = frame.cpu().numpy()
    frame_image = np.transpose(frame_image, (1, 2, 0))

    # Normalize from [-1, 1] to [0, 1]
    frame_image = (frame_image + 1) / 2
    frame_image = np.clip(frame_image, 0, 1)  # Ensuring the values are within [0, 1]

    ax.imshow(frame_image)
    ax.set_title(title)

def test_hidden(transform=None):
    print(f'\n------Testing load and transform for hidden subset------')
    dataset = Hidden_Dataset(root_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

    for i_batch, sample_batched in enumerate(dataloader):
        frames, mask = sample_batched
        print(f'  Batch {i_batch + 1}:')
        print(f'  Frames size: {frames.size()}') # (batch_size, seq_len, C, H, W)
        print(f'  Mask size: {mask.size()}')
        
        if i_batch == 0:
            break

def test_labeled(subset, transform=None):
    print(f'\n------Testing load and transform for {subset} subset------')
    dataset = Labeled_Segementation_Dataset(root_dir, subset=subset, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

    for i_batch, sample_batched in enumerate(dataloader):
        frames, mask = sample_batched
        print(f'  Batch {i_batch + 1}:')
        print(f'  Frames size: {frames.size()}') # (batch_size, seq_len, C, H, W)
        print(f'  Mask size: {mask.size()}')
        
        if i_batch == 0:
            break

def test_one_to_one(subset, transform=None):
    print(f'\n------Testing load and transform for {subset} subset------')
    dataset = one_to_one_Segmentation_Dataset(root_dir, subset=subset, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    print(f'  Dataset size: {len(dataset)}')
    for i_batch, sample_batched in enumerate(dataloader):
        frame, mask = sample_batched
        print(f'  Batch {i_batch + 1}:')
        print(f'  Frame size: {frame.size()}') # (batch_size, seq_len, C, H, W)
        print(f'  Mask size: {mask.size()}')

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))  # Adjust figsize as needed
        
        plot_frame(axes[0], frame[0, :, :, :], 'image') 
        axes[1].imshow(mask.squeeze(0).squeeze(0))

        plt.show()
        
        if i_batch == 0:
            break

def test_last_frame_and_mask(subset, transform=None):
    print(f'\n------Testing load and transform for {subset} subset------')
    dataset = LastFrame_and_Mask_Dataset(root_dir, subset=subset, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

    for i_batch, sample_batched in enumerate(dataloader):
        frame, mask = sample_batched
        print(f'  Batch {i_batch + 1}:')
        print(f'  Frame size: {frame.size()}') # (batch_size, seq_len, C, H, W)
        print(f'  Mask size: {mask.size()}')

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))  # Adjust figsize as needed
        
        plot_frame(axes[0], frame[0, :, :, :], 'image') 
        axes[1].imshow(mask.squeeze(0).squeeze(0))

        plt.show()
        
        if i_batch == 0:
            break

if __name__ == '__main__':
    # Example usage and Test
    root_dir = '/Users/zhanghanyuan/Document/Git/Semantic_Segmentation_in_Video_Sequences_Competition/Data'

    # Test transforms
    from customized_transform import SegmentationTrainingTransform, SegmentationValidationTransform

    # test_hidden('hidden')
    # test_labeled('train', transform=SegmentationTrainingTransform())
    # test_labeled('train')
    # test_labeled('val')
    # test_one_to_one('train', transform=SegmentationTrainingTransform())
    test_last_frame_and_mask('train', transform=SegmentationTrainingTransform())

