import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class Labeled_Segementation_Dataset(Dataset):
    def __init__(self, root_dir, subset='train', transform=None):
        """
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

if __name__ == '__main__':
    # Example usage and Test
    root_dir = '/Users/zhanghanyuan/Document/Git/Semantic_Segmentation_in_Video_Sequences_Competition/Data'

    # Test transforms
    from customized_transform import SegmentationTrainingTransform, SegmentationValidationTransform

    test_hidden('hidden')
    # test_labeled('train', transform=SegmentationTrainingTransform())
    # test_labeled('train')
    # test_labeled('val')

