import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class SSL_Reconstruction_Dataset(Dataset):
    """
    return X, Y where
    X is the first 11 frames, Y is the 22nd frame.
    
    Args:
        root_dir (string): Directory with all the video folders.
        transform (callable, optional): Optional transform to be applied on a sample.
    """
    def __init__(self, root_dir, subset, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        assert subset in ['unlabeled', 'train'], 'subset must be either train or unlabeled'
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
            
            # After transformation, ensure the frame is in C x H x W format
            frame_22 = frame_22.permute(2, 0, 1) if frame_22.shape[-1] == 3 else frame_22

            # stack the frames into a single tensor
            frames = torch.stack(frames)

        else:
            # If no transform is specified, convert images to tensor
            frames = torch.stack([transforms.ToTensor()(frame) for frame in frames])
            frame_22 = transforms.ToTensor()(frame_22)
        
        # Return the first 11 frames and the 22nd frame
        return frames, frame_22

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
    
def test(subset, transform=None):
    # Test the dataset class by loading the first sample
    dataset = SSL_Reconstruction_Dataset(root_dir, subset, transform=transform)
    print('Dataset length: ', len(dataset))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

    for i_batch, sample_batched in enumerate(dataloader):
        frames, frame_22 = sample_batched
        print(f'frames shape: {frames.shape}')
        print(f'frame_22 shape: {frame_22.shape}')
        break

if __name__ == '__main__':
    # Example usage and Test
    root_dir = '/Users/zhanghanyuan/Document/Git/Semantic_Segmentation_in_Video_Sequences_Competition/Data'

    # Test transforms
    from customized_transform import SegmentationTrainingTransform, SegmentationValidationTransform

    train_transform = SegmentationTrainingTransform()
    print('------test 1------')
    test('train', train_transform)
    print('------test 2------')
    test('unlabeled', train_transform)
    
    
