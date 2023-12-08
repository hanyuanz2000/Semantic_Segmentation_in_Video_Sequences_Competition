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

class Labeled_Segementation_Dataset(Dataset):
    def __init__(self, root_dir, subset='train', transform=None):
        """
        return X, Y where
        X is the first 11 frames, Y is the the mask of the 22nd frame.
        Here for some operations, we only transform the image, but not the mask.

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
        return X, Y where
        X is the first 11 frames, Y is zero.

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

class VideoFrameDataPt(Dataset):
    def __init__(self, root_dir ,subset='unlabeled_partition', frame_target=False, first=None, last = None, transform=None):
        """
        Args:
            root_dir (string): Directory with all the videos (subfolders).
            subset (string): 'train', 'val', or 'unlabeled' to specify the dataset type.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.subset = subset
        
        # handle partition pt and extract first n files
        if first and last:
            files = sorted(os.listdir(os.path.join(root_dir, subset)))[first: last]
            self.frames = []
            self.target = []
            for f in files:
                f_name = '/'.join([root_dir, subset, f_name])
                tensor = torch.load(f_name)
                print(tensor.size())
                # first 11 frames in each partition as frames, last as target
                frames = tensor[:, :11, :, :, :]
                target = tensor[:, 22, :, :, :]
                self.frames.append(frames)
                self.target.append(target)

            self.frames = torch.cat(self.frames, dim=0)
            self.target = torch.cat(self.target, dim=0)

        # handle pt file with no partitions
        else:
            input_file_name = os.path.join(root_dir, subset+'.pt')
            if frame_target:
                target_file_name = os.path.join(root_dir, 'target_'+subset+'_frame.pt')
            else:
                target_file_name = os.path.join(root_dir, 'target_'+subset+'_mask.pt')
            
            self.frames = torch.load(input_file_name)
            self.target = torch.load(target_file_name)

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

if __name__ == '__main__':
    # Example usage and Test
    root_dir = '/Users/zhanghanyuan/Document/Git/Semantic_Segmentation_in_Video_Sequences_Competition/Data'

    # Test transforms
    from customized_transform import SSLTrainingTransform
    from customized_transform import SegmentationTrainingTransform

    transform1 = SSLTrainingTransform()
    transform2 = SegmentationTrainingTransform()
    test_SSL('train', transform=transform1)
    test_SSL('train', transform=transform2)
    
    
