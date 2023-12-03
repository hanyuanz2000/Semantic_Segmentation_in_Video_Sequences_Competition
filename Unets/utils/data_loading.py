import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class VideoFrameDataset(Dataset):
    def __init__(self, root_dir, subset='train', transform=None):
        """
        Args:
            root_dir (string): Directory with all the videos (subfolders).
            subset (string): 'train', 'val', or 'unlabeled' to specify the dataset type.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.subset = subset
        subset_dir = os.path.join(root_dir, subset)
        self.video_folders = [f for f in sorted(os.listdir(subset_dir)) if os.path.isdir(os.path.join(subset_dir, f))]
        # example: ['video_000', 'video_001', ...]
        self.img_list = ['image_' + str(i) + '.png' for i in range(11)]
        # example: ['image_0.png', 'image_1.png', ...]

    def __len__(self):
        return len(self.video_folders)

    def __getitem__(self, idx):
        video_folder = self.video_folders[idx]
        video_path = os.path.join(self.root_dir, self.subset, video_folder)

        frames = []
        for i in range(11):
            frame_path = os.path.join(video_path, f'image_{i}.png')
            frame = Image.open(frame_path).convert("RGB")
            frames.append(frame)

            # check dimension (frame is a PIL image)
            # print(f'type of frame before transformation: {type(frame)}')
            # print(f'frame size before transform: {frame.size}')
        
        # Labeled data
        if self.subset == 'train' or self.subset == 'val':
            mask = np.load(os.path.join(video_path, 'mask.npy'))
            mask = torch.from_numpy(mask)
        
            if self.transform:
                frames, mask = self.transform(frames, mask)

                # check dimension
                # print(f'frame size after transform: {frames[0].size()}')
            else:
                frames = transforms.ToTensor()(frames)
                mask = transforms.ToTensor()(mask)

            return frames, mask
        
        # Unlabeled data
        else:
            if self.transform:
                frames = self.transform(frames)
            else:
                frames = [transforms.ToTensor()(frame) for frame in frames]  # Apply to each frame

            return frames, 0
       

def test(subset, transform=None):
    print(f'\n------Testing load and transform for {subset} subset------')
    dataset = VideoFrameDataset(root_dir, subset=subset, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    for i_batch, sample_batched in enumerate(dataloader):
        frames, mask = sample_batched
        concatenated_frames = torch.cat(frames, dim=1)
        print(f'Batch {i_batch + 1}:')
        print(f'  Number of frames in batch: {len(frames)}')
        print(f'  Frame size: {frames[0].size()}')
        print(f'  Mask size: {mask.size()}')
        print(f'  Concatenated frames size: {concatenated_frames.size()}')

        # Test only the first batch
        if i_batch == 0:
            break

if __name__ == '__main__':
    # Example usage and Test
    root_dir = '/Users/zhanghanyuan/Document/Git/Semantic_Segmentation_in_Video_Sequences_Competition/Data'

    # Test transforms
    from customized_transform import SegmentationTrainingTransform, SegmentationValidationTransform

    test('train', transform=SegmentationTrainingTransform())
    test('val', transform=SegmentationValidationTransform())
    test('unlabeled')
