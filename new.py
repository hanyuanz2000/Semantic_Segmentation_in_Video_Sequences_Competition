class VideoFrameDataPt(Dataset):
    def __init__(self, root_dir='data' ,subset='train', frame_target=False, transform=None):
        """
        Args:
            root_dir (string): Directory with all the videos (subfolders).
            subset (string): 'train', 'val', or 'unlabeled' to specify the dataset type.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
#         self.frame_target = frame_target
        self.subset = subset
        self.input_file_name = os.path.join(root_dir, subset+'.pt')
        if frame_target:
            self.target_file_name = os.path.join(root_dir, 'target_'+szhegubset+'_frame.pt')
        else:
            self.target_file_name = os.path.join(root_dir, 'target_'+subset+'_mask.pt')
            
        self.frames = torch.load(self.input_file_name)
        self.target = torch.load(self.target_file_name)
        
#         self.video_folders = [f for f in sorted(os.listdir(subset_dir)) if os.path.isdir(os.path.join(subset_dir, f))]
        # example: ['video_000', 'video_001', ...]
#         self.img_list = ['image_' + str(i) + '.png' for i in range(11)]
        # example: ['image_0.png', 'image_1.png', ...]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frames = [i for i in self.frames[idx]] # single input dp
        target = self.target[idx]
        if self.transform:
            image_frames = [transforms.ToPILImage()(f) for f in frames]
            frames, target = self.transform(image_frames, target)
        return frames, target # convert input to list of tensors