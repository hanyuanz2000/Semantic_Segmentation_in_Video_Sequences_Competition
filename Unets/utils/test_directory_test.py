import os
root_dir = '/Users/zhanghanyuan/Document/Git/Semantic_Segmentation_in_Video_Sequences_Competition/Data'
subset = 'val'
subset_dir = os.path.join(root_dir, subset)
video_folders = [os.path.join(subset_dir, f) for f in sorted(os.listdir(subset_dir)) if os.path.isdir(os.path.join(subset_dir, f))]
print(video_folders)