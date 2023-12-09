import os
root_dir = '/Users/zhanghanyuan/Document/Git/Semantic_Segmentation_in_Video_Sequences_Competition/Data'
subset = 'unlabeled_partition'
subset_dir = os.path.join(root_dir, subset)
files = sorted(os.listdir(os.path.join(root_dir, subset)))[0:6]
for f in files:
    f_name = '/'.join([root_dir, subset, f])
    print(f_name)
