from data_loading import SSL_Reconstruction_Dataset
import torch
import os

# Define root directory and subsets
root_dir = '/Users/zhanghanyuan/Document/Git/Semantic_Segmentation_in_Video_Sequences_Competition/Data'
subsets = ['train']

partition_size = 1000

# Iterate over each subset and collect frames

# To-do:
# 1. checkpoint follow up
# 2. partition pt files
# 3. save the pt file at every 1000 files


for subset in subsets:
    dataset = SSL_Reconstruction_Dataset(root_dir=root_dir, subset=subset)
    # Initialize lists to store frames and future frames
    collected_frames, future_frames = [], []

    for i in range(len(dataset)):
        if (i+1) % partition_size == 0:
            print(f"Processing frame {i} in subset '{subset}', saving the caches now...")
            
            # Convert the lists to tensors
            collected_frames_tensor = torch.stack(collected_frames, dim=0)
            future_frames_tensor = torch.stack(future_frames, dim=0)
            print(f"Collected Frames Tensor Size: {collected_frames_tensor.size()}")
            print(f"Future Frames Tensor Size: {future_frames_tensor.size()}")
            
            # Save the tensors to a single file
            save_path = os.path.join(root_dir, f"{subset}_partition_{i//partition_size}.pt")
            torch.save({'frames': collected_frames_tensor, 'future_frames': future_frames_tensor}, save_path)
            print(f"Saved data to {save_path}, clear caching now...")
            
            # clear the caches for next partition
            collected_frames, future_frames = [], []
            print("Cache cleared! Go on")
            
            
        try:
            temp_frames, temp_future_frame = dataset[i]

            # Append both or none - now inside the try block
            collected_frames.append(temp_frames)
            future_frames.append(temp_future_frame)

        except Exception as e:
            print(f"Error processing frame {i}: {e}")
            # No need for 'continue' as it's the last statement in the loop

    if collected_frames:
        # Save the tensors to a single file
        save_path = os.path.join(root_dir, f"{subset}_partition_{len(dataset)//partition_size}.pt")
        torch.save({'frames': collected_frames_tensor, 'future_frames': future_frames_tensor}, save_path)