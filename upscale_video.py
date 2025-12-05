import cv2
import glob
import os
import torch
from gfpgan import GFPGANer
from tqdm import tqdm
import imageio

# --- CONFIGURATION ---
input_video = "face_morphing_720p.mp4" # The video we made earlier
output_video = "face_morphing_HD_AI.mp4"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"🚀 Starting AI Restoration on {device}...")

# 1. Initialize the GFPGAN model (It will download the weights automatically)
# This model is specialized for cleaning up blurry human faces
restorer = GFPGANer(
    model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
    upscale=2,    # Upscale factor (2x)
    arch='clean', 
    channel_multiplier=2, 
    bg_upsampler=None
)

# 2. Extract frames from video
print("extracting frames...")
reader = imageio.get_reader(input_video)
fps = reader.get_meta_data()['fps']
frames = []
for im in reader:
    frames.append(im)
reader.close()

# 3. Process frames
print(f"✨ Restoring {len(frames)} frames (This might take a while)...")
restored_frames = []

for frame in tqdm(frames):
    # Convert RGB to BGR (OpenCV format)
    img_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Run AI inference
    # cropped_faces, restored_faces, restored_img
    _, _, output = restorer.enhance(img_bgr, has_aligned=False, only_center_face=False, paste_back=True)
    
    # Convert back to RGB
    img_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    restored_frames.append(img_rgb)

# 4. Save new video
print(f"💾 Saving High-Quality Video to {output_video}...")
imageio.mimsave(output_video, restored_frames, fps=fps)
print("✅ Done! Compare the two videos now!")