import os
import shutil
import subprocess
import imageio
from tqdm import tqdm

# --- CONFIGURATION ---
input_video = "face_morphing_HD_AI.mp4"    # Your HD video
output_video = "face_morphing_60fps.mp4"   # The final result
rife_exe = "rife-ncnn-vulkan.exe"          # The tool you just downloaded
model_name = "rife-v4.6"                   # The model folder name

# Temporary folders for the frames
temp_in = "temp_frames_in"
temp_out = "temp_frames_out"

def clean_folders():
    if os.path.exists(temp_in): shutil.rmtree(temp_in)
    if os.path.exists(temp_out): shutil.rmtree(temp_out)
    os.makedirs(temp_in)
    os.makedirs(temp_out)

print(f"🌊 Smoothing video: {input_video}...")

# 1. Extract Frames
clean_folders()
print("📂 Extracting frames to temp folder...")
reader = imageio.get_reader(input_video)
fps = reader.get_meta_data()['fps']
for i, frame in enumerate(reader):
    # Save as 00001.png, 00002.png, etc.
    imageio.imwrite(os.path.join(temp_in, f"{i:08d}.png"), frame)
reader.close()

# 2. Run the RIFE Executable
print("🚀 Running AI Interpolation (this may take a moment)...")
# Command: rife-ncnn-vulkan.exe -i input -o output -m model
cmd = [rife_exe, "-i", temp_in, "-o", temp_out, "-m", model_name]

try:
    subprocess.run(cmd, check=True)
except FileNotFoundError:
    print("❌ ERROR: Could not find rife-ncnn-vulkan.exe!")
    print("Please make sure you downloaded it and put it in this folder.")
    exit()

# 3. Stitch Frames back to Video
print("🎥 Stitching 60fps video...")
frames = []
# Read all images from the output folder (sorted by name)
files = sorted(os.listdir(temp_out))

# We use imageio writer to stream frames to video file
writer = imageio.get_writer(output_video, fps=fps*2) # Double the FPS!

for file in tqdm(files):
    if file.endswith(".png"):
        file_path = os.path.join(temp_out, file)
        frame = imageio.imread(file_path)
        writer.append_data(frame)

writer.close()

# 4. Cleanup
print("🧹 Cleaning up temp files...")
clean_folders()
os.rmdir(temp_in)
os.rmdir(temp_out)

print(f"✨ DONE! Watch '{output_video}' for silky smooth motion.")