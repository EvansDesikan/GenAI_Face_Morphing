import torch
import torch.nn as nn
import numpy as np
import imageio
import os
from PIL import Image  # Using Pillow to resize images

# --- CONFIGURATION ---
model_path = "generator_final.pth"
output_video = "face_morphing_720p.mp4" # New filename
fps = 30
duration_sec = 15
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
nz = 100
ngf = 64
target_size = (720, 720) # Target HD resolution

print(f"🎬 Preparing to generate 720p video on {device}...")

# --- 1. DEFINE GENERATOR ---
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# --- 2. LOAD THE BRAIN ---
netG = Generator().to(device)
if os.path.exists(model_path):
    netG.load_state_dict(torch.load(model_path, map_location=device))
    print("✅ Model loaded successfully!")
else:
    print(f"❌ Error: Could not find '{model_path}'. Check your folder.")
    exit()
    
netG.eval()

# --- 3. HELPER: SLERP ---
def slerp(val, low, high):
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm*high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1)*high
    return res

# --- 4. GENERATION LOOP ---
print("🚀 Generating frames and upscaling to 720p...")

num_keyframes = int(duration_sec / 2)
keyframes = [torch.randn(1, nz, 1, 1, device=device) for _ in range(num_keyframes + 1)]

all_frames = []
frames_per_morph = fps * 2 

for i in range(num_keyframes):
    start_z = keyframes[i]
    end_z = keyframes[i+1]
    
    for step in range(frames_per_morph):
        val = step / float(frames_per_morph)
        z = slerp(val, start_z, end_z)
        
        with torch.no_grad():
            fake = netG(z).detach().cpu()
        
        # 1. Convert to simple Image
        img_tensor = (fake[0] * 0.5 + 0.5).clamp(0, 1)
        img_array = img_tensor.mul(255).byte().permute(1, 2, 0).numpy()
        
        # 2. Resize using Pillow (The Magic Step)
        img_pil = Image.fromarray(img_array)
        # BICUBIC mode makes it smoother than just stretching pixels
        img_pil = img_pil.resize(target_size, Image.BICUBIC) 
        
        all_frames.append(np.array(img_pil))

# --- 5. SAVE VIDEO ---
print(f"💾 Saving {len(all_frames)} HD frames to {output_video}...")
imageio.mimsave(output_video, all_frames, fps=fps)
print(f"✨ DONE! Watch 'face_morphing_720p.mp4'")