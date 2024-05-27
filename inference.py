from pathlib import Path

import torch
from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
from PIL import Image

SAMPLE_IMAGE_PATH = (
    Path(__file__).resolve().parent / "assets" / "samples" / "sample1.jpg"
)

# load model and preprocessor
model, preprocessor = convert_v2_5_from_siglip(
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
model = model.to(torch.bfloat16).cuda()

# load image to evaluate
image = Image.open(SAMPLE_IMAGE_PATH).convert("RGB")

# preprocess image
pixel_values = (
    preprocessor(images=image, return_tensors="pt")
    .pixel_values.to(torch.bfloat16)
    .cuda()
)

# predict aesthetic score
with torch.inference_mode():
    score = model(pixel_values).logits.squeeze().float().cpu().numpy()

# print result
print(f"Aesthetics score: {score:.2f}")
