import sys
sys.path.insert(0, "diffusionpen/DiffusionPen")
from diffusionpen.DiffusionPen.utils.bressay_dataset import BRESSAY_Dataset
import diffusionpen.DiffusionPen.utils.bressay_dataset as m
print(m.__file__)
ds = BRESSAY_Dataset("./bressay_split", "train", transforms=None)
for i in range(5):
    p = ds.data[i * 997][0]
    ds.load_image(p).save(f"/tmp/check_{i}.png")
    print(p)
print("veja /tmp/check_*.png")