import numpy as np
from PIL import Image

n_pula = 0
for i in range(500):
    p = ds.data[i * 31][0]
    a = np.asarray(Image.open(p).convert("L"), dtype=np.float32)
    lo, hi = np.percentile(a, [5, 95])
    if hi - lo <= 10:
        n_pula += 1
print(f"{n_pula}/500 sem normalizacao")
