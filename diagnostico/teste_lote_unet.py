"""Erro do forward em funcao do tamanho do lote.

Lote 1 ja foi verificado contra a CPU (9.65e-07), entao ele serve de
referencia aqui. Mesma entrada replicada; compara a linha 0 de cada lote.
"""
import os, sys, argparse, torch, torch.nn as nn
from torch.nn import DataParallel
from transformers import CanineModel, CanineTokenizer

sys.path.insert(0, "/home/leo/repos/htg-tcc/diffusionpen/DiffusionPen")
os.chdir("/home/leo/repos/htg-tcc")
from unet import UNetModel

CKPT = os.environ.get("CKPT", "")
if not CKPT or not os.path.isfile(CKPT):
    raise SystemExit(
        "defina CKPT com um checkpoint .pt do UNet, por exemplo:\n"
        "  CKPT=./meu_modelo/models/ema_ckpt.pt python " + sys.argv[0])

a = argparse.Namespace(
    device="cuda:0", img_size=(64, 256), channels=4, emb_dim=320, num_heads=4,
    num_res_blocks=1, latent=True, img_feat=True, interpolation=False,
    mix_rate=None, model_name="diffusionpen", color=True, level="word",
    unet="unet_latent", batch_size=1, save_path="",
)

tok = CanineTokenizer.from_pretrained("google/canine-c")
te = DataParallel(CanineModel.from_pretrained("google/canine-c"), device_ids=[0]).to("cuda:0")
te.requires_grad_(False); te.eval()

unet = UNetModel(image_size=a.img_size, in_channels=4, model_channels=320, out_channels=4,
                 num_res_blocks=1, attention_resolutions=(1, 1), channel_mult=(1, 1),
                 num_heads=4, num_classes=339, context_dim=320, vocab_size=79,
                 text_encoder=te, args=a)
unet = DataParallel(unet, device_ids=[0]).to("cuda:0")
unet.load_state_dict(torch.load(CKPT, map_location="cuda:0", weights_only=True))
unet.eval()

torch.manual_seed(0)
x1 = torch.randn(1, 4, 8, 32, device="cuda:0")
t1 = torch.full((1,), 500, device="cuda:0").long()
sf1 = torch.randn(5, 1280, device="cuda:0")
y1 = torch.zeros(1, device="cuda:0").long()
txt1 = tok(["não"], padding="max_length", truncation=True, return_tensors="pt", max_length=40)
txt1 = {k: v.to("cuda:0") for k, v in txt1.items()}


def roda(n):
    with torch.no_grad():
        return unet(x1.repeat(n, 1, 1, 1), timesteps=t1.repeat(n),
                    context={k: v.repeat(n, 1) for k, v in txt1.items()},
                    y=y1.repeat(n), style_extractor=sf1.repeat(n, 1))


with torch.no_grad():
    roda(1)  # aquecimento
ref = roda(1)[0]

print("erro da linha 0 em relacao ao lote 1 (que bate com a CPU em 9.65e-07)\n")
print(f"{'lote':>6} {'erro relativo':>16}")
for n in (1, 2, 3, 4, 8, 16, 32):
    o = roda(n)[0]
    print(f"{n:>6} {float((ref - o).norm() / ref.norm()):>16.2e}")
