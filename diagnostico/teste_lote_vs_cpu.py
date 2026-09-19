"""A mesma entrada, sozinha e dentro de um lote, da o mesmo resultado?

Minimo possivel: UMA passada do UNet. Sem laco DDIM, sem VAE, sem dataset.
Se a linha 0 do lote 4 diferir do lote 1, o tamanho do lote corrompe o
resultado -- e ai toda amostra gerada com varios estilos de uma vez esta suja.
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
sf1 = torch.randn(5, 1280, device="cuda:0")          # 5 imagens de estilo, 1 escritor
y1 = torch.zeros(1, device="cuda:0").long()
txt1 = tok(["não"], padding="max_length", truncation=True, return_tensors="pt", max_length=40)
txt1 = {k: v.to("cuda:0") for k, v in txt1.items()}

N = 4
xN = x1.repeat(N, 1, 1, 1)
tN = t1.repeat(N)
sfN = sf1.repeat(N, 1)
yN = y1.repeat(N)
txtN = {k: v.repeat(N, 1) for k, v in txt1.items()}

with torch.no_grad():
    unet(x1, timesteps=t1, context=txt1, y=y1, style_extractor=sf1)   # aquecimento
    o1 = unet(x1, timesteps=t1, context=txt1, y=y1, style_extractor=sf1)
    oN = unet(xN, timesteps=tN, context=txtN, y=yN, style_extractor=sfN)

print(f"checkpoint: {CKPT}\n")
print(f"lote 1 -> saida {tuple(o1.shape)} | lote {N} -> saida {tuple(oN.shape)}\n")
for i in range(N):
    d = float((o1[0] - oN[i]).norm() / o1[0].norm())
    print(f"  linha {i} do lote {N} vs lote 1: erro relativo {d:.2e}")

print()
for i in range(1, N):
    d = float((oN[0] - oN[i]).norm() / oN[0].norm())
    print(f"  linha {i} vs linha 0, dentro do mesmo lote: {d:.2e}")

print("\nreferencia: ~1e-6 = ruido normal de fp32 | >=1e-2 = lote corrompe")

# ---- quem esta certo? a CPU nao usa MIOpen, entao ela e a referencia ----
class Passa(nn.Module):
    def __init__(self, m):
        super().__init__(); self.module = m
    def forward(self, *a, **k):
        return self.module(*a, **k)

te_c = Passa(CanineModel.from_pretrained("google/canine-c")).to("cpu")
te_c.requires_grad_(False); te_c.eval()
a.device = "cpu"
u_c = UNetModel(image_size=a.img_size, in_channels=4, model_channels=320, out_channels=4,
                num_res_blocks=1, attention_resolutions=(1, 1), channel_mult=(1, 1),
                num_heads=4, num_classes=339, context_dim=320, vocab_size=79,
                text_encoder=te_c, args=a)
u_c = Passa(u_c).to("cpu")
u_c.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True))
u_c.eval()

with torch.no_grad():
    c1 = u_c(x1.cpu(), timesteps=t1.cpu(), context={k: v.cpu() for k, v in txt1.items()},
             y=y1.cpu(), style_extractor=sf1.cpu())
    cN = u_c(xN.cpu(), timesteps=tN.cpu(), context={k: v.cpu() for k, v in txtN.items()},
             y=yN.cpu(), style_extractor=sfN.cpu())

r = lambda p, q: float((p - q).norm() / p.norm())
print("\n=== CPU como referencia ===")
print(f"  CPU lote 1 vs CPU lote 4 (linha 0): {r(c1[0], cN[0]):.2e}   <- a CPU depende do lote?")
print(f"  GPU lote 1 vs CPU lote 1:           {r(c1[0], o1[0].cpu()):.2e}")
print(f"  GPU lote 4 vs CPU lote 4 (linha 0): {r(cN[0], oN[0].cpu()):.2e}")
