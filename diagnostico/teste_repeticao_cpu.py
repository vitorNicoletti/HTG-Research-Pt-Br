"""Controle na CPU do teste_repeticao.py -- a medida que sustenta a conclusao.

Mesmo modelo, mesmas entradas, mesma logica do teste_repeticao.py, mas na CPU,
que nao usa MIOpen. Serve para responder a pergunta que decide tudo:

    o lote 4 dar diferente do lote 1 e DEFEITO DA PLACA,
    ou e uma propriedade do modelo?

Se a CPU der ~1e-7 entre lote 1 e lote 4, a resposta e "defeito da placa", e a
discussao acaba. Se a CPU tambem variar, o efeito e do modelo e a conclusao cai.

Uso:
    CKPT=./meu_modelo/models/ema_ckpt.pt python teste_repeticao_cpu.py

E LENTO: cada passada na CPU leva minutos. Por isso REPS=2 e so os lotes 1, 2
e 4 -- o suficiente para a comparacao, ja que o lote 2 e o controle que descarta
erro no proprio harness de replicacao.
"""

import os, sys, argparse, itertools, time

import torch
import torch.nn as nn
from transformers import CanineModel, CanineTokenizer

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "DiffusionPen"))
os.chdir(RAIZ)
from unet import UNetModel

CKPT = os.environ.get("CKPT", "")
if not CKPT or not os.path.isfile(CKPT):
    raise SystemExit("defina CKPT com o caminho do checkpoint .pt do UNet")

REPS = int(os.environ.get("REPS", "2"))
LOTES = tuple(int(x) for x in os.environ.get("LOTES", "1,2,4").split(","))


class Passa(nn.Module):
    """Reproduz o prefixo 'module.' do DataParallel sem envolver CUDA.

    DataParallel(device_ids=[]) nao serve: estoura no construtor quando ha CUDA
    na maquina. E o checkpoint foi salvo com esse prefixo, entao precisamos dele.
    """

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *a, **k):
        return self.module(*a, **k)


a = argparse.Namespace(
    device="cpu", img_size=(64, 256), channels=4, emb_dim=320, num_heads=4,
    num_res_blocks=1, latent=True, img_feat=True, interpolation=False,
    mix_rate=None, model_name="diffusionpen", color=True, level="word",
    unet="unet_latent", batch_size=1, save_path="",
)

tok = CanineTokenizer.from_pretrained("google/canine-c")
te = Passa(CanineModel.from_pretrained("google/canine-c")).to("cpu")
te.requires_grad_(False)
te.eval()

unet = UNetModel(
    image_size=a.img_size, in_channels=4, model_channels=320, out_channels=4,
    num_res_blocks=1, attention_resolutions=(1, 1), channel_mult=(1, 1),
    num_heads=4, num_classes=339, context_dim=320, vocab_size=79,
    text_encoder=te, args=a,
)
unet = Passa(unet).to("cpu")
unet.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True))
unet.eval()

# Mesmas entradas do teste da GPU: mesma semente, mesma construcao.
torch.manual_seed(0)
x1 = torch.randn(1, 4, 8, 32)
t1 = torch.full((1,), 500).long()
sf1 = torch.randn(5, 1280)
y1 = torch.zeros(1).long()
txt1 = tok(["nao"], padding="max_length", truncation=True,
           return_tensors="pt", max_length=40)


def roda(n):
    with torch.no_grad():
        o = unet(
            x1.repeat(n, 1, 1, 1),
            timesteps=t1.repeat(n),
            context={k: v.repeat(n, 1) for k, v in txt1.items()},
            y=y1.repeat(n),
            style_extractor=sf1.repeat(n, 1),
        )
    return o[0].detach().float().clone()


def rel(p, q):
    return float((p - q).norm() / p.norm())


print(f"CPU | REPS={REPS} | lotes {LOTES}\n")
print(f"{'lote':>5} {'intra (max)':>13} {'bit-identicos':>14} {'tempo':>9}")

medianas = {}
for n in LOTES:
    t0 = time.time()
    saidas = [roda(n) for _ in range(REPS)]
    pares = list(itertools.combinations(range(REPS), 2))
    erros = [rel(saidas[i], saidas[j]) for i, j in pares] or [0.0]
    identicos = sum(torch.equal(saidas[i], saidas[j]) for i, j in pares)
    print(f"{n:>5} {max(erros):>13.2e} {identicos:>7}/{len(pares):<6} "
          f"{time.time()-t0:>8.0f}s")
    medianas[n] = torch.stack(saidas).median(dim=0).values

print("\nefeito do tamanho do lote NA CPU (mediana do lote N vs mediana do lote 1)\n")
for n in LOTES:
    print(f"  lote {n:>2}: {rel(medianas[1], medianas[n]):.2e}")

print("\n~1e-7 em todos os lotes = a CPU independe do lote, como deve ser.")
print("Nesse caso, a variacao por lote observada na GPU e defeito da placa.")
