"""Repetibilidade com lote FIXO, separada do efeito do tamanho do lote.

O teste_lote_unet.py roda cada lote uma vez so, entao ele mistura duas
perguntas. Este script separa:

  (1) DISPERSAO INTRA-LOTE: R chamadas identicas, mesmo lote, comparadas
      entre si. Mede so a irreprodutibilidade da placa.
  (2) EFEITO DO LOTE: a mediana de cada lote comparada com a mediana do
      lote 1. So faz sentido se (1) for pequeno.

Uso:
    CKPT=./meu_modelo/models/ema_ckpt.pt python teste_repeticao.py

Leitura:
  - intra-lote 0.00e+00  -> determinismo perfeito nesse caminho
  - intra-lote ~1e-7     -> ruido de fp32, saudavel
  - intra-lote >= 1e-3   -> IRREPRODUTIBILIDADE. e o defeito.
"""

import os, sys, argparse, itertools, torch
from torch.nn import DataParallel
from transformers import CanineModel, CanineTokenizer

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "DiffusionPen"))
os.chdir(RAIZ)
from unet import UNetModel

CKPT = os.environ.get("CKPT", "")
if not CKPT or not os.path.isfile(CKPT):
    raise SystemExit("defina CKPT com o caminho do checkpoint .pt do UNet")

REPS = int(os.environ.get("REPS", "5"))
LOTES = (1, 2, 4, 16)

a = argparse.Namespace(
    device="cuda:0",
    img_size=(64, 256),
    channels=4,
    emb_dim=320,
    num_heads=4,
    num_res_blocks=1,
    latent=True,
    img_feat=True,
    interpolation=False,
    mix_rate=None,
    model_name="diffusionpen",
    color=True,
    level="word",
    unet="unet_latent",
    batch_size=1,
    save_path="",
)

tok = CanineTokenizer.from_pretrained("google/canine-c")
te = DataParallel(CanineModel.from_pretrained("google/canine-c"), device_ids=[0]).to(
    "cuda:0"
)
te.requires_grad_(False)
te.eval()

unet = UNetModel(
    image_size=a.img_size,
    in_channels=4,
    model_channels=320,
    out_channels=4,
    num_res_blocks=1,
    attention_resolutions=(1, 1),
    channel_mult=(1, 1),
    num_heads=4,
    num_classes=339,
    context_dim=320,
    vocab_size=79,
    text_encoder=te,
    args=a,
)
unet = DataParallel(unet, device_ids=[0]).to("cuda:0")
unet.load_state_dict(torch.load(CKPT, map_location="cuda:0", weights_only=True))
unet.eval()

torch.manual_seed(0)
x1 = torch.randn(1, 4, 8, 32, device="cuda:0")
t1 = torch.full((1,), 500, device="cuda:0").long()
sf1 = torch.randn(5, 1280, device="cuda:0")
y1 = torch.zeros(1, device="cuda:0").long()
txt1 = tok(
    ["nao"], padding="max_length", truncation=True, return_tensors="pt", max_length=40
)
txt1 = {k: v.to("cuda:0") for k, v in txt1.items()}


def roda(n):
    """Uma passada; devolve a LINHA 0 do lote, na CPU, para comparacao."""
    with torch.no_grad():
        o = unet(
            x1.repeat(n, 1, 1, 1),
            timesteps=t1.repeat(n),
            context={k: v.repeat(n, 1) for k, v in txt1.items()},
            y=y1.repeat(n),
            style_extractor=sf1.repeat(n, 1),
        )
    return o[0].detach().float().cpu()


def rel(p, q):
    return float((p - q).norm() / p.norm())


roda(1)  # aquecimento: absorve a autotunagem de kernels da 1a chamada

print(f"REPS={REPS} chamadas identicas por tamanho de lote\n")
print(
    f"{'lote':>5} {'1a vs resto':>13} {'intra (max)':>13} {'intra (min)':>13} "
    f"{'bit-identicos':>14}"
)

medianas = {}
for n in LOTES:
    saidas = [roda(n) for _ in range(REPS)]
    pares = list(itertools.combinations(range(REPS), 2))
    # dispersao ignorando a 1a chamada, para isolar o efeito de "primeira chamada"
    pares_sem1 = [(i, j) for i, j in pares if i != 0]
    erros = [rel(saidas[i], saidas[j]) for i, j in pares_sem1] or [0.0]
    prim = max(rel(saidas[0], saidas[j]) for j in range(1, REPS))
    identicos = sum(torch.equal(saidas[i], saidas[j]) for i, j in pares)
    print(
        f"{n:>5} {prim:>13.2e} {max(erros):>13.2e} {min(erros):>13.2e} "
        f"{identicos:>7}/{len(pares):<6}"
    )
    # mediana elemento a elemento das repeticoes 2..R, como representante do lote
    medianas[n] = torch.stack(saidas[1:]).median(dim=0).values

print("\nefeito do tamanho do lote (mediana do lote N vs mediana do lote 1)")
print("SO E INTERPRETAVEL se a coluna 'intra (max)' acima for ~1e-7 ou menor\n")
for n in LOTES:
    print(f"  lote {n:>2}: {rel(medianas[1], medianas[n]):.2e}")

print("\nintra-lote 0.00e+00 = deterministico | ~1e-7 = ok | >=1e-3 = defeito")
