"""O TESTE DEFINITIVO: gradiente CPU vs GPU com entradas bit a bit identicas.

Resultado nesta maquina (RX 6600 XT / ROCm), 5 passos, lote 32:

     passo    loss CPU    loss GPU   |grad| CPU   |grad| GPU    razao
         0    2.825638    2.825733        6.780       63.326     9.34x
         1    3.078844    3.078287        8.228       74.275     9.03x
         2    3.052705    3.053572        7.605       55.827     7.34x
         3    3.815079    3.813430       10.920       56.599     5.18x
         4    2.880469    2.879600        7.012       51.308     7.32x

O loss concorda na 4a casa -- o forward esta correto. O gradiente sai de 5 a 9
vezes maior na GPU, em todos os passos. O forward funciona como controle interno
da propria medida: se as entradas ou os pesos diferissem, o loss tambem diferiria.

Consequencia: o clip_grad_norm_ normaliza a MAGNITUDE mas nao corrige a DIRECAO,
entao o treino caminha para o lado errado enquanto o loss parece saudavel.

Uso (precisa do checkpoint publico do DiffusionPen e de uma GPU):
    python diagnostico/teste_gradiente_pareado.py

Leitura: razao ~1,00x em todos os passos = placa saudavel.

O teste anterior gerava as entradas com torch.randn(device=...), e CPU e GPU
tem geradores de numeros aleatorios distintos -- as entradas nao eram as mesmas.
Aqui tudo e gerado na CPU e copiado para a GPU.
"""
import os, sys, argparse, torch, torch.nn as nn
from torch.nn import DataParallel
from transformers import CanineModel, CanineTokenizer

sys.path.insert(0, "DiffusionPen")
from unet import UNetModel

CKPT = "./DiffusionPen/diffusionpen_iam_model_path/models/ema_ckpt.pt"
LOTE, PASSOS = 32, 5


class Passa(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.module = m

    def forward(self, *a, **k):
        return self.module(*a, **k)


def monta(dev):
    a = argparse.Namespace(
        device=dev, img_size=(64, 256), channels=4, emb_dim=320, num_heads=4,
        num_res_blocks=1, latent=True, img_feat=True, interpolation=False,
        mix_rate=None, model_name="diffusionpen", color=True, level="word",
        unet="unet_latent", batch_size=LOTE, save_path="",
    )
    emb = (lambda m: DataParallel(m, device_ids=[0])) if dev != "cpu" else Passa
    te = emb(CanineModel.from_pretrained("google/canine-c")).to(dev)
    te.requires_grad_(False)
    te.eval()
    u = UNetModel(
        image_size=a.img_size, in_channels=4, model_channels=320, out_channels=4,
        num_res_blocks=1, attention_resolutions=(1, 1), channel_mult=(1, 1),
        num_heads=4, num_classes=339, context_dim=320, vocab_size=79,
        text_encoder=te, args=a,
    )
    u = emb(u).to(dev)
    u.load_state_dict(torch.load(CKPT, map_location=dev, weights_only=True))
    u.train()
    return u


tok = CanineTokenizer.from_pretrained("google/canine-c")
txt_cpu = tok(["nao"] * LOTE, padding="max_length", truncation=True,
              return_tensors="pt", max_length=40)

# TODAS as entradas geradas na CPU, uma unica vez
torch.manual_seed(0)
lotes = []
for _ in range(PASSOS):
    lotes.append(dict(
        x=torch.randn(LOTE, 4, 8, 32),
        alvo=torch.randn(LOTE, 4, 8, 32),
        t=torch.randint(0, 1000, (LOTE,)).long(),
        sf=torch.randn(LOTE * 5, 1280),
        y=torch.zeros(LOTE).long(),
    ))

mse = nn.MSELoss()
resultados = {}
for dev in ("cpu", "cuda:0"):
    u = monta(dev)
    txt = {k: v.to(dev) for k, v in txt_cpu.items()}
    normas = []
    for b in lotes:
        u.zero_grad(set_to_none=True)
        loss = mse(b["alvo"].to(dev),
                   u(b["x"].to(dev), timesteps=b["t"].to(dev), context=txt,
                     y=b["y"].to(dev), style_extractor=b["sf"].to(dev)))
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(u.parameters(), 1.0)
        normas.append((float(loss.detach()), float(gn)))
    u.zero_grad(set_to_none=True)
    resultados[dev] = normas
    del u
    if dev != "cpu":
        torch.cuda.empty_cache()

print(f"\n{'passo':>6} {'loss CPU':>11} {'loss GPU':>11} {'|grad| CPU':>12} {'|grad| GPU':>12} {'razao':>8}")
for i, (c, g) in enumerate(zip(resultados["cpu"], resultados["cuda:0"])):
    razao = g[1] / c[1] if c[1] else float("nan")
    print(f"{i:>6} {c[0]:>11.6f} {g[0]:>11.6f} {c[1]:>12.3f} {g[1]:>12.3f} {razao:>8.2f}x")
print("\nloss igual + |grad| diferente = o forward concorda e o BACKWARD nao")
