"""O forward esta correto em TODOS os tamanhos de lote? CPU como referencia.

Por que este teste existe: o teste pareado comparava o LOSS, que e um escalar.
Um escalar pode esconder erros que se cancelam na media. Aqui comparamos o
TENSOR DE SAIDA INTEIRO do UNet, elemento a elemento.

Isso importa porque o backward reutiliza as ativacoes do forward. Se alguma
ativacao estiver errada de um jeito que o loss nao revela, o backward herda o
erro -- e ai nao daria para dizer "o defeito e do backward".

Cuidados que o teste toma:
  - entradas geradas UMA vez na CPU e copiadas para a GPU (torch.randn com
    device diferente usa geradores diferentes);
  - aquecimento de CADA formato antes de medir, porque a primeira chamada de
    um formato difere das seguintes;
  - eval() nos dois lados, senao o dropout do CANINE aninhado no UNet sorteia
    mascaras diferentes e a comparacao perde o sentido;
  - amostras DIFERENTES dentro do lote, nao replicadas.

    CKPT=./DiffusionPen/diffusionpen_iam_model_path/models/ema_ckpt.pt \\
        python diagnostico/teste_forward_completo.py

LEITURA
    ~1e-6 em todos os lotes  -> o forward esta correto; o defeito e do backward
    >=1e-3 em algum lote     -> o forward tambem esta contaminado
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.nn import DataParallel
from transformers import CanineModel, CanineTokenizer

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "DiffusionPen"))
os.chdir(RAIZ)
from unet import UNetModel  # noqa: E402

CKPT = os.environ.get("CKPT", "")
if not CKPT or not os.path.isfile(CKPT):
    raise SystemExit("defina CKPT com o caminho de um checkpoint .pt do UNet")

LOTES = tuple(int(x) for x in os.environ.get("LOTES", "1,2,4,8,16,32").split(","))
MAX = max(LOTES)


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
        unet="unet_latent", batch_size=MAX, save_path="",
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
    u.eval()
    return u


def main():
    tok = CanineTokenizer.from_pretrained("google/canine-c")

    # MAX amostras DIFERENTES, geradas uma unica vez na CPU
    torch.manual_seed(0)
    X = torch.randn(MAX, 4, 8, 32)
    T = torch.randint(0, 1000, (MAX,)).long()
    SF = torch.randn(MAX * 5, 1280)
    Y = torch.zeros(MAX).long()
    txt1 = tok(["nao"], padding="max_length", truncation=True,
               return_tensors="pt", max_length=40)

    saidas = {}
    for dev in ("cpu", "cuda:0"):
        u = monta(dev)
        por_lote = {}
        for n in LOTES:
            txt = {k: v.repeat(n, 1).to(dev) for k, v in txt1.items()}
            args_n = dict(timesteps=T[:n].to(dev), context=txt,
                          y=Y[:n].to(dev), style_extractor=SF[: n * 5].to(dev))
            with torch.no_grad():
                u(X[:n].to(dev), **args_n)            # aquece este formato
                o = u(X[:n].to(dev), **args_n)        # mede
            por_lote[n] = o.detach().float().cpu()
        saidas[dev] = por_lote
        del u
        if dev != "cpu":
            torch.cuda.empty_cache()

    print(f"\ncheckpoint: {CKPT}")
    print("tensor de saida INTEIRO do UNet, CPU vs GPU, por tamanho de lote\n")
    print(f"{'lote':>5} {'erro relativo':>15} {'maior dif abs':>15}  veredito")

    for n in LOTES:
        c, g = saidas["cpu"][n], saidas["cuda:0"][n]
        rel = float((c - g).norm() / c.norm())
        amax = float((c - g).abs().max())
        v = "ok" if rel < 1e-5 else ("suspeito" if rel < 1e-3 else "ERRADO")
        print(f"{n:>5} {rel:>15.2e} {amax:>15.2e}  {v}")

    print("\n~1e-6 em todos = forward correto, defeito isolado no backward")
    print(">=1e-3 em algum = o forward tambem esta contaminado")


if __name__ == "__main__":
    main()
