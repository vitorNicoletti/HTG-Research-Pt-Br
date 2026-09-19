"""A acumulacao de gradiente salva o treino nesta placa?

A pergunta pratica: se o defeito aparece em lotes grandes, da para treinar com
lote 1 acumulando gradiente ao longo de N amostras? Matematicamente e
equivalente a um lote N, e so aciona os formatos pequenos.

Antes de reescrever o treino, isso PRECISA ser medido -- o backward e codigo
diferente do forward, com kernels proprios. O gradiente dos pesos de uma
convolucao e uma reducao sobre a dimensao do lote, exatamente o tipo de
operacao em que o tamanho do lote muda a estrategia de soma.

O que o teste faz, com N amostras DIFERENTES (nao replicadas):

  A) um unico backward com lote N
  B) N backwards com lote 1, somando os gradientes (cada um escalado por 1/N,
     porque o MSELoss ja faz media sobre o lote)
  C) o mesmo que (A), mas na CPU -- a referencia

Compara A vs B na GPU, e ambos contra C.

    CKPT=./DiffusionPen/diffusionpen_iam_model_path/models/ema_ckpt.pt \\
        python diagnostico/teste_acumulacao_gradiente.py

LEITURA

  A vs B ~1e-6  -> acumular gradiente da o MESMO resultado que o lote grande.
                   Se alem disso ambos baterem com a CPU, a placa serve para
                   treinar por acumulacao.
  A vs B >=1e-2 -> os dois caminhos divergem. Ai o que importa e qual deles
                   bate com a CPU: esse e o caminho utilizavel.
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

N = int(os.environ.get("N", "4"))


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
        unet="unet_latent", batch_size=N, save_path="",
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
    # eval() e OBRIGATORIO: o CANINE fica aninhado dentro do UNet e tem dropout
    # 0.1. Em train(), lote 1 e lote N sorteiam mascaras diferentes e a
    # comparacao vira lixo -- a linha de sanidade na CPU acusa isso na hora.
    # O backward continua funcionando normalmente em eval.
    u.eval()
    return u


def colhe(u):
    return {n: p.grad.detach().float().cpu().clone()
            for n, p in u.named_parameters() if p.grad is not None}


def rel(ga, gb):
    num = den = 0.0
    for k, v in ga.items():
        if k in gb:
            num += float((v - gb[k]).pow(2).sum())
            den += float(v.pow(2).sum())
    return (num / den) ** 0.5 if den else float("nan")


def main():
    mse = nn.MSELoss()
    tok = CanineTokenizer.from_pretrained("google/canine-c")

    # N amostras DIFERENTES, geradas uma unica vez na CPU
    torch.manual_seed(0)
    amostras = [dict(
        x=torch.randn(1, 4, 8, 32), alvo=torch.randn(1, 4, 8, 32),
        t=torch.randint(0, 1000, (1,)).long(), sf=torch.randn(5, 1280),
        y=torch.zeros(1).long(),
    ) for _ in range(N)]
    txt1 = tok(["nao"], padding="max_length", truncation=True,
               return_tensors="pt", max_length=40)

    def junta(chave):
        return torch.cat([a[chave] for a in amostras], dim=0)

    resultados = {}
    for dev in ("cuda:0", "cpu"):
        u = monta(dev)
        txtN = {k: v.repeat(N, 1).to(dev) for k, v in txt1.items()}
        txt1d = {k: v.to(dev) for k, v in txt1.items()}

        # aquecimento dos dois formatos
        for n, tt in ((N, txtN), (1, txt1d)):
            u.zero_grad(set_to_none=True)
            xb = junta("x")[:n].to(dev)
            mse(junta("alvo")[:n].to(dev),
                u(xb, timesteps=junta("t")[:n].to(dev), context=tt,
                  y=junta("y")[:n].to(dev),
                  style_extractor=junta("sf")[: n * 5].to(dev))).backward()
        u.zero_grad(set_to_none=True)

        # (A) um backward com lote N
        loss = mse(junta("alvo").to(dev),
                   u(junta("x").to(dev), timesteps=junta("t").to(dev), context=txtN,
                     y=junta("y").to(dev), style_extractor=junta("sf").to(dev)))
        loss.backward()
        g_lote = colhe(u)
        u.zero_grad(set_to_none=True)

        # (B) N backwards com lote 1, acumulando
        for a in amostras:
            l1 = mse(a["alvo"].to(dev),
                     u(a["x"].to(dev), timesteps=a["t"].to(dev), context=txt1d,
                       y=a["y"].to(dev), style_extractor=a["sf"].to(dev)))
            (l1 / N).backward()
        g_acum = colhe(u)
        u.zero_grad(set_to_none=True)

        resultados[dev] = (g_lote, g_acum, float(loss.detach()))
        del u
        if dev != "cpu":
            torch.cuda.empty_cache()

    g_gpu_lote, g_gpu_acum, loss_gpu = resultados["cuda:0"]
    g_cpu_lote, g_cpu_acum, loss_cpu = resultados["cpu"]

    def norma(g):
        return sum(float(v.pow(2).sum()) for v in g.values()) ** 0.5

    print(f"\nN = {N} amostras diferentes | checkpoint: {CKPT}\n")
    print(f"loss do lote {N}:  CPU {loss_cpu:.6f} | GPU {loss_gpu:.6f} "
          f"(dif {abs(loss_cpu - loss_gpu):.2e})\n")
    print(f"{'comparacao':<44} {'erro relativo':>14}")
    print(f"{'-' * 60}")
    print(f"{'CPU: lote N vs acumulado (sanidade do metodo)':<44} "
          f"{rel(g_cpu_lote, g_cpu_acum):>14.2e}")
    print(f"{f'GPU: lote {N} vs acumulado em lotes 1':<44} "
          f"{rel(g_gpu_lote, g_gpu_acum):>14.2e}")
    print(f"{f'GPU lote {N}      vs CPU lote {N}':<44} "
          f"{rel(g_cpu_lote, g_gpu_lote):>14.2e}")
    print(f"{'GPU acumulado  vs CPU acumulado':<44} "
          f"{rel(g_cpu_acum, g_gpu_acum):>14.2e}")
    print(f"\nnormas: CPU lote {norma(g_cpu_lote):.3f} | CPU acum {norma(g_cpu_acum):.3f} "
          f"| GPU lote {norma(g_gpu_lote):.3f} | GPU acum {norma(g_gpu_acum):.3f}")
    print("\nA linha que decide e 'GPU acumulado vs CPU acumulado': se der ~1e-6,")
    print("treinar por acumulacao de gradiente nesta placa e viavel.")


if __name__ == "__main__":
    main()
