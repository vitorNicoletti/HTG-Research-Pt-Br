"""O teste para rodar em OUTRAS maquinas. Conta gradientes nao-finitos.

Por que este e o teste certo para comparar placas:

  - Nao precisa do BRESSAY nem de nenhum checkpoint nosso. Usa dados sinteticos
    e o checkpoint PUBLICO do DiffusionPen (huggingface.co/konnik/DiffusionPen),
    entao qualquer pessoa reproduz, inclusive uma banca.
  - Mede o caminho do BACKWARD, que e onde o dano real aconteceu: foi o
    gradiente nao-finito que envenenou o otimizador e inviabilizou os treinos.
  - E o fenomeno REPRODUTIVEL. Os testes de forward (teste_repeticao.py) so
    mostram o defeito as vezes, dependendo da sequencia de formatos chamada
    dentro do processo -- ver a secao "o que nao reproduz sob demanda" no
    README desta pasta.

Uso:
    CKPT=./DiffusionPen/diffusionpen_iam_model_path/models/ema_ckpt.pt \\
        python diagnostico/teste_gradiente_sintetico.py

    DEVICE=cpu PASSOS=10 ... python ...   # referencia (LENTO, minutos/passo)

Variaveis: PASSOS (padrao 60), LOTE (padrao 32), DEVICE (padrao cuda:0).

COMO LER

    0/60 nao-finitos        -> placa saudavel para esta carga
    qualquer valor > 0      -> a placa tem o mesmo problema desta maquina

Um gradiente `inf` ou `NaN` em fp32, partindo de um modelo treinado e com
entradas normalizadas, nao e comportamento esperado. Na maquina de
desenvolvimento (RX 6600 XT via ROCm com HSA_OVERRIDE_GFX_VERSION=10.3.0) este
teste da ~2%, e o treino real dava ~11% em todas as epocas.

Rode 2 ou 3 vezes: o resultado varia entre processos.
"""

import os
import sys
import argparse

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
    raise SystemExit(
        "defina CKPT com o checkpoint do UNet. O publico do DiffusionPen serve:\n"
        "  CKPT=./DiffusionPen/diffusionpen_iam_model_path/models/ema_ckpt.pt"
    )

DEVICE = os.environ.get("DEVICE", "cuda:0")
PASSOS = int(os.environ.get("PASSOS", "60"))
LOTE = int(os.environ.get("LOTE", "32"))


def main():
    a = argparse.Namespace(
        device=DEVICE, img_size=(64, 256), channels=4, emb_dim=320, num_heads=4,
        num_res_blocks=1, latent=True, img_feat=True, interpolation=False,
        mix_rate=None, model_name="diffusionpen", color=True, level="word",
        unet="unet_latent", batch_size=LOTE, save_path="",
    )
    na_gpu = DEVICE != "cpu"

    def embrulha(m):
        if na_gpu:
            return DataParallel(m, device_ids=[0])
        # Na CPU o DataParallel nao serve, mas o checkpoint espera o prefixo
        # "module." nas chaves; este wrapper reproduz so o prefixo.
        class Passa(nn.Module):
            def __init__(self, mod):
                super().__init__()
                self.module = mod

            def forward(self, *x, **k):
                return self.module(*x, **k)

        return Passa(m)

    tok = CanineTokenizer.from_pretrained("google/canine-c")
    te = embrulha(CanineModel.from_pretrained("google/canine-c")).to(DEVICE)
    te.requires_grad_(False)
    te.eval()

    unet = UNetModel(
        image_size=a.img_size, in_channels=4, model_channels=320, out_channels=4,
        num_res_blocks=1, attention_resolutions=(1, 1), channel_mult=(1, 1),
        num_heads=4, num_classes=339, context_dim=320, vocab_size=79,
        text_encoder=te, args=a,
    )
    unet = embrulha(unet).to(DEVICE)
    unet.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=True))
    unet.train()

    mse = nn.MSELoss()
    torch.manual_seed(0)
    txt = tok(["nao"] * LOTE, padding="max_length", truncation=True,
              return_tensors="pt", max_length=40)
    txt = {k: v.to(DEVICE) for k, v in txt.items()}

    print(f"dispositivo: {DEVICE} | lote: {LOTE} | passos: {PASSOS}")
    print(f"checkpoint: {CKPT}\n")

    ruins, normas = 0, []
    for i in range(PASSOS):
        x = torch.randn(LOTE, 4, 8, 32, device=DEVICE)
        alvo = torch.randn(LOTE, 4, 8, 32, device=DEVICE)
        t = torch.randint(0, 1000, (LOTE,), device=DEVICE).long()
        sf = torch.randn(LOTE * 5, 1280, device=DEVICE)
        y = torch.zeros(LOTE, device=DEVICE).long()

        unet.zero_grad(set_to_none=True)
        loss = mse(alvo, unet(x, timesteps=t, context=txt, y=y, style_extractor=sf))
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        if not torch.isfinite(gn):
            ruins += 1
            if ruins == 1:
                print(f"  primeiro gradiente nao-finito no passo {i}")
        else:
            normas.append(float(gn))

    unet.zero_grad(set_to_none=True)
    normas.sort()
    print(f"\n{ruins}/{PASSOS} gradientes nao-finitos ({100 * ruins / PASSOS:.0f}%)")
    if normas:
        print(f"|grad| dos validos: mediana {normas[len(normas) // 2]:.1f}, "
              f"max {normas[-1]:.1f}")
    print("\n0 nao-finitos = placa saudavel para esta carga")
    print("qualquer valor > 0 = mesmo problema da maquina de desenvolvimento")


if __name__ == "__main__":
    main()
