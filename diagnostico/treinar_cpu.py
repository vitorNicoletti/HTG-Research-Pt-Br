"""Fine-tune na CPU -- referencia de hardware saudavel.

NAO substitui o train.py: e um script separado, com a mesma matematica de
treino, feito para rodar sem GPU. Existe porque o backward da placa desta
maquina sai de 5 a 9 vezes maior que o da CPU (ver diagnostico/), e precisamos
de uma corrida em hardware confiavel para comparar.

E lento, mas viavel: ~2,3 s por passo de forward+backward com lote 32 no UNet.
Com --max_samples 20000 sao 625 passos por epoca.

    python scripts/treinar_cpu.py --epochs 1 --max_samples 20000

O que ele registra, que e o que interessa para a comparacao:
  - quantos batches deram gradiente nao-finito (na GPU eram ~11%; aqui deve
    ser ZERO)
  - a norma do gradiente antes do clip (na GPU a mediana era 57; na CPU, ~7)
  - o MSE por epoca

Salva em <save_path>/models/ nos mesmos nomes do train.py, entao o
scripts/gerar_amostras.py le o resultado sem adaptacao.
"""

import argparse
import os
import statistics
import sys
import time

import torch
import torch.nn as nn
import torchvision
from diffusers import AutoencoderKL, DDIMScheduler
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import CanineModel, CanineTokenizer

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "DiffusionPen"))
os.chdir(RAIZ)

from feature_extractor import ImageEncoder  # noqa: E402
from unet import UNetModel  # noqa: E402
from utils.bressay_dataset import BRESSAY_Dataset  # noqa: E402

DEV = "cpu"


class Passa(nn.Module):
    """Reproduz o prefixo 'module.' do DataParallel sem envolver CUDA.

    Os checkpoints foram salvos a partir de modelos embrulhados em
    DataParallel, entao as chaves tem esse prefixo. Na CPU o DataParallel nao
    serve, mas o prefixo continua sendo necessario para o load_state_dict.
    """

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *a, **k):
        return self.module(*a, **k)


def cli():
    p = argparse.ArgumentParser(description="Fine-tune do DiffusionPen na CPU.")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max_samples", type=int, default=20000,
                   help="amostras por epoca (0 = split inteiro, 74.882)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--threads", type=int, default=12,
                   help="threads do torch; deixe nucleos livres para os workers")
    p.add_argument("--save_path", type=str, default="./model_cpu_ref")
    p.add_argument("--pretrained_path", type=str,
                   default="./DiffusionPen/diffusionpen_iam_model_path/models")
    p.add_argument("--style_path", type=str,
                   default="./DiffusionPen/style_models/iam_style_diffusionpen.pth",
                   help="por padrao o extrator dos autores, que nao passou por esta placa")
    p.add_argument("--stable_dif_path", type=str, default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--dataset_folder", type=str, default="./bressay_split")
    return p.parse_args()


def main():
    cfg = cli()
    torch.set_num_threads(cfg.threads)
    os.makedirs(os.path.join(cfg.save_path, "models"), exist_ok=True)

    args = argparse.Namespace(
        device=DEV, img_size=(64, 256), channels=4, emb_dim=320, num_heads=4,
        num_res_blocks=1, latent=True, img_feat=True, interpolation=False,
        mix_rate=None, model_name="diffusionpen", color=True, level="word",
        unet="unet_latent", batch_size=cfg.batch_size, save_path=cfg.save_path,
        max_samples=cfg.max_samples,
    )

    print(f"CPU | threads {cfg.threads} | workers {cfg.num_workers} | "
          f"lote {cfg.batch_size} | lr {cfg.lr}")

    tf = transforms.Compose([
        transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])
    ds = BRESSAY_Dataset(cfg.dataset_folder, "train", transforms=tf, args=args)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                    num_workers=cfg.num_workers)

    tok = CanineTokenizer.from_pretrained("google/canine-c")
    te = Passa(CanineModel.from_pretrained("google/canine-c")).to(DEV)
    te.requires_grad_(False)
    te.eval()

    unet = UNetModel(
        image_size=args.img_size, in_channels=4, model_channels=320, out_channels=4,
        num_res_blocks=1, attention_resolutions=(1, 1), channel_mult=(1, 1),
        num_heads=4, num_classes=339, context_dim=320, vocab_size=79,
        text_encoder=te, args=args,
    )
    unet = Passa(unet).to(DEV)

    # Mesma logica de carregamento do train.py: copia so as chaves cujo shape bate.
    for mdl, nome in [(unet, "ckpt.pt")]:
        sd = torch.load(os.path.join(cfg.pretrained_path, nome),
                        map_location=DEV, weights_only=True)
        md = mdl.state_dict()
        ok = {k: v for k, v in sd.items() if k in md and md[k].shape == v.shape}
        md.update(ok)
        mdl.load_state_dict(md)
        print(f"{nome}: {len(ok)}/{len(md)} chaves carregadas")
    unet.train()

    vae = AutoencoderKL.from_pretrained(cfg.stable_dif_path, subfolder="vae").to(DEV)
    vae.requires_grad_(False)
    vae.eval()

    fe = ImageEncoder(model_name="mobilenetv2_100", num_classes=0,
                      pretrained=True, trainable=True)
    md = fe.state_dict()
    sd = torch.load(cfg.style_path, map_location=DEV, weights_only=True)
    md.update({k: v for k, v in sd.items() if k in md and md[k].shape == v.shape})
    fe.load_state_dict(md)
    fe = fe.to(DEV)
    fe.requires_grad_(False)
    fe.eval()

    sched = DDIMScheduler.from_pretrained(cfg.stable_dif_path, subfolder="scheduler")
    opt = optim.AdamW(unet.parameters(), lr=cfg.lr, eps=1e-6)
    mse = nn.MSELoss()

    passos_epoca = len(ds) // cfg.batch_size
    print(f"{len(ds)} amostras | {passos_epoca} passos por epoca\n")

    for ep in range(cfg.epochs):
        soma, n_ok, ruins, normas = 0.0, 0, 0, []
        t_ini = time.time()

        for i, data in enumerate(dl):
            imgs, transcr, s_id, estilos = data[0], data[1], data[2], data[3]

            txt = tok(list(transcr), padding="max_length", truncation=True,
                      return_tensors="pt", max_length=40)
            with torch.no_grad():
                sf = fe(estilos.reshape(-1, 3, 64, 256))
                lat = vae.encode(imgs.to(torch.float32)).latent_dist.sample() * 0.18215

            ruido = torch.randn(lat.shape)
            t = torch.randint(0, 1000, (lat.shape[0],)).long()
            x_t = sched.add_noise(lat, ruido, t)

            opt.zero_grad(set_to_none=True)
            pred = unet(x_t, timesteps=t, context=txt, y=s_id, style_extractor=sf)
            loss = mse(ruido, pred)

            if not torch.isfinite(loss):
                ruins += 1
                continue

            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            if not torch.isfinite(gn):
                ruins += 1
                continue

            normas.append(float(gn))
            opt.step()
            soma += float(loss.detach())
            n_ok += 1

            if i % 25 == 0:
                decorrido = time.time() - t_ini
                seg_passo = decorrido / (i + 1)
                falta = (passos_epoca - i - 1) * seg_passo / 60
                print(f"  ep{ep} passo {i}/{passos_epoca} | MSE {soma / max(n_ok, 1):.4f} "
                      f"| |grad| {normas[-1] if normas else float('nan'):.1f} "
                      f"| descartados {ruins} | {seg_passo:.1f}s/passo "
                      f"| faltam ~{falta:.0f} min", flush=True)

        normas.sort()
        med = normas[len(normas) // 2] if normas else float("nan")
        print(f"\nepoca {ep}: MSE {soma / max(n_ok, 1):.4f} | "
              f"{ruins} batches descartados de {passos_epoca} | "
              f"|grad| mediana {med:.1f} max {normas[-1] if normas else float('nan'):.1f} | "
              f"{(time.time() - t_ini) / 60:.0f} min\n", flush=True)

        torch.save(unet.state_dict(), os.path.join(cfg.save_path, "models", "ckpt.pt"))
        # O gerar_amostras.py le ema_ckpt.pt por padrao; sem EMA aqui, gravamos
        # o proprio modelo com esse nome para nao precisar de adaptacao.
        torch.save(unet.state_dict(), os.path.join(cfg.save_path, "models", "ema_ckpt.pt"))
        torch.save({"epoch": ep, "ema_step": 0},
                   os.path.join(cfg.save_path, "models", "estado.pt"))
        print(f"checkpoint salvo em {cfg.save_path}/models/\n", flush=True)


if __name__ == "__main__":
    main()
