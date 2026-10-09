"""O VAE do Stable Diffusion preserva o diacritico em 64x256? E em 32x128?

O DiffusionPen nao gera pixels: gera o latente do VAE do SD-1.5, que reduz a
imagem 8x em cada eixo (64x256 -> latente 8x32; 32x128 -> 4x16). Se o VAE
sozinho, ida e volta, apaga o acento numa resolucao, nenhum treino do UNet
recupera esse acento nela. Sem treinar nada, este teste codifica e decodifica
palavras acentuadas nas duas resolucoes e monta uma folha:

    original 64x256 | VAE 64x256 | original 32x128 | VAE 32x128

As de 32x128 sao mostradas ampliadas 2x (vizinho mais proximo) para caberem
na mesma coluna. Fontes: BRESSAY (pre-processamento v2), RIMES (palavras
cursivas, alta resolucao; opcional) e IAM (controle, sem acento).

    python diagnostico/teste_vae_resolucao.py --rimes rimes/imagettes_mots_cursif.zip
"""

import argparse
import io
import os
import random
import subprocess
import sys
import zipfile
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "DiffusionPen"))
from utils import bressay_dataset as bd  # noqa: E402

DIAC = set("àáâãçéêíóôõúÀÁÂÃÇÉÊÍÓÔÕÚ")


def prep_iam(im):
    """Altura 64 preservando o aspecto, cabendo em 256 -- como o IAM."""
    return ImageOps.pad(im.convert("RGB"), size=(256, 64), color="white")


def ida_e_volta(vae, im, device):
    x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 127.5 - 1.0)
    x = x.permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        lat = vae.encode(x).latent_dist.mean
        y = vae.decode(lat).sample
    y = ((y[0].permute(1, 2, 0).clamp(-1, 1) + 1) * 127.5).cpu().numpy().astype(np.uint8)
    return Image.fromarray(y), tuple(lat.shape[1:])


def fonte(t):
    try:
        cam = subprocess.run(["fc-match", "-f", "%{file}", "sans:lang=pt"],
                             capture_output=True, text=True).stdout.strip()
        return ImageFont.truetype(cam, t)
    except Exception:
        return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="./bressay_split_25_v2")
    ap.add_argument("--rimes", help="imagettes_mots_cursif.zip do RIMES (opcional)")
    ap.add_argument("--n", type=int, default=6, help="palavras por fonte")
    ap.add_argument("--sd", default=os.environ.get("SD", "stable-diffusion-v1-5/stable-diffusion-v1-5"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--saida", default="./saidas/diffusionpen/vae_resolucao.png")
    a = ap.parse_args()
    os.chdir(RAIZ)
    rnd = random.Random(3)

    amostras = []   # (fonte, palavra, imagem 64x256 RGB)
    ds = bd.BRESSAY_Dataset(a.split, "train", args=SimpleNamespace(max_samples=0, preproc="v2"))
    alt = {}
    cache = os.path.join(os.path.dirname(a.split.rstrip("/")), "bressay_split_25", "alturas_tinta.tsv")
    if os.path.isfile(cache):
        alt = {l.split("\t")[0]: float(l.split("\t")[1]) for l in open(cache, encoding="utf-8")}
    acentuadas = [d for d in ds.data if any(c in DIAC for c in d[1]) and len(d[1]) > 3]
    # melhor caso do BRESSAY: as palavras de tinta mais alta
    acentuadas.sort(key=lambda d: -alt.get(os.path.relpath(d[0], ds.images_root), 0))
    for cam, t, _ in rnd.sample(acentuadas[:300], a.n):
        amostras.append(("BRESSAY", t, ds.load_image(cam)))

    if a.rimes:
        z = zipfile.ZipFile(a.rimes)
        dat = z.read("imagettes_mots_cursif/goodSnippets_total/goodSnippets_total.dat").decode("utf-8-sig")
        nomes = set(z.namelist())
        cand = []
        for l in dat.splitlines():
            caminho, palavra, _ = l.rstrip().rsplit(" ", 2)
            if any(c in "éêàçô" for c in palavra) and len(palavra) > 3:
                partes = caminho.replace("\\", "/").split("/")
                lote = partes[-3].replace("lot_", "")
                n = f"imagettes_mots_cursif/lot_{lote}_rimes_version_definitive/{partes[-2]}/{partes[-1]}"
                if n in nomes:
                    cand.append((n, palavra))
        for n, palavra in rnd.sample(cand, a.n):
            amostras.append(("RIMES", palavra, prep_iam(Image.open(io.BytesIO(z.read(n))))))

    iam = os.path.join(RAIZ, "DiffusionPen", "iam_data", "words")
    if os.path.isdir(iam):
        import glob
        pngs = glob.glob(os.path.join(iam, "*", "*", "*.png"))
        for p in rnd.sample(pngs, a.n // 2):
            amostras.append(("IAM", "(controle)", prep_iam(Image.open(p))))

    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(a.sd, subfolder="vae").to(a.device).eval()

    f_rot = fonte(15)
    LX, CW, CH = 190, 256, 64
    cols = ["original 64x256", "VAE 64x256 (latente 8x32)", "original 32x128", "VAE 32x128 (latente 4x16)"]
    folha = Image.new("RGB", (LX + 4 * (CW + 10), 40 + len(amostras) * (CH + 8)), "white")
    d = ImageDraw.Draw(folha)
    for k, c in enumerate(cols):
        d.text((LX + k * (CW + 10), 12), c, fill=(140, 0, 0), font=f_rot)
    y = 40
    for fonte_, palavra, im64 in amostras:
        rec64, lat64 = ida_e_volta(vae, im64, a.device)
        im32 = im64.resize((128, 32), Image.BICUBIC)
        rec32, lat32 = ida_e_volta(vae, im32, a.device)
        d.text((8, y + 10), fonte_, fill=(100, 100, 100), font=f_rot)
        d.text((8, y + 32), palavra, fill="black", font=f_rot)
        for k, im in enumerate((im64, rec64, im32.resize((256, 64), Image.NEAREST),
                                rec32.resize((256, 64), Image.NEAREST))):
            x = LX + k * (CW + 10)
            folha.paste(im, (x, y))
            d.rectangle([x - 1, y - 1, x + CW, y + CH], outline=(180, 180, 180))
        y += CH + 8
    os.makedirs(os.path.dirname(a.saida), exist_ok=True)
    folha.save(a.saida)
    print(f"latentes: 64x256 -> {lat64}, 32x128 -> {lat32}")
    print("folha:", a.saida)


if __name__ == "__main__":
    main()
