"""O que o modelo REALMENTE recebe como alvo de treino?

Por que este teste existe: toda a investigacao ate aqui mexeu em lr, passos e
hardware, e nunca olhou a entrada. Se a imagem-alvo que o dataloader entrega
estiver deformada, nenhum lr conserta -- o modelo vai aprender a deformacao, e
quanto mais passos, mais ele aprende. Isso explicaria por que o resultado piora
com o numero de passos e nao com o tamanho do lr.

O teste nao carrega modelo, nao usa GPU e roda em segundos.

O que ele compara, na MESMA imagem crua:
  (A) o pre-processamento do BRESSAY (utils/bressay_dataset.py::load_image)
  (B) o pre-processamento do IAM (utils/iam_dataset.py), que e a distribuicao
      em que o checkpoint pre-treinado foi treinado

A diferenca entre A e B e a mudanca de distribuicao que o fine-tune persegue.

    BRESSAY_IMAGES=./bressay/data/words \\
        python diagnostico/inspecionar_dados.py --split bressay_split/train.tsv

LEITURA
    distorcao de aspecto ~1.00     -> a letra chega com a proporcao correta
    distorcao longe de 1.00        -> a letra chega esticada/achatada; o modelo
                                      esta sendo treinado para desenhar isso
"""

import argparse
import os
import random
import sys

import numpy as np
from PIL import Image, ImageOps

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def prep_bressay(im):
    """Copia fiel de bressay_dataset.py::load_image. Devolve (img, ramo)."""
    P_TINTA, P_FUNDO = 3, 40
    g = np.asarray(im.convert("L"), dtype=np.float32)
    lo = np.percentile(g, P_TINTA)
    hi = np.percentile(g, P_FUNDO)
    if hi - lo >= 8:
        g = np.clip((g - lo) / (hi - lo), 0, 1) * 255
        im = Image.fromarray(g.astype(np.uint8)).convert("RGB")
    # NOTA: a linha que normaliza a altura para 64 esta COMENTADA no arquivo
    # original. O ramo abaixo decide com a largura NATIVA, nao com a largura
    # apos a altura virar 64.
    if im.size[0] < 256:
        return ImageOps.pad(im, size=(256, 64), color="white"), "pad"
    return im.resize((256, 64), Image.BICUBIC), "squash"


def prep_iam(im):
    """Copia fiel de iam_dataset.py::main_loader -- preserva aspecto sempre."""
    w, h = im.size
    im = im.resize((max(1, int(w * 64 / h)), 64))
    w, h = im.size
    if w < 256:
        return ImageOps.pad(im, size=(256, 64), color="white"), "pad"
    while w > 256:
        w = w - 20
        im = im.resize((w, max(1, int(im.size[1] * w / im.size[0]))))
        w, h = im.size
    fundo = Image.new("RGB", (256, 64), (255, 255, 255))
    fundo.paste(im, ((256 - w) // 2, (64 - h) // 2))
    return fundo, "centered"


def folha(imagens, caminho, cols=4):
    if not imagens:
        return
    linhas = (len(imagens) + cols - 1) // cols
    fl = Image.new("RGB", (cols * 258, linhas * 66), (128, 128, 128))
    for i, im in enumerate(imagens):
        fl.paste(im, ((i % cols) * 258 + 1, (i // cols) * 66 + 1))
    fl.save(caminho)
    print(f"  escrito: {caminho}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="bressay_split/train.tsv")
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--out", default="diagnostico/resultados/dados")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    os.chdir(RAIZ)
    raiz_img = os.environ.get("BRESSAY_IMAGES", "./bressay/data/words")
    os.makedirs(a.out, exist_ok=True)

    linhas = [l for l in open(a.split, encoding="utf-8").read().strip().split("\n")
              if len(l.split("\t")) >= 3]
    print(f"split: {a.split} -- {len(linhas)} linhas")

    # 1) varredura larga: so le o cabecalho de cada PNG, e barato
    random.seed(a.seed)
    amostra = random.sample(linhas, min(400, len(linhas)))
    ramos, distorcoes, tamanhos = {}, [], []
    for l in amostra:
        rel = l.split("\t")[0]
        try:
            with Image.open(os.path.join(raiz_img, rel)) as im:
                w, h = im.size
        except Exception:
            ramos["ilegivel"] = ramos.get("ilegivel", 0) + 1
            continue
        tamanhos.append((w, h))
        ramo = "pad" if w < 256 else "squash"
        ramos[ramo] = ramos.get(ramo, 0) + 1
        # aspecto final / aspecto nativo. 1.00 = sem deformacao.
        asp_nat = w / h
        asp_fin = 256 / 64 if ramo == "squash" else asp_nat
        distorcoes.append(asp_fin / asp_nat)

    print(f"\n{len(amostra)} imagens amostradas do split")
    print("ramo do pre-processamento do BRESSAY:")
    for k, v in sorted(ramos.items()):
        print(f"  {k:<10} {v:>4}  ({100*v/len(amostra):.1f}%)")

    if tamanhos:
        ws = np.array([t[0] for t in tamanhos]); hs = np.array([t[1] for t in tamanhos])
        print(f"\ntamanho nativo: largura {ws.min()}-{ws.max()} (mediana {int(np.median(ws))})"
              f" | altura {hs.min()}-{hs.max()} (mediana {int(np.median(hs))})")
    if distorcoes:
        d = np.array(distorcoes)
        print(f"\ndistorcao de aspecto (1.00 = letra com a proporcao correta)")
        print(f"  mediana {np.median(d):.2f} | min {d.min():.2f} | max {d.max():.2f}")
        print(f"  fora da faixa 0.9-1.1: {100*np.mean((d<0.9)|(d>1.1)):.1f}% das imagens")

    # 2) folhas de contato: o mesmo recorte pelos dois caminhos
    print("\nfolhas de contato:")
    random.seed(a.seed)
    sel = random.sample(linhas, min(a.n, len(linhas)))
    brs, iams, escuros = [], [], 0
    for l in sel:
        rel, _, transcr = l.split("\t")[:3]
        try:
            im = Image.open(os.path.join(raiz_img, rel)).convert("RGB")
        except Exception:
            continue
        b, _ = prep_bressay(im.copy())
        i2, _ = prep_iam(im.copy())
        brs.append(b); iams.append(i2)
        # quanto da imagem virou tinta? no IAM tipico fica abaixo de ~25%
        if np.mean(np.asarray(b.convert("L"), dtype=np.float32) < 128) > 0.5:
            escuros += 1

    folha(brs, os.path.join(a.out, "bressay_como_o_modelo_ve.png"))
    folha(iams, os.path.join(a.out, "mesmo_recorte_no_padrao_iam.png"))
    if brs:
        print(f"\nimagens com mais de 50% de pixels escuros: {escuros}/{len(brs)}"
              "  (alto = a normalizacao de contraste esta saturando)")
    print("\nAbra as duas folhas lado a lado. Se a da esquerda estiver esticada,")
    print("achatada ou preta, o alvo do treino esta errado e nenhum lr conserta.")


if __name__ == "__main__":
    main()
