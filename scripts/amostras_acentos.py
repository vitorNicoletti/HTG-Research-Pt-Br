"""Gera amostras de acentos sinteticos sobre palavras do IAM, para avaliacao.

Duas folhas e um manifesto em --saida:

  amostras.png   -- palavras sorteadas: original | acentuada | depuracao.
                    Na depuracao: fatias por letra (cinza), fatia escolhida
                    (amarelo), corpo da palavra -- altura-x e base (azul) -- e
                    ponto de contato (vermelho).
  variacoes.png  -- a mesma palavra e a mesma letra com varias sementes, um
                    bloco por tipo de sinal: mostra que nenhum sinal se repete.
  manifesto.jsonl e amostras/*.png -- cada amostra gerada, com os parametros.

    python scripts/amostras_acentos.py --n 30 --seed 0
"""

import argparse
import json
import os
import random
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
from acentos_sinteticos import gerador, iam  # noqa: E402

ALTURA_VIS = 80        # altura de exibicao de cada imagem nas folhas
LARGURA_VIS = 330      # largura maxima de exibicao
# letra forcada por tipo na folha de variacoes (o i mostra a troca do pingo)
VARIACOES = [("til", "a", "ã"), ("til", "o", "õ"), ("agudo", "e", "é"),
             ("agudo", "i", "í"), ("circunflexo", "e", "ê"),
             ("grave", "a", "à"), ("cedilha", "c", "ç")]


def fonte(t):
    try:
        cam = subprocess.run(["fc-match", "-f", "%{file}", "sans:lang=pt"],
                             capture_output=True, text=True).stdout.strip()
        return ImageFont.truetype(cam, t)
    except Exception:
        return ImageFont.load_default()


def para_vis(im):
    """PIL -> altura ALTURA_VIS preservando o aspecto, largura limitada."""
    w, h = im.size
    esc = min(ALTURA_VIS / h, LARGURA_VIS / w)
    return im.resize((max(1, int(w * esc)), max(1, int(h * esc))), Image.LANCZOS)


def depuracao(a):
    im = Image.fromarray(a.imagem).convert("RGB")
    d = ImageDraw.Draw(im, "RGBA")
    h = im.size[1]
    xa, xb = a.fatias[a.indice]
    d.rectangle([xa, 0, xb, h - 1], fill=(255, 220, 0, 70))
    for x0, _ in a.fatias[1:]:
        d.line([(x0, 0), (x0, h - 1)], fill=(120, 120, 120, 200))
    for y in a.corpo:
        d.line([(0, y), (im.size[0] - 1, y)], fill=(0, 90, 255, 180))
    cx, cy = a.contato
    d.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=(230, 0, 0, 255))
    return im


def elegiveis(palavras):
    """Palavras minusculas, so letras ASCII, 3-10 letras, com candidato."""
    return [p for p in palavras
            if p.texto.isascii() and p.texto.isalpha() and p.texto.islower()
            and 3 <= len(p.texto) <= 10 and gerador.candidatos(p.texto)]


def folha_amostras(itens, caminho, f_rot):
    linha_h = ALTURA_VIS + 26
    larg = 3 * (LARGURA_VIS + 12) + 20
    folha = Image.new("RGB", (larg, 30 + len(itens) * linha_h), "white")
    d = ImageDraw.Draw(folha)
    for k, t in enumerate(("original (IAM)", "acentuada", "depuracao")):
        d.text((10 + k * (LARGURA_VIS + 12), 8), t, fill=(140, 0, 0), font=f_rot)
    y = 30
    for p, a in itens:
        d.text((10, y), f"{a.original} -> {a.rotulo}   ({a.tipo}, escritor {p.escritor})",
               fill="black", font=f_rot)
        ims = (Image.open(p.caminho).convert("RGB"), Image.fromarray(a.imagem).convert("RGB"),
               depuracao(a))
        for k, im in enumerate(ims):
            folha.paste(para_vis(im), (10 + k * (LARGURA_VIS + 12), y + 20))
        y += linha_h
    folha.save(caminho)


def folha_variacoes(blocos, n_sementes, caminho, f_rot):
    col_w = 200
    linha_h = ALTURA_VIS + 26
    folha = Image.new("RGB", (10 + n_sementes * (col_w + 8) + 10, 10 + len(blocos) * linha_h), "white")
    d = ImageDraw.Draw(folha)
    y = 10
    for rotulo, amostras in blocos:
        d.text((10, y), rotulo, fill=(140, 0, 0), font=f_rot)
        for k, a in enumerate(amostras):
            im = Image.fromarray(a.imagem).convert("RGB")
            w, h = im.size
            esc = min(ALTURA_VIS / h, col_w / w)
            folha.paste(im.resize((int(w * esc), int(h * esc)), Image.LANCZOS),
                        (10 + k * (col_w + 8), y + 20))
        y += linha_h
    folha.save(caminho)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clone", default=os.path.join(RAIZ, "DiffusionPen"))
    ap.add_argument("--n", type=int, default=30, help="palavras na folha de amostras")
    ap.add_argument("--sementes", type=int, default=6, help="variacoes por tipo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--saida", default=os.path.join(RAIZ, "saidas", "acentos_sinteticos"))
    a = ap.parse_args()

    os.makedirs(os.path.join(a.saida, "amostras"), exist_ok=True)
    f_rot = fonte(14)
    palavras = elegiveis(iam.listar(a.clone))
    print(f"palavras elegiveis no IAM: {len(palavras)}")
    rnd = random.Random(a.seed)

    # ---- folha de amostras ----
    itens = []
    with open(os.path.join(a.saida, "manifesto.jsonl"), "w", encoding="utf-8") as man:
        for k, p in enumerate(rnd.sample(palavras, a.n * 2)):
            semente = a.seed * 1_000_003 + k
            am = gerador.acentuar(iam.carregar_cinza(p.caminho), p.texto, random.Random(semente))
            if am is None:
                continue
            nome = f"{len(itens):03d}_{am.rotulo}.png"
            Image.fromarray(am.imagem).save(os.path.join(a.saida, "amostras", nome))
            man.write(json.dumps({"arquivo": nome, "iam": os.path.relpath(p.caminho, a.clone),
                                  "escritor": p.escritor, "semente": semente, **am.manifesto()},
                                 ensure_ascii=False) + "\n")
            itens.append((p, am))
            if len(itens) == a.n:
                break
    folha_amostras(itens, os.path.join(a.saida, "amostras.png"), f_rot)
    tipos = {}
    for _, am in itens:
        tipos[am.letra] = tipos.get(am.letra, 0) + 1
    print("letras sorteadas:", dict(sorted(tipos.items(), key=lambda x: -x[1])))

    # ---- folha de variacoes: mesma palavra, mesma letra, sementes diferentes ----
    blocos = []
    for tipo, base, letra in VARIACOES:
        cands = [p for p in palavras if base in p.texto and 4 <= len(p.texto) <= 8]
        for p in rnd.sample(cands, 2):
            i = p.texto.index(base)
            g = iam.carregar_cinza(p.caminho)
            ams = [gerador.acentuar(g, p.texto, random.Random(s), escolha=(i, letra))
                   for s in range(a.sementes)]
            ams = [x for x in ams if x is not None]
            if ams:
                blocos.append((f"{tipo}: {p.texto} -> {ams[0].rotulo}  (escritor {p.escritor})", ams))
    folha_variacoes(blocos, a.sementes, os.path.join(a.saida, "variacoes.png"), f_rot)
    print("folhas:", os.path.join(a.saida, "amostras.png"), os.path.join(a.saida, "variacoes.png"))


if __name__ == "__main__":
    main()
