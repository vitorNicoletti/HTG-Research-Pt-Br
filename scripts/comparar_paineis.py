"""Figuras comparativas a partir dos paineis salvos pelo avaliar_pt.py.

O avaliar_pt.py gera, para todo modelo, as MESMAS palavras com os MESMOS
escritores e sementes, e grava cada imagem em paineis/<modelo>/<k>_<escritor>_<semente>.png.
Este script poe os modelos lado a lado, por par minimo: para cada palavra
acentuada, uma figura com uma linha por (modelo, acentuada) e (modelo,
esqueleto) e uma coluna por escritor. A barra acima de cada painel e a
deteccao de marca solta do avaliar_pt.py (medir_marcas): verde = marca na
acentuada (o esperado), vermelho = marca no esqueleto (vazamento).

    python scripts/comparar_paineis.py --modelo iam=avaliacao_pt_val:iam \
        --modelo peso5=avaliacao_peso_val:peso5_16ep --saida saidas/comparacao_peso
"""

import argparse
import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
from acentos_sinteticos.vocabulario import esqueleto  # noqa: E402

FONTES = ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "arial.ttf")
VERDE, VERMELHO = (40, 160, 60), (210, 40, 40)


def fonte(tam):
    for f in FONTES:
        try:
            return ImageFont.truetype(f, tam)
        except OSError:
            pass
    return ImageFont.load_default()


def ler_avaliacao(pasta, rotulo):
    """-> (palavras na ordem k, escritores, {(palavra, escritor, semente): (marca, cer)})."""
    with open(os.path.join(pasta, "resumo.json"), encoding="utf-8") as f:
        r = json.load(f)
    medidas = {}
    with open(os.path.join(pasta, "paineis.tsv"), encoding="utf-8") as f:
        next(f)
        for l in f:
            m, p, _, e, s, marca, _, cer = l.rstrip("\n").split("\t")
            if m == rotulo:
                medidas[(p, e, int(s))] = (int(marca), float(cer))
    if not medidas:
        raise SystemExit(f"modelo {rotulo} nao esta em {pasta}/paineis.tsv")
    return r["palavras"], [str(e) for e in r["escritores"]], medidas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", action="append", required=True,
                    help="nome=pasta_da_avaliacao:rotulo_la_dentro (ordem = ordem das linhas)")
    ap.add_argument("--escritores", type=int, default=6)
    ap.add_argument("--semente", type=int, default=42)
    ap.add_argument("--n_pares", type=int, default=0, help="0 = todos")
    ap.add_argument("--saida", required=True)
    a = ap.parse_args()
    os.makedirs(a.saida, exist_ok=True)

    modelos = []
    for m in a.modelo:
        nome, resto = m.split("=", 1)
        pasta, rot = resto.rsplit(":", 1)
        palavras, escritores, medidas = ler_avaliacao(pasta, rot)
        modelos.append((nome, os.path.join(pasta, "paineis", rot), palavras, escritores, medidas))
    # as avaliacoes tem de ser o mesmo protocolo: mesmas palavras, escritores
    for nome, _, p, e, _ in modelos[1:]:
        if p != modelos[0][2] or e != modelos[0][3]:
            raise SystemExit(f"{nome}: palavras/escritores diferentes de {modelos[0][0]}; nao comparaveis")
    palavras, escritores = modelos[0][2], modelos[0][3][:a.escritores]
    k_de = {p: k for k, p in enumerate(palavras)}
    pares = [(p, esqueleto(p)) for p in palavras if not p.isascii() and esqueleto(p) in k_de]
    if a.n_pares:
        pares = pares[:a.n_pares]

    W, H, BARRA, ROT, TOPO = 256, 64, 5, 210, 30
    f_rot, f_tit = fonte(15), fonte(18)
    for n, (ac, esq) in enumerate(pares):
        linhas = [(nome, pasta, med, palavra, palavra == ac)
                  for nome, pasta, _, _, med in modelos for palavra in (ac, esq)]
        tela = Image.new("RGB", (ROT + len(escritores) * (W + 4), TOPO + len(linhas) * (H + BARRA + 4)), "white")
        d = ImageDraw.Draw(tela)
        d.text((4, 4), f"{ac} / {esq}  -  semente {a.semente}, escritores do iam_test", fill="black", font=f_tit)
        for i, (nome, pasta, med, palavra, acentuada) in enumerate(linhas):
            y = TOPO + i * (H + BARRA + 4)
            if not acentuada:
                d.line([(0, y + H + BARRA + 2), (tela.width, y + H + BARRA + 2)], fill=(150, 150, 150))
            d.text((4, y + 22), f"{nome}: {palavra}", fill="black", font=f_rot)
            for j, e in enumerate(escritores):
                x = ROT + j * (W + 4)
                img = Image.open(os.path.join(pasta, f"{k_de[palavra]:03d}_{e}_{a.semente}.png")).convert("RGB")
                tela.paste(img, (x, y + BARRA))
                marca, _ = med[(palavra, e, a.semente)]
                if marca:
                    d.rectangle([x, y, x + W - 1, y + BARRA - 1], fill=VERDE if acentuada else VERMELHO)
        tela.save(os.path.join(a.saida, f"{n:02d}_{esq}.png"))
    print(f"{len(pares)} figuras em {a.saida}")


if __name__ == "__main__":
    main()
