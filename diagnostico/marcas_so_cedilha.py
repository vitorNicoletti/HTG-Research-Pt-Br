"""Marca em cima e marca embaixo, separadas, nas palavras so com cedilha.

A coluna "marca" do paineis.tsv (scripts/avaliar_pt.py) junta as duas. Numa
palavra cujo unico sinal e a cedilha, marca embaixo e quase sempre a propria
cedilha solta do c, e so a marca em cima e acento a mais. A anotacao humana do
modelo de 40 epocas (avaliacao_diacriticos/anotacao_40ep.md) mostrou a
diferenca: a coluna acusou 15 de 55 imagens, as pessoas viram acento a mais em
1, e a marca so em cima acusou 4.

Precisa das imagens da avaliacao, que nao ficam no git. Grava uma linha por
imagem em diagnostico/resultados/acento_separado/marcas_so_cedilha.tsv; o
diagnostico/grafico_epocas.py le esse arquivo. Modelo ja presente no arquivo
e pulado.

    python diagnostico/marcas_so_cedilha.py \\
        --modelo zona_vogais_40ep=<pasta paineis/zona_vogais_40ep>:<pasta com o paineis.tsv>
"""
import argparse
import csv
import os
import sys
import unicodedata

import numpy as np
from PIL import Image

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "scripts"))
from medir_marcas import marcas  # noqa: E402

SAIDA = os.path.join(RAIZ, "diagnostico", "resultados", "acento_separado", "marcas_so_cedilha.tsv")
CABECALHO = ["modelo", "palavra", "escritor", "semente", "acima", "abaixo"]


def so_cedilha(palavra):
    return {c for c in unicodedata.normalize("NFD", palavra) if unicodedata.combining(c)} == {"̧"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", action="append", required=True, help="rotulo=pasta_dos_paineis:pasta_do_paineis.tsv")
    a = ap.parse_args()
    feitos = set()
    if os.path.isfile(SAIDA):
        with open(SAIDA, encoding="utf-8") as f:
            feitos = {r["modelo"] for r in csv.DictReader(f, delimiter="\t")}
    novo = not os.path.isfile(SAIDA)
    with open(SAIDA, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        if novo:
            w.writerow(CABECALHO)
        for m in a.modelo:
            rot, resto = m.split("=", 1)
            pasta, tsv = resto.rsplit(":", 1)
            if rot in feitos:
                print(f"{rot}: ja no arquivo")
                continue
            with open(os.path.join(tsv, "paineis.tsv"), encoding="utf-8") as g:
                rs = [r for r in csv.DictReader(g, delimiter="\t", quoting=csv.QUOTE_NONE) if r["modelo"] == rot]
            if len(rs) != 3200:
                raise SystemExit(f"{rot}: {len(rs)} linhas no paineis.tsv, esperava 3200")
            # o indice do arquivo e a ordem em que (palavra, tipo) aparece no paineis.tsv
            ordem = []
            for r in rs:
                if (r["palavra"], r["tipo"]) not in ordem:
                    ordem.append((r["palavra"], r["tipo"]))
            k = {p: i for i, p in enumerate(ordem)}
            n = acima = abaixo = 0
            for r in rs:
                if r["tipo"] != "acentuada" or not so_cedilha(r["palavra"]):
                    continue
                img = os.path.join(pasta, f"{k[(r['palavra'], r['tipo'])]:03d}_{r['escritor']}_{r['semente']}.png")
                g = np.asarray(Image.open(img).convert("L"), dtype=np.float32)
                ac, ab, _, _ = marcas(g)
                if int(ac + ab > 0) != int(r["marca"]):
                    raise SystemExit(f"{rot}: {img} nao reproduz a coluna marca do paineis.tsv")
                w.writerow([rot, r["palavra"], r["escritor"], r["semente"], int(ac > 0), int(ab > 0)])
                n += 1; acima += ac > 0; abaixo += ab > 0
            print(f"{rot}: {n} imagens, marca em cima {acima / n:.1%}, embaixo {abaixo / n:.1%}")


if __name__ == "__main__":
    main()
