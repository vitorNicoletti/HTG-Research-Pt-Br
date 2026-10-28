"""
Metrica final, etapas 1 e 2: aplica sinais.py a uma avaliacao do
scripts/avaliar_pt.py e resume por modelo.

Para cada imagem: presenca dos sinais pedidos (acento em cima e cedilha),
letras com marca em cima e a categoria da palavra (correto, a mais, letra
errada, ausente; limpa ou marca indevida nas palavras sem nada pedido).

Uma avaliacao e a pasta do avaliar_pt.py, com paineis.tsv e
paineis/<rotulo>/<k>_<escritor>_<semente>.png, onde k e a ordem em que o par
(palavra, tipo) aparece no paineis.tsv. Em vez da pasta, pode-se dar o
paineis.tsv e a pasta das imagens separados (--paineis_tsv).

Grava <saida>/sinais_<rotulo>.tsv, uma linha por imagem, e acrescenta o
resumo do modelo em <saida>/resumo_sinais.tsv. Modelo ja no resumo e pulado.

Validacao contra o gabarito humano: experimentos/validar_presenca.py e
experimentos/validar_categorias.py (ver README, secao "Metrica final").

    python avaliacao_diacriticos/avaliar_sinais.py --saida avaliacao_diacriticos/resultados/sinais \\
        --modelo zona_vogais_40ep=<pasta das imagens>:<pasta com o paineis.tsv>
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
from PIL import Image

MODULO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, MODULO)

from sinais import Detector, sinais_pedidos  # noqa: E402

CATEGORIAS = ["correto", "a mais", "letra errada", "ausente"]


def ler_paineis(tsv, rotulo):
    with open(tsv, encoding="utf-8") as f:
        rs = [r for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE) if r["modelo"] == rotulo]
    ordem = []
    for r in rs:
        if (r["palavra"], r["tipo"]) not in ordem:
            ordem.append((r["palavra"], r["tipo"]))
    k = {p: i for i, p in enumerate(ordem)}
    return rs, k


def resumir(rotulo, linhas):
    ac = [l for l in linhas if l["tipo"] == "acentuada" and l["categoria"]]
    sem = [l for l in linhas if l["tipo"] != "acentuada" and l["categoria"]]
    pct = lambda v, n: f"{100 * v / n:.1f}" if n else ""
    r = {"modelo": rotulo, "acentuadas": len(ac)}
    for c in CATEGORIAS:
        r[c] = pct(sum(l["categoria"] == c for l in ac), len(ac))
    # presenca de cada tipo de sinal, so nas palavras que o pedem
    cima = [l for l in ac if l["pede_acima"] and l["acima"] != ""]
    ced = [l for l in ac if l["pede_cedilha"] and l["cedilha"] != ""]
    r["presente_acento_em_cima"] = pct(sum(l["acima"] == "1" for l in cima), len(cima))
    r["presente_cedilha"] = pct(sum(l["cedilha"] == "1" for l in ced), len(ced))
    r["sem_nada_pedido"] = len(sem)
    r["marca_indevida"] = pct(sum(l["categoria"] == "marca indevida" for l in sem), len(sem))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", action="append", required=True, help="rotulo=pasta_das_imagens:pasta_do_paineis.tsv")
    ap.add_argument("--saida", required=True)
    a = ap.parse_args()
    os.makedirs(a.saida, exist_ok=True)
    resumo = os.path.join(a.saida, "resumo_sinais.tsv")
    feitos = set()
    if os.path.isfile(resumo):
        with open(resumo, encoding="utf-8") as f:
            feitos = {r["modelo"] for r in csv.DictReader(f, delimiter="\t")}
    d = Detector()
    for m in a.modelo:
        rotulo, resto = m.split("=", 1)
        pasta, tsv = resto.rsplit(":", 1)
        if rotulo in feitos:
            print(f"{rotulo}: ja no resumo")
            continue
        t0 = time.time()
        rs, k = ler_paineis(os.path.join(tsv, "paineis.tsv"), rotulo)
        if len(rs) != 3200:
            raise SystemExit(f"{rotulo}: {len(rs)} linhas no paineis.tsv, esperava 3200")
        linhas = []
        for r in rs:
            img = os.path.join(pasta, f"{k[(r['palavra'], r['tipo'])]:03d}_{r['escritor']}_{r['semente']}.png")
            g = np.asarray(Image.open(img).convert("L"), dtype=np.float32)
            v = d.avaliar(g, r["palavra"])
            b = lambda x: "" if x is None else str(int(x))
            linhas.append({"palavra": r["palavra"], "tipo": r["tipo"], "escritor": r["escritor"], "semente": r["semente"],
                           "sinais": " ".join(s for _, s in sinais_pedidos(r["palavra"])),
                           "pede_acima": v["pede_acima"], "pede_cedilha": v["pede_cedilha"],
                           "acima": b(v["acima"]), "cedilha": b(v["cedilha"]),
                           "letras": "" if v["letras"] is None else " ".join(map(str, sorted(v["letras"]))),
                           "categoria": v["categoria"] or ""})
        with open(os.path.join(a.saida, f"sinais_{rotulo}.tsv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(linhas[0]), delimiter="\t")
            w.writeheader(); w.writerows(linhas)
        r = resumir(rotulo, linhas)
        novo = not os.path.isfile(resumo)
        with open(resumo, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(r), delimiter="\t")
            if novo:
                w.writeheader()
            w.writerow(r)
        print(f"{rotulo}: {time.time() - t0:.0f} s  " + "  ".join(f"{c} {v}" for c, v in r.items() if c != "modelo"), flush=True)


if __name__ == "__main__":
    main()
