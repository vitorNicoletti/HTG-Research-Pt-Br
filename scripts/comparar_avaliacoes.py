"""Tabela e comparacao estatistica de modelos avaliados pelo avaliar_pt.py.

Le os paineis.tsv de uma ou mais avaliacoes (mesmo protocolo: mesmas palavras,
escritores e sementes) e imprime, por modelo:
  - marca na acentuada, no esqueleto e na palavra sem acento;
  - diferenca pareada (marca na acentuada - marca no esqueleto, mesmo escritor
    e semente);
  - CER nas palavras sem acento pedido (esqueleto + sem acento), e o mesmo CER
    so nos paineis SEM marca -- separa "a letra piorou" de "o leitor leu a
    marca como letra".
Com --comparar A B, faz bootstrap POR PALAVRA (a palavra e a unidade: os 40
paineis de uma palavra nao sao independentes) da diferenca A - B em cada
medida, com IC95.

    python scripts/comparar_avaliacoes.py \
        --modelo avaliacao_pt_val:iam --modelo avaliacao_peso_val:peso5_16ep \
        --comparar peso5_16ep iam
"""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
from acentos_sinteticos.vocabulario import esqueleto  # noqa: E402


def ler(pasta, rotulo):
    """-> {(palavra, escritor, semente): (tipo, marca, cer)}"""
    d = {}
    with open(os.path.join(pasta, "paineis.tsv"), encoding="utf-8") as f:
        next(f)
        for l in f:
            m, p, t, e, s, mk, _, c = l.rstrip("\n").split("\t")
            if m == rotulo:
                d[(p, e, s)] = (t, int(mk), float(c))
    if not d:
        raise SystemExit(f"{rotulo} nao esta em {pasta}/paineis.tsv")
    return d


def por_palavra(d):
    """Medidas por palavra: {medida: {palavra: valor}}."""
    acc = defaultdict(lambda: defaultdict(list))
    for (p, e, s), (t, mk, c) in d.items():
        acc[f"marca_{t}"][p].append(mk)
        if t == "acentuada":
            acc["diferenca_pareada"][p].append(mk - d[(esqueleto(p), e, s)][1])
        else:
            acc["cer_sem_acento_pedido"][p].append(c)
            if mk == 0:
                acc["cer_sem_acento_pedido_sem_marca"][p].append(c)
    return {k: {p: float(np.mean(v)) for p, v in vv.items()} for k, vv in acc.items()}


MEDIDAS = ("marca_acentuada", "marca_esqueleto", "marca_sem_acento", "diferenca_pareada",
           "cer_sem_acento_pedido", "cer_sem_acento_pedido_sem_marca")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", action="append", required=True, help="pasta_da_avaliacao:rotulo")
    ap.add_argument("--comparar", nargs=2, metavar=("A", "B"), help="rotulos para o bootstrap A - B")
    ap.add_argument("--n_boot", type=int, default=10000)
    a = ap.parse_args()

    dados, medidas = {}, {}
    for m in a.modelo:
        pasta, rot = m.rsplit(":", 1)
        dados[rot] = ler(pasta, rot)
        medidas[rot] = por_palavra(dados[rot])
    # media sobre PAINEIS (como o resumo do avaliar_pt), nao sobre palavras
    print("| modelo | " + " | ".join(MEDIDAS) + " |")
    print("|---|" + "---|" * len(MEDIDAS))
    for rot, d in dados.items():
        cel = []
        for med in MEDIDAS:
            if med.startswith("marca_"):
                v = [mk for (t, mk, _) in d.values() if t == med[6:]]
            elif med == "diferenca_pareada":
                v = [mk - d[(esqueleto(p), e, s)][1] for (p, e, s), (t, mk, _) in d.items() if t == "acentuada"]
            elif med == "cer_sem_acento_pedido":
                v = [c for (t, _, c) in d.values() if t != "acentuada"]
            else:
                v = [c for (t, mk, c) in d.values() if t != "acentuada" and mk == 0]
            cel.append(f"{np.mean(v):+.3f}" if med == "diferenca_pareada" else f"{np.mean(v):.3f}")
        print(f"| {rot} | " + " | ".join(cel) + " |")

    if a.comparar:
        A, B = (medidas[r] for r in a.comparar)
        rnd = np.random.default_rng(0)
        print(f"\nbootstrap por palavra, {a.comparar[0]} - {a.comparar[1]} ({a.n_boot} reamostragens):")
        for med in MEDIDAS:
            ks = sorted(set(A.get(med, {})) & set(B.get(med, {})))
            if not ks:
                continue
            x = np.array([A[med][k] - B[med][k] for k in ks])
            bs = [x[rnd.integers(0, len(x), len(x))].mean() for _ in range(a.n_boot)]
            print(f"  {med:34s} {x.mean():+.3f}  IC95 [{np.percentile(bs, 2.5):+.3f}, "
                  f"{np.percentile(bs, 97.5):+.3f}]  ({len(ks)} palavras)")


if __name__ == "__main__":
    main()
