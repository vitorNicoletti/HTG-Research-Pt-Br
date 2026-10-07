"""
teste_folga.py -- a janela de coluna do E1 e robusta em material REAL?

coluna_do_caractere() divide a caixa de tinta em fatias iguais. No par minimo
isso e defensavel: e a MESMA aproximacao nos dois gemeos, que tem o mesmo
numero de caracteres e (medido no Passo 1) o mesmo alinhamento, entao o erro
se cancela na diferenca. Em recorte REAL nao ha gemeo, a letra e cursiva e a
largura por caractere varia muito -- o erro nao se cancela.

`folga` alarga a janela em fracoes de largura de caractere para cada lado.
Com folga=0.5 a janela tem 2x a largura de um caractere: numa palavra de 3
letras isso ja cobre quase a palavra inteira, e "tinta na coluna do caractere
acentuado" deixa de significar coisa alguma.

Este teste mede duas coisas:

  1. quanto a janela cobre da palavra, por comprimento de palavra;
  2. quanto o escore E1 e a SEPARACAO contra o controle negativo mudam quando
     a folga varia. Se a separacao for estavel, a fragilidade e teorica; se
     virar de ponta cabeca, o E1 por faixa em material real e fragil e isso
     precisa ser declarado no TCC.

    python avaliacao_diacriticos/teste_folga.py \\
        --real avaliacao_diacriticos/reais_test \\
        --negativo avaliacao_diacriticos/ger_iam_puro
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrica as M  # noqa: E402
from calibrar import auc, melhor_limiar  # noqa: E402

FOLGAS = [0.0, 0.25, 0.5, 1.0, 2.0]


def escores(dirbase, folga):
    """{marca: [massa_rel, ...]} para as amostras acentuadas."""
    out = collections.defaultdict(list)
    for linha in open(os.path.join(dirbase, "manifest.jsonl"), encoding="utf-8"):
        r = json.loads(linha)
        if not r.get("acentuada") or r.get("colapsada"):
            continue
        t = M.carregar_tinta(os.path.join(dirbase, r["arquivo"]))
        for d in M.e1_por_faixa(t, r["palavra"], folga):
            if d["medivel"]:
                out[d["marca"]].append(d["massa_rel"])
    return out


def cobertura(dirbase, folga):
    """Fracao da caixa de tinta coberta pela janela, por comprimento."""
    por_len = collections.defaultdict(list)
    for linha in open(os.path.join(dirbase, "manifest.jsonl"), encoding="utf-8"):
        r = json.loads(linha)
        if not r.get("acentuada") or r.get("colapsada"):
            continue
        t = M.carregar_tinta(os.path.join(dirbase, r["arquivo"]))
        mask, _ = M.mascara_de_tinta(t)
        cx = M.caixa_tinta(mask)
        if cx is None:
            continue
        n = len(M.sem_acento(r["palavra"]))
        for indice, _, _ in M.diacriticos(r["palavra"]):
            col = M.coluna_do_caractere(mask, indice, n, folga)
            if col:
                por_len[n].append((col[1] - col[0]) / (cx[1] - cx[0]))
    return por_len


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--negativo", required=True)
    a = ap.parse_args()

    print("1. QUANTO A JANELA COBRE DA PALAVRA (recortes reais)\n")
    print(f"{'n chars':>8s}" + "".join(f"{f'folga={f}':>13s}" for f in FOLGAS))
    cobs = {f: cobertura(a.real, f) for f in FOLGAS}
    comprimentos = sorted({n for c in cobs.values() for n in c})
    for n in comprimentos:
        v = [cobs[f].get(n, []) for f in FOLGAS]
        if not any(v):
            continue
        cel = [f"{100*np.median(x):.0f}%" if x else "--" for x in v]
        amostras = max(len(x) for x in v)
        print(f"{n:>8d}" + "".join(f"{c:>13s}" for c in cel) + f"   (n={amostras})")
    print("\n100% = a janela cobre a palavra inteira, e o E1 deixa de ser")
    print("'tinta na coluna do caractere' e vira 'tinta acima da palavra'.")

    print("\n\n2. ESCORE E SEPARACAO EM FUNCAO DA FOLGA\n")
    pos = {f: escores(a.real, f) for f in FOLGAS}
    neg = {f: escores(a.negativo, f) for f in FOLGAS}
    marcas = sorted(set(pos[FOLGAS[0]]) | set(neg[FOLGAS[0]]))

    print(f"{'marca':<14}" + "".join(f"{f'folga={f}':>16s}" for f in FOLGAS))
    print(f"{'':14}" + "".join(f"{'real / AUC':>16s}" for _ in FOLGAS))
    print("-" * (14 + 16 * len(FOLGAS)))
    for m in marcas + ["TODAS"]:
        cel = []
        for f in FOLGAS:
            if m == "TODAS":
                vp = [x for v in pos[f].values() for x in v]
                vn = [x for v in neg[f].values() for x in v]
            else:
                vp, vn = pos[f].get(m, []), neg[f].get(m, [])
            if not vp or not vn:
                cel.append("--")
                continue
            cel.append(f"{np.mean(vp):.2f} / {auc(vn, vp):.3f}")
        print(f"{m:<14}" + "".join(f"{c:>16s}" for c in cel))

    print("\n\n3. NO MELHOR LIMIAR DE CADA FOLGA (todas as marcas juntas)\n")
    print(f"{'folga':>7s} {'limiar':>8s} {'omissao neg':>13s} {'presenca real':>15s} {'AUC':>7s}")
    for f in FOLGAS:
        vp = [x for v in pos[f].values() for x in v]
        vn = [x for v in neg[f].values() for x in v]
        t, _ = melhor_limiar(vn, vp)
        om = np.mean([x <= t for x in vn])
        pr = np.mean([x > t for x in vp])
        print(f"{f:>7.2f} {t:>8.4f} {100*om:>12.1f}% {100*pr:>14.1f}% {auc(vn, vp):>7.3f}")


if __name__ == "__main__":
    main()
