"""
calibrar.py -- Passo 4: valida a metrica nos dois casos de resposta conhecida
e escolhe o limiar do E1 a partir deles.

O limiar NAO e cravado a priori. Ele sai da separacao entre:
  negativo -- IAM puro, que nunca viu portugues e comprovadamente apaga til e
              cedilha. Resposta esperada: acento ausente.
  positivo -- recortes REAIS do BRESSAY, manuscrito humano. Resposta esperada:
              acento presente.

Se os dois nao separarem, a metrica esta quebrada e nenhum numero em
checkpoint de fine-tune significa coisa alguma.
"""
import argparse
import csv
import json

import numpy as np


def le(caminho, coluna):
    por_marca = {}
    with open(caminho, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                v = float(r[coluna])
            except (ValueError, KeyError, TypeError):
                continue
            por_marca.setdefault(r["marca"], []).append(v)
    return por_marca


def auc(neg, pos):
    """AUC = P(escore de um positivo > escore de um negativo). Mann-Whitney."""
    if not neg or not pos:
        return float("nan")
    v = np.concatenate([np.asarray(neg), np.asarray(pos)])
    r = v.argsort().argsort().astype(float) + 1
    # empates recebem o rank medio
    ordem = np.argsort(v)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[ordem[j + 1]] == v[ordem[i]]:
            j += 1
        if j > i:
            r[ordem[i:j + 1]] = r[ordem[i:j + 1]].mean()
        i = j + 1
    n0, n1 = len(neg), len(pos)
    return (r[n0:].sum() - n1 * (n1 + 1) / 2) / (n0 * n1)


def melhor_limiar(neg, pos):
    """Limiar que maximiza (taxa de acerto no positivo + no negativo)/2."""
    cand = sorted(set(neg) | set(pos))
    melhor = (None, -1.0)
    for i in range(len(cand)):
        t = cand[i]
        sens = np.mean([p > t for p in pos]) if pos else 0
        espec = np.mean([n <= t for n in neg]) if neg else 0
        b = (sens + espec) / 2
        if b > melhor[1]:
            melhor = (t, b)
    return melhor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--negativo", required=True, help="CSV do IAM puro")
    ap.add_argument("--positivo", required=True, help="CSV dos recortes reais")
    ap.add_argument("--coluna", default="e1_escore")
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    neg, pos = le(a.negativo, a.coluna), le(a.positivo, a.coluna)
    marcas = sorted(set(neg) | set(pos))

    print(f"coluna avaliada: {a.coluna}")
    print(f"{'marca':14s} {'n-':>4s} {'n+':>4s} {'media-':>8s} {'media+':>8s} "
          f"{'AUC':>6s} {'limiar':>8s} {'omissao IAM':>12s} {'presenca real':>14s}")
    print("-" * 88)
    resumo = {}
    for m in marcas + ["TODAS"]:
        if m == "TODAS":
            vn = [x for v in neg.values() for x in v]
            vp = [x for v in pos.values() for x in v]
        else:
            vn, vp = neg.get(m, []), pos.get(m, [])
        if not vn or not vp:
            print(f"{m:14s} {len(vn):4d} {len(vp):4d}   (sem os dois lados)")
            continue
        A = auc(vn, vp)
        t, _ = melhor_limiar(vn, vp)
        om = float(np.mean([x <= t for x in vn]))   # IAM: acento ausente
        pr = float(np.mean([x > t for x in vp]))    # real: acento presente
        print(f"{m:14s} {len(vn):4d} {len(vp):4d} {np.mean(vn):8.4f} "
              f"{np.mean(vp):8.4f} {A:6.3f} {t:8.4f} {100*om:11.1f}% {100*pr:13.1f}%")
        resumo[m] = {"n_neg": len(vn), "n_pos": len(vp),
                     "media_neg": float(np.mean(vn)), "media_pos": float(np.mean(vp)),
                     "auc": float(A), "limiar": float(t),
                     "omissao_iam": om, "presenca_real": pr}

    print("\nLeitura: AUC = probabilidade de um recorte real pontuar acima de uma")
    print("amostra do IAM puro. 0.5 = a metrica nao distingue nada; 1.0 = separa")
    print("perfeitamente. 'omissao IAM' e 'presenca real' sao as duas taxas que o")
    print("Passo 4 exige que fiquem perto de 100%.")

    if a.json_out:
        json.dump(resumo, open(a.json_out, "w"), indent=1)
        print("\njson:", a.json_out)


if __name__ == "__main__":
    main()
