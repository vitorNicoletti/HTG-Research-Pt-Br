"""Mede onde cada estimador poe o x do acento, contra uma verdade automatica.

Verdade: palavras do IAM (val + test, fora do treino do alinhador) em que
cada componente conexo e exatamente uma letra -- numero de componentes
significativos igual ao numero de letras, ordenados da esquerda para a
direita com pouca sobreposicao em x, sem i/j (o pingo seria componente a
mais). Para cada letra-alvo de acento (a o e u c), conta:

  acerto -- o x do contato cai dentro da caixa da letra certa
  vies   -- (x - centro da letra) / largura da letra, com sinal

Estimadores:
  igual     -- fatias iguais + centroide do corpo (gerador sem alinhador)
  ctc_meio  -- fronteiras no meio entre disparos CTC + centroide
  ctc_pico  -- x direto do disparo CTC
  ctc_vale  -- fronteiras ctc_meio ajustadas a coluna de menos tinta
  igual_vale -- fatias iguais ajustadas aos vales

Ressalva: letras soltas sao o caso facil; o numero e um limite inferior do
erro na escrita cursiva.

    python scripts/avaliar_posicao_letras.py --alinhador modelos/alinhador_iam.pt
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
from acentos_sinteticos import alinhamento, geometria, iam  # noqa: E402

ALVOS = set("aoeuc")
SOBREPOSICAO_MAX = 0.2   # sobreposicao em x entre letras vizinhas / largura da menor
AREA_REL_MIN = 0.03      # componente significativo: area >= isto * maior componente


def letras_soltas(geo, n):
    """Caixas [(x0, x1)] das n letras se cada componente for uma letra; senao None."""
    m = geo.mask.astype(np.uint8)
    k, _, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if k < 2:
        return None
    areas = st[1:, cv2.CC_STAT_AREA]
    sig = [i + 1 for i in range(k - 1) if areas[i] >= AREA_REL_MIN * areas.max()]
    if len(sig) != n:
        return None
    caixas = sorted((st[i, cv2.CC_STAT_LEFT], st[i, cv2.CC_STAT_LEFT] + st[i, cv2.CC_STAT_WIDTH])
                    for i in sig)
    for (a0, a1), (b0, b1) in zip(caixas, caixas[1:]):
        if a1 - b0 > SOBREPOSICAO_MAX * min(a1 - a0, b1 - b0):
            return None
    return caixas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clone", default=os.path.join(RAIZ, "DiffusionPen"))
    ap.add_argument("--alinhador", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max", type=int, default=0, help="limite de palavras (0 = todas)")
    a = ap.parse_args()

    alin = alinhamento.Alinhador(a.alinhador, a.device)
    palavras = [p for s in ("iam_val.txt", "iam_test.txt") for p in iam.listar(a.clone, s)
                if p.texto.isascii() and p.texto.isalpha() and p.texto.islower()
                and 3 <= len(p.texto) <= 10 and not set("ij") & set(p.texto)
                and ALVOS & set(p.texto)]
    nomes = ["igual", "igual_vale", "ctc_meio", "ctc_vale", "ctc_pico"]
    reg = []    # (estimador, posicao, acerto, vies, leitura_ok, logp)
    n_pal = 0
    for p in palavras:
        g = iam.carregar_cinza(p.caminho)
        geo = geometria.analisar(g)
        if geo is None:
            continue
        caixas = letras_soltas(geo, len(p.texto))
        if caixas is None:
            continue
        al = alin.alinhar(g, p.texto)
        if al is None:
            continue
        n_pal += 1
        f_igual = geometria.fatias(geo, len(p.texto))
        f_ctc = alin.fatias(g, geo, p.texto)
        fats = {"igual": f_igual, "igual_vale": geometria.ajustar_aos_vales(geo, f_igual),
                "ctc_meio": f_ctc, "ctc_vale": geometria.ajustar_aos_vales(geo, f_ctc)}
        ok = al.leitura == p.texto
        for k, ch in enumerate(p.texto):
            if ch not in ALVOS:
                continue
            x0, x1 = caixas[k]
            pos = "primeira" if k == 0 else ("ultima" if k == len(p.texto) - 1 else "meio")
            for nome in nomes:
                if nome == "ctc_pico":
                    x = al.centros[k]
                else:
                    x = geometria.contato_superior(geo, fats[nome][k]).x
                reg.append((nome, pos, x0 <= x <= x1, (x - (x0 + x1) / 2) / max(1, x1 - x0),
                            ok, al.logp_medio))
        if a.max and n_pal >= a.max:
            break

    print(f"palavras com letras soltas: {n_pal} (de {len(palavras)} candidatas)")
    print(f"{'estimador':<11} {'posicao':<9} {'n':>5} {'acerto':>7} {'vies med':>9} {'|vies| med':>10}")
    res = {}
    for nome in nomes:
        for pos in ("todas", "primeira", "meio", "ultima"):
            r = [x for x in reg if x[0] == nome and (pos == "todas" or x[1] == pos)]
            if not r:
                continue
            ac = np.mean([x[2] for x in r])
            vies = np.median([x[3] for x in r])
            absv = np.median([abs(x[3]) for x in r])
            res[f"{nome}/{pos}"] = {"n": len(r), "acerto": round(float(ac), 4),
                                    "vies_mediano": round(float(vies), 3)}
            print(f"{nome:<11} {pos:<9} {len(r):>5} {ac:>7.3f} {vies:>+9.3f} {absv:>10.3f}")
    # filtro: o acerto do CTC muda com a leitura livre / log-prob?
    print("\nctc_meio por qualidade do alinhamento:")
    r = [x for x in reg if x[0] == "ctc_meio"]
    for rot, sel in (("leitura == rotulo", [x for x in r if x[4]]),
                     ("leitura != rotulo", [x for x in r if not x[4]])):
        if sel:
            print(f"  {rot:<18} n={len(sel):>5} acerto={np.mean([x[2] for x in sel]):.3f}")
    if r:
        lp = np.array([x[5] for x in r])
        for q0, q1 in ((0, 25), (25, 50), (50, 75), (75, 100)):
            lo, hi = np.percentile(lp, q0), np.percentile(lp, q1)
            sel = [x for x in r if lo <= x[5] <= hi]
            print(f"  logp_medio p{q0}-p{q1} [{lo:.3f}, {hi:.3f}] n={len(sel):>5} "
                  f"acerto={np.mean([x[2] for x in sel]):.3f}")
    print(json.dumps({"palavras": n_pal, **res}))


if __name__ == "__main__":
    main()
