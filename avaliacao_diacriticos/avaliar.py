"""
avaliar.py -- aplica E1 e E2 a um conjunto (gerado ou real) e classifica.

Entrada: uma pasta com manifest.jsonl no formato que gerar_pares.py e
preparar_reais.py produzem. Saida: um CSV por amostra e um resumo.

E1 roda sempre por faixa (funciona com ou sem gemeo) e, quando o gemeo ASCII
existe na mesma pasta, tambem por diferenca. Os dois numeros sao gravados; a
classificacao usa o que for pedido em --eixo1.

E2 so roda com --com-e2, porque carrega o TrOCR.

Classificacao 2x2 do planejamento, por amostra ACENTUADA:

                     base integra        base degradada
    acento presente  acerto              degradacao com diacritico
    acento ausente   omissao do acento   degradacao sem diacritico
"""
import argparse
import csv
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrica as M  # noqa: E402


def carregar(dirbase):
    itens = []
    with open(os.path.join(dirbase, "manifest.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                itens.append(json.loads(line))
    return itens


def ic_bootstrap(valores, n=2000, alfa=0.05, seed=0):
    """IC percentil por bootstrap. Devolve (media, low, high)."""
    v = np.asarray(valores, dtype=np.float64)
    if len(v) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    med = rng.choice(v, size=(n, len(v)), replace=True).mean(axis=1)
    return float(v.mean()), float(np.percentile(med, 100 * alfa / 2)), \
        float(np.percentile(med, 100 * (1 - alfa / 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--csv-out", required=True)
    ap.add_argument("--com-e2", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--eixo1", default="auto", choices=["auto", "faixa", "diff"],
                    help="auto = diff quando ha gemeo, faixa quando nao ha")
    ap.add_argument("--limiar-e1", type=float, default=None,
                    help="escore E1 acima do qual o acento e considerado presente")
    ap.add_argument("--limiar-e2", type=float, default=0.5,
                    help="CER ASCII ate o qual a base e considerada integra")
    ap.add_argument("--folga-coluna", type=float, default=0.5)
    a = ap.parse_args()

    itens = carregar(a.dir)
    por_chave = {}
    for d in itens:
        por_chave[(d["par_id"], d["escritor"], d["semente"], d["acentuada"])] = d

    cache = {}

    def tinta(d):
        if d["arquivo"] not in cache:
            cache[d["arquivo"]] = M.carregar_tinta(os.path.join(a.dir, d["arquivo"]))
        return cache[d["arquivo"]]

    linhas = []
    for d in itens:
        if not d["acentuada"]:
            continue
        if d.get("colapsada"):
            continue
        t_acc = tinta(d)
        gem = por_chave.get((d["par_id"], d["escritor"], d["semente"], False))
        faixa = M.e1_por_faixa(t_acc, d["palavra"], a.folga_coluna)
        diff = (M.e1_por_diff(t_acc, tinta(gem), d["palavra"], a.folga_coluna)
                if gem else None)
        for k, f in enumerate(faixa):
            linha = {
                "arquivo": d["arquivo"], "palavra": d["palavra"],
                "ascii": M.sem_acento(d["palavra"]),
                "par_id": d["par_id"], "escritor": d["escritor"],
                "semente": d["semente"], "real": bool(d.get("real", False)),
                "marca": f["marca"], "onde": f["onde"], "indice": f["indice"],
                "medivel": f["medivel"],
                "faixa_massa": f["massa"], "faixa_massa_rel": f["massa_rel"],
                "faixa_densidade": f["densidade"],
                "tem_gemeo": bool(gem),
            }
            if diff:
                g = diff[k]
                linha.update({"diff_massa_acc": g["massa_acc"],
                              "diff_massa_asc": g["massa_asc"],
                              "diff_delta": g["delta"],
                              "diff_delta_rel": g["delta_rel"],
                              "diff_acima_do_topo": g["acima_do_topo"]})
            else:
                linha.update({"diff_massa_acc": "", "diff_massa_asc": "",
                              "diff_delta": "", "diff_delta_rel": "",
                              "diff_acima_do_topo": ""})
            linhas.append(linha)

    # ---------------- E2 ----------------
    if a.com_e2:
        from reconhecedor import Reconhecedor
        rec = Reconhecedor(device=a.device)
        # le TODAS as imagens (acentuadas e ASCII): o grupo ASCII e o controle
        # que estabelece o erro de base do proprio gerador.
        alvo = [d for d in itens if not d.get("colapsada")]
        caminhos = [os.path.join(a.dir, d["arquivo"]) for d in alvo]
        lidos, cers = rec.cer_ascii(caminhos, [d["palavra"] for d in alvo])
        leitura = {d["arquivo"]: (l, c) for d, l, c in zip(alvo, lidos, cers)}
        json.dump({k: {"lido": v[0], "cer": v[1]} for k, v in leitura.items()},
                  open(os.path.join(a.dir, "e2_leituras.json"), "w"),
                  ensure_ascii=False, indent=1)
        for linha in linhas:
            l, c = leitura.get(linha["arquivo"], ("", None))
            linha["e2_lido"] = l
            linha["e2_cer"] = c
    else:
        for linha in linhas:
            linha["e2_lido"] = ""
            linha["e2_cer"] = None

    # ---------------- classificacao ----------------
    def escore(linha):
        usar = a.eixo1
        if usar == "auto":
            usar = "diff" if linha["tem_gemeo"] else "faixa"
        if usar == "diff":
            return float(linha["diff_delta_rel"] or 0.0)
        return float(linha["faixa_massa_rel"] or 0.0)

    for linha in linhas:
        linha["e1_escore"] = round(escore(linha), 5)
        if a.limiar_e1 is None:
            linha["e1_presente"] = ""
            linha["categoria"] = ""
            continue
        pres = linha["e1_escore"] > a.limiar_e1
        linha["e1_presente"] = pres
        if linha["e2_cer"] is None:
            linha["categoria"] = "presente" if pres else "ausente"
        else:
            base_ok = linha["e2_cer"] <= a.limiar_e2
            linha["categoria"] = (
                "acerto" if (pres and base_ok) else
                "omissao_do_acento" if (not pres and base_ok) else
                "degradacao_com_diacritico" if pres else
                "degradacao_sem_diacritico")

    campos = list(linhas[0].keys()) if linhas else []
    with open(a.csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(linhas)

    # ---------------- resumo ----------------
    print(f"\n{len(linhas)} diacriticos avaliados em {a.dir}")
    print(f"amostras nao mediveis (sem tinta): "
          f"{sum(1 for l in linhas if not l['medivel'])}")
    marcas = sorted({l["marca"] for l in linhas})
    print(f"\n{'marca':14s} {'n':>5s} {'E1 escore medio':>16s} {'IC95':>22s}")
    for m in marcas:
        v = [l["e1_escore"] for l in linhas if l["marca"] == m]
        mu, lo, hi = ic_bootstrap(v)
        print(f"{m:14s} {len(v):5d} {mu:16.4f}   [{lo:.4f}, {hi:.4f}]")
    v = [l["e1_escore"] for l in linhas]
    mu, lo, hi = ic_bootstrap(v)
    print(f"{'TODAS':14s} {len(v):5d} {mu:16.4f}   [{lo:.4f}, {hi:.4f}]")

    if a.limiar_e1 is not None:
        print(f"\npresenca (E1 > {a.limiar_e1}):")
        for m in marcas:
            v = [1.0 if l["e1_presente"] else 0.0
                 for l in linhas if l["marca"] == m]
            mu, lo, hi = ic_bootstrap(v)
            print(f"  {m:14s} {len(v):5d} {100 * mu:6.1f}%   "
                  f"[{100 * lo:.1f}%, {100 * hi:.1f}%]")
        v = [1.0 if l["e1_presente"] else 0.0 for l in linhas]
        mu, lo, hi = ic_bootstrap(v)
        print(f"  {'TODAS':14s} {len(v):5d} {100 * mu:6.1f}%   "
              f"[{100 * lo:.1f}%, {100 * hi:.1f}%]")
        cats = {}
        for l in linhas:
            cats[l["categoria"]] = cats.get(l["categoria"], 0) + 1
        print("\ncategorias:")
        for k in sorted(cats):
            print(f"  {k:28s} {cats[k]:5d}  {100 * cats[k] / len(linhas):5.1f}%")

    if a.com_e2:
        acc = [l["e2_cer"] for l in linhas if l["e2_cer"] is not None]
        leit = json.load(open(os.path.join(a.dir, "e2_leituras.json")))
        asc_arqs = {d["arquivo"] for d in itens
                    if not d["acentuada"] and not d.get("colapsada")}
        asci = [leit[k]["cer"] for k in leit if k in asc_arqs]
        for nome, v in (("acentuadas", acc), ("ASCII (controle)", asci)):
            if v:
                mu, lo, hi = ic_bootstrap(v)
                print(f"\nE2 CER {nome:20s} n={len(v):5d} "
                      f"media={mu:.4f} IC95=[{lo:.4f}, {hi:.4f}] "
                      f"mediana={np.median(v):.4f}")
    print(f"\ncsv: {a.csv_out}")


if __name__ == "__main__":
    main()
