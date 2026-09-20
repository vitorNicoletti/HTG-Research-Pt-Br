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
    ap.add_argument("--reusar-e2", action="store_true",
                    help="usa e2_leituras.json da pasta em vez de rodar o "
                         "TrOCR de novo (o reconhecedor leva ~20 min em CPU "
                         "para 240 imagens, e a leitura nao muda quando so o "
                         "E1 muda)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--eixo1", default="auto", choices=["auto", "faixa", "diff"],
                    help="auto = diff quando ha gemeo, faixa quando nao ha")
    ap.add_argument("--limiar-e1", type=float, default=None,
                    help="escore E1 acima do qual o acento e considerado presente")
    ap.add_argument("--limiar-e2", type=float, default=None,
                    help="CER ASCII ate o qual a base e considerada integra. "
                         "Se omitido, sai do grupo ASCII do proprio conjunto "
                         "(ver --limiar-e2-quantil)")
    ap.add_argument("--limiar-e2-quantil", type=float, default=0.75,
                    help="quantil do CER do grupo ASCII usado como limiar "
                         "quando --limiar-e2 nao e dado")
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
                "fileiras_pauta": f.get("fileiras_pauta", 0),
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
    cache_e2 = os.path.join(a.dir, "e2_leituras.json")
    if a.reusar_e2:
        if not os.path.exists(cache_e2):
            sys.exit(f"--reusar-e2 pedido mas {cache_e2} nao existe; "
                     f"rode uma vez com --com-e2")
        guardado = json.load(open(cache_e2, encoding="utf-8"))
        leitura = {k: (v["lido"], v["cer"]) for k, v in guardado.items()}
        print(f"E2 reaproveitado de {cache_e2} ({len(leitura)} leituras)")
        for linha in linhas:
            l, c = leitura.get(linha["arquivo"], ("", None))
            linha["e2_lido"] = l
            linha["e2_cer"] = c
        a.com_e2 = True
    elif a.com_e2:
        from reconhecedor import Reconhecedor
        rec = Reconhecedor(device=a.device)
        # le TODAS as imagens (acentuadas e ASCII): o grupo ASCII e o controle
        # que estabelece o erro de base do proprio gerador.
        alvo = [d for d in itens if not d.get("colapsada")]
        caminhos = [os.path.join(a.dir, d["arquivo"]) for d in alvo]
        lidos, cers = rec.cer_ascii(caminhos, [d["palavra"] for d in alvo])
        leitura = {d["arquivo"]: (l, c) for d, l, c in zip(alvo, lidos, cers)}
        json.dump({k: {"lido": v[0], "cer": v[1]} for k, v in leitura.items()},
                  open(cache_e2, "w"), ensure_ascii=False, indent=1)
        for linha in linhas:
            l, c = leitura.get(linha["arquivo"], ("", None))
            linha["e2_lido"] = l
            linha["e2_cer"] = c
    else:
        for linha in linhas:
            linha["e2_lido"] = ""
            linha["e2_cer"] = None

    # ---------------- limiar do E2 ----------------
    # Um limiar ABSOLUTO de CER nao serve aqui. O TrOCR e treinado em ingles e
    # nao le bem manuscrito portugues: nos recortes REAIS do BRESSAY ele ja
    # erra muito. Com um corte fixo em 0.5, manuscrito humano de verdade seria
    # classificado como "base degradada", e a categoria deixaria de medir o
    # gerador para medir o reconhecedor.
    #
    # Por isso o padrao e relativo, como pede o planejamento: o grupo ASCII do
    # PROPRIO conjunto estabelece o erro de base, e "base integra" quer dizer
    # "nao erra mais do que o proprio gerador ja erra sem diacritico nenhum".
    # O valor absoluto continua disponivel em --limiar-e2, para comparar
    # conjuntos entre si.
    limiar_e2 = a.limiar_e2
    origem_limiar = "absoluto (--limiar-e2)"
    if limiar_e2 is None and a.com_e2:
        asc_arqs = {d["arquivo"] for d in itens
                    if not d["acentuada"] and not d.get("colapsada")}
        base = [v[1] for k, v in leitura.items()
                if k in asc_arqs and v[1] is not None]
        if base:
            limiar_e2 = float(np.quantile(base, a.limiar_e2_quantil))
            origem_limiar = (f"quantil {a.limiar_e2_quantil:.2f} do grupo "
                             f"ASCII (n={len(base)})")
        else:
            limiar_e2 = 0.5
            origem_limiar = "0.5 (nao havia grupo ASCII no conjunto)"
    if a.com_e2:
        print(f"\nlimiar de E2 = {limiar_e2:.4f}  -- {origem_limiar}")

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
            base_ok = linha["e2_cer"] <= limiar_e2
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
    com_pauta = sum(1 for l in linhas if l.get("fileiras_pauta", 0) > 0)
    print(f"amostras com linha pautada removida: {com_pauta} "
          f"({100 * com_pauta / max(1, len(linhas)):.0f}%)")
    marcas = sorted({l["marca"] for l in linhas})
    print(f"\n{'marca':14s} {'n':>5s} {'E1 escore medio':>16s} {'IC95':>22s}")
    for m in marcas:
        v = [l["e1_escore"] for l in linhas if l["marca"] == m]
        mu, lo, hi = ic_bootstrap(v)
        print(f"{m:14s} {len(v):5d} {mu:16.4f}   [{lo:.4f}, {hi:.4f}]")
    v = [l["e1_escore"] for l in linhas]
    mu, lo, hi = ic_bootstrap(v)
    print(f"{'TODAS':14s} {len(v):5d} {mu:16.4f}   [{lo:.4f}, {hi:.4f}]")

    # Quando existe gemeo, os DOIS eixos sao calculados e vale ver os dois: a
    # variante por faixa e a unica comparavel com recorte real, e a por
    # diferenca e a que tem piso de ruido zero (ver teste_sensibilidade_diff).
    com_gemeo = [l for l in linhas if l["tem_gemeo"]]
    if com_gemeo:
        print(f"\nos dois eixos, nas {len(com_gemeo)} amostras com gemeo ASCII:")
        for rot, col in (("por faixa (massa_rel)", "faixa_massa_rel"),
                         ("por diferenca (delta_rel)", "diff_delta_rel")):
            vv = [float(l[col] or 0.0) for l in com_gemeo]
            mu, lo, hi = ic_bootstrap(vv)
            print(f"  {rot:28s} {mu:8.4f}   [{lo:.4f}, {hi:.4f}]")
        ref = ("referencia do eixo por diferenca (teste_sensibilidade_diff): "
               "0.0000 = gemeos identicos, 0.0389 = meio acento nominal, "
               "0.1657 = acento nominal")
        print(f"  {ref}")

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
