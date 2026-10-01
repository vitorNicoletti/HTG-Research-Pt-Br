"""
ssim_psnr.py -- SSIM e PSNR servem para medir presenca de diacritico?

O orientador sugeriu comparar a metrica com SSIM e PSNR. As duas sao metricas
de REFERENCIA COMPLETA: precisam de uma imagem correta, alinhada pixel a pixel,
para comparar. Geracao de manuscrito nao tem isso, porque o modelo produz uma
instancia nova e nao a reconstrucao de uma imagem existente.

O par minimo e a unica excecao. Dentro dele a gemea ASCII e referencia para
tudo que nao e o acento. Entao e la, e so la, que as duas cabem.

Veto de escopo, declarado antes de medir: nenhuma das duas serve em recorte
real, onde nao existe gemeo. Elas disputam apenas com e1_por_diff, nunca com
e1_por_faixa, que e a variante usada em material real.

Quatro controles, dois ja em disco e dois construidos aqui:

  NEG-gerador   o IAM puro, que nao desenha acento
  POS-pintado   acento nominal pintado sobre a propria gemea
  NEG-deriva    ASCII(semente i) contra ASCII(semente j). Zero acento, mas
                variacao real do gerador. O E1 nao precisava deste controle;
                o SSIM global precisa, porque mede a deriva junto com o acento.
  NEG-deslocado acento pintado na coluna de um caractere SEM acento. Mesma
                massa de tinta, lugar errado. O E1 ignora, porque restringe a
                coluna do caractere acentuado. SSIM e PSNR globais nao tem como
                ignorar. E o controle que decide entre global e local.

    python avaliacao_diacriticos/ssim_psnr.py \\
        --dir avaliacao_diacriticos/amostras/ger_iam_n696 \\
        --csv-out avaliacao_diacriticos/resultados/ssim_psnr.csv
"""
import argparse
import collections
import csv
import json
import os
import sys

import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrica as M      # noqa: E402
import e1                # noqa: E402
from calibrar import auc  # noqa: E402
from teste_sensibilidade_diff import pinta_acento  # noqa: E402

JANELA = 7   # o SSIM exige janela impar; 7 e o padrao do skimage


def alinha_cinza(acc, asc):
    """Desloca a imagem cinza `acc` pelo mesmo shift que o E1 usa."""
    dx, dy = e1.deslocamento(e1.tinta(acc), e1.tinta(asc))
    H, W = acc.shape
    out = np.zeros_like(acc)
    out[max(0, dy):min(H, H + dy), max(0, dx):min(W, W + dx)] = \
        acc[max(0, -dy):min(H, H - dy), max(0, -dx):min(W, W - dx)]
    return out


def regiao(asc, palavra, indice, onde, folga=0.5):
    """A mesma regiao que o E1 mede: faixa do acento x coluna do caractere."""
    mask, _ = M.mascara_de_tinta(asc)
    n = len(M.sem_acento(palavra))
    return M._regiao(mask, indice, n, onde, folga)


def _psnr(ref, img):
    """-PSNR, com teto. Regiao identica da PSNR infinito; aqui vira -120 dB,
    que e "maximamente parecido" sem quebrar media e AUC."""
    v = peak_signal_noise_ratio(ref, img, data_range=1.0)
    return -120.0 if not np.isfinite(v) else -float(v)


def escores(acc, asc, r):
    """(dissim global, dissim local, -psnr global, -psnr local, motivo).

    Orientados "maior = mais diferente", para o AUC ter o mesmo sentido do E1.
    `motivo` diz por que o local ficou vazio, quando ficar.
    """
    g_ssim = 1.0 - structural_similarity(acc, asc, data_range=1.0)
    g_psnr = _psnr(asc, acc)
    if r is None:
        return g_ssim, None, g_psnr, None, "sem regiao"
    a, b = acc[r[0], r[1]], asc[r[0], r[1]]
    h, w = a.shape
    lado = min(h, w)
    if lado < 3:
        return g_ssim, None, g_psnr, None, "regiao < 3 px"
    win = min(JANELA, lado if lado % 2 else lado - 1)
    l_ssim = 1.0 - structural_similarity(a, b, data_range=1.0, win_size=win)
    l_psnr = _psnr(b, a)
    return g_ssim, l_ssim, g_psnr, l_psnr, ("ok" if win == JANELA
                                            else f"janela {win}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--csv-out", required=True)
    ap.add_argument("--escala", type=float, default=1.0,
                    help="tamanho do acento pintado no grupo POS")
    a = ap.parse_args()

    itens = [json.loads(l) for l in
             open(os.path.join(a.dir, "manifest.jsonl"), encoding="utf-8")]
    gem = {(x["par_id"], x["escritor"], x["semente"]): x
           for x in itens if not x["acentuada"]}
    # para o controle de deriva: todas as ASCII de cada (par_id, escritor)
    ascii_por_chave = collections.defaultdict(list)
    for x in itens:
        if not x["acentuada"]:
            ascii_por_chave[(x["par_id"], x["escritor"])].append(x)

    linhas = []

    def registra(grupo, marca, palavra, acc, asc, r, k):
        gs, ls, gp, lp, motivo = escores(acc, asc, r)
        # O E1 medido nas MESMAS imagens, para a tabela ser comparavel.
        try:
            e1v = e1.e1(acc, asc, palavra)[k]
        except Exception:
            e1v = float("nan")
        linhas.append({"grupo": grupo, "marca": marca, "palavra": palavra,
                       "e1": round(e1v, 6),
                       "dissim_global": round(gs, 6),
                       "dissim_local": "" if ls is None else round(ls, 6),
                       "psnr_global": round(gp, 4),
                       "psnr_local": "" if lp is None else round(lp, 4),
                       "motivo": motivo})

    for d in itens:
        if not d["acentuada"] or d.get("colapsada"):
            continue
        g = gem.get((d["par_id"], d["escritor"], d["semente"]))
        if not g:
            continue
        pal = d["palavra"]
        t_acc = M.carregar_tinta(os.path.join(a.dir, d["arquivo"]))
        t_asc = M.carregar_tinta(os.path.join(a.dir, g["arquivo"]))
        if not e1.tinta(t_acc).any() or not e1.tinta(t_asc).any():
            continue
        t_acc = alinha_cinza(t_acc, t_asc)
        n = len(M.sem_acento(pal))
        acentuados = [i for i, _, _ in M.diacriticos(pal)]

        for k, (indice, marca, onde) in enumerate(M.diacriticos(pal)):
            r = regiao(t_asc, pal, indice, onde)
            registra("NEG-gerador", marca, pal, t_acc, t_asc, r, k)

            p_img = pinta_acento(t_asc, pal, indice, onde, a.escala)
            registra("POS-pintado", marca, pal, p_img, t_asc, r, k)

            # acento do tamanho certo, na coluna de um caractere sem acento
            livres = [j for j in range(n) if j not in acentuados]
            if livres:
                j = livres[len(livres) // 2]
                desl = pinta_acento(t_asc, pal, j, onde, a.escala)
                registra("NEG-deslocado", marca, pal, desl, t_asc, r, k)

        # deriva: duas ASCII do mesmo par e escritor, sementes diferentes
        irmas = ascii_por_chave[(d["par_id"], d["escritor"])]
        if len(irmas) >= 2 and d["semente"] == irmas[0]["semente"]:
            for k in range(1, len(irmas)):
                t0 = M.carregar_tinta(os.path.join(a.dir, irmas[0]["arquivo"]))
                t1 = M.carregar_tinta(os.path.join(a.dir, irmas[k]["arquivo"]))
                if not e1.tinta(t0).any() or not e1.tinta(t1).any():
                    continue
                t1a = alinha_cinza(t1, t0)
                for k, (indice, marca, onde) in enumerate(M.diacriticos(pal)):
                    registra("NEG-deriva", marca, pal, t1a, t0,
                             regiao(t0, pal, indice, onde), k)

    campos = list(linhas[0].keys())
    with open(a.csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(linhas)

    # ---------------- resumo ----------------
    grupos = collections.defaultdict(lambda: collections.defaultdict(list))
    for l in linhas:
        for col in ("e1", "dissim_global", "dissim_local",
                    "psnr_global", "psnr_local"):
            if l[col] != "" and not (isinstance(l[col], float)
                                     and np.isnan(l[col])):
                grupos[col][l["grupo"]].append(float(l[col]))

    print(f"\n{len(linhas)} medidas em {a.dir}")
    motivos = collections.Counter(l["motivo"] for l in linhas)
    print("regiao local:", dict(motivos))

    print(f"\n{'escore':16s}" + "".join(f"{g:>16s}" for g in
          ("NEG-gerador", "POS-pintado", "NEG-deriva", "NEG-deslocado")))
    for col in ("e1", "dissim_global", "dissim_local",
                "psnr_global", "psnr_local"):
        cel = []
        for g in ("NEG-gerador", "POS-pintado", "NEG-deriva", "NEG-deslocado"):
            v = grupos[col][g]
            cel.append(f"{np.mean(v):.4f}" if v else "--")
        print(f"{col:16s}" + "".join(f"{c:>16s}" for c in cel))

    print(f"\nAUC contra POS-pintado (maior = separa melhor)")
    print(f"{'escore':16s}{'vs NEG-gerador':>16s}{'vs NEG-deriva':>16s}{'vs NEG-deslocado':>18s}")
    for col in ("e1", "dissim_global", "dissim_local",
                "psnr_global", "psnr_local"):
        pos = grupos[col]["POS-pintado"]
        cel = []
        for g in ("NEG-gerador", "NEG-deriva", "NEG-deslocado"):
            neg = grupos[col][g]
            cel.append(f"{auc(neg, pos):.3f}" if neg and pos else "--")
        print(f"{col:16s}{cel[0]:>16s}{cel[1]:>16s}{cel[2]:>18s}")

    print("\nAUC por marca, POS-pintado contra cada negativo")
    marcas = sorted({l["marca"] for l in linhas})
    for g in ("NEG-gerador", "NEG-deriva", "NEG-deslocado"):
        print(f"\n  contra {g}")
        print(f"  {'marca':14s}" + "".join(f"{c:>16s}" for c in
              ("e1", "dissim_local", "psnr_local")))
        for m in marcas:
            cel = []
            for col in ("e1", "dissim_local", "psnr_local"):
                pos = [float(l[col]) for l in linhas
                       if l["grupo"] == "POS-pintado" and l["marca"] == m
                       and l[col] != ""]
                neg = [float(l[col]) for l in linhas
                       if l["grupo"] == g and l["marca"] == m and l[col] != ""]
                cel.append(f"{auc(neg, pos):.3f}" if neg and pos else "--")
            print(f"  {m:14s}" + "".join(f"{c:>16s}" for c in cel))

    print(f"\ncsv: {a.csv_out}")


if __name__ == "__main__":
    main()
