"""Avaliacao pequena, para CPU: poucos textos, poucos escritores, sem leitor.

Nao substitui scripts/avaliar_pt.py (3.200 imagens, leitor CTC, bootstrap):
serve para comparar depressa modelos treinados na CPU. Para cada modelo gera
os mesmos textos com os mesmos escritores do iam_test e o mesmo ruido, e mede

  marca      fracao das imagens com marca solta acima do corpo
             (scripts/medir_marcas.py), por tipo de texto
  pareada    marca na acentuada menos marca no esqueleto, mesma coluna
  mudanca    diferenca media por pixel em relacao ao PRIMEIRO --modelo, so nos
             textos sem acento pedido: quanto a escrita se afastou da referencia.
             Nao e legibilidade; e um aviso de que a letra mudou.

Os paineis ficam em <saida>/paineis/<rotulo>/ e nao sao gerados de novo.

    python diagnostico/avaliar_mini.py --saida diagnostico/resultados/mini \\
        --modelo iam=DiffusionPen/diffusionpen_iam_model_path/models/ema_ckpt.pt \\
        --modelo tudo_2ep=model_cpu_tudo/models/etapa_2ep.pt --bases iam_pt_sub
"""
import argparse
import os
import random
import sys

import numpy as np
import torch

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "scripts"))
sys.path.insert(0, os.path.join(RAIZ, "diagnostico"))
os.chdir(RAIZ)
from diag_casas_trocadas import esqueleto, fonte  # noqa: E402

ACENTUADAS = ["haverá", "módulo", "provável", "câmera", "fogão"]
# Todas do split de VALIDACAO do vocabulario_pt e fora das bases de treino.
# ("campo", "mundo" e "tarde" sao palavras de treino; "você" e do teste.)
SEM_ACENTO = ["nunca", "sempre", "cabelo", "bolo"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", action="append", required=True, help="rotulo=ckpt; o primeiro e a referencia")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--bases", nargs="*", default=[], help="bases de treino, para conferir que os textos nao estao nelas")
    ap.add_argument("--style", default="DiffusionPen/style_models/iam_style_diffusionpen.pth")
    ap.add_argument("--escritores", type=int, default=4)
    ap.add_argument("--sementes", type=int, nargs="+", default=[42, 43])
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)

    from PIL import Image, ImageDraw
    from comum.diffusionpen import EscritoresIAM, GeradorDiffusionPen
    from medir_marcas import marcas

    textos = [(p, "acentuada") for p in ACENTUADAS] + [(esqueleto(p), "esqueleto") for p in ACENTUADAS] + \
             [(p, "sem_acento") for p in SEM_ACENTO]
    grupos = {esqueleto(t) for t, _ in textos}
    for b in a.bases:
        with open(os.path.join(b, "split.txt"), encoding="utf-8") as f:
            vistos = {esqueleto(l.rstrip("\n").split(",", 2)[2]) for l in f if l.count(",") >= 2}
        if grupos & vistos:
            raise SystemExit(f"VAZAMENTO: textos avaliados presentes na base {b}: {sorted(grupos & vistos)}")

    teste = EscritoresIAM("iam_test.txt")
    completos = [e for e in teste.escritores if all(os.path.isfile(c) for c, _ in teste.por_escritor[e])]
    escritores = random.Random(0).sample(completos, a.escritores)      # os mesmos dos outros diag_*.py
    colunas = [(s, e) for s in a.sementes for e in escritores]
    modelos = [m.split("=", 1) for m in a.modelo]

    def caminho(rot, texto, s, e):
        return os.path.join(a.saida, "paineis", rot, f"{texto}_{e}_{s}.png")

    for rot, ckpt in modelos:
        falta = [t for t, _ in textos if not all(os.path.isfile(caminho(rot, t, s, e)) for s, e in colunas)]
        if not falta:
            continue
        os.makedirs(os.path.join(a.saida, "paineis", rot), exist_ok=True)
        g = GeradorDiffusionPen(ckpt, a.style, "cpu")   # reconhece checkpoints --acento_separado
        for t in falta:
            for s in a.sementes:
                refs = [teste.referencias(e, random.Random(s * 1000 + j)) for j, e in enumerate(escritores)]
                for e, img in zip(escritores, g.gerar([t] * len(escritores), refs, s)):
                    Image.fromarray((img.mean(0).numpy() * 255).astype(np.uint8)).save(caminho(rot, t, s, e))
            print(f"{rot}: {t}", flush=True)
        del g

    def painel(rot, t, s, e):
        return np.asarray(Image.open(caminho(rot, t, s, e)).convert("L"))

    ref = modelos[0][0]
    linhas = ["modelo\ttexto\ttipo\tescritor\tsemente\tmarca"]
    tab = []
    for rot, _ in modelos:
        m = {k: [] for k in ("acentuada", "esqueleto", "sem_acento")}
        par, mud = [], []
        for t, tipo in textos:
            for s, e in colunas:
                g_ = painel(rot, t, s, e)
                marca = int(marcas(g_)[0] > 0)
                m[tipo].append(marca)
                linhas.append(f"{rot}\t{t}\t{tipo}\t{e}\t{s}\t{marca}")
                if tipo != "acentuada":
                    mud.append(np.abs(g_.astype(np.float32) - painel(ref, t, s, e)).mean() / 255)
        for p in ACENTUADAS:
            for s, e in colunas:
                par.append(int(marcas(painel(rot, p, s, e))[0] > 0) - int(marcas(painel(rot, esqueleto(p), s, e))[0] > 0))
        tab.append((rot, np.mean(m["acentuada"]), np.mean(m["esqueleto"]), np.mean(m["sem_acento"]), np.mean(par), np.mean(mud)))

    with open(os.path.join(a.saida, "paineis.tsv"), "w", encoding="utf-8") as f:
        f.write("\n".join(linhas) + "\n")
    n = len(colunas)
    txt = [f"{len(ACENTUADAS)} acentuadas + esqueletos + {len(SEM_ACENTO)} sem acento, {n} colunas "
           f"({a.escritores} escritores do iam_test x sementes {a.sementes}). Referencia da mudanca: {ref}.", "",
           "| modelo | marca: acentuada | marca: esqueleto | marca: sem acento | pareada | mudanca da escrita |",
           "|---|---|---|---|---|---|"]
    txt += [f"| {r} | {x:.0%} | {y:.0%} | {z:.0%} | {p:+.0%} | {d:.3f} |" for r, x, y, z, p, d in tab]
    with open(os.path.join(a.saida, "resumo.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(txt) + "\n")
    print("\n".join(txt))

    # folhas: uma por palavra acentuada (acentuada e esqueleto de cada modelo) e uma das sem acento
    f_ = fonte()
    LEG = 230

    def folha(nome, titulo, fileiras):
        im = Image.new("L", (LEG + n * 258, len(fileiras) * 66 + 22), 255)
        d = ImageDraw.Draw(im)
        d.text((4, 3), titulo + "    mesma coluna = mesmo escritor e mesmo ruido", fill=0, font=f_)
        for r, (leg, rot, t) in enumerate(fileiras):
            d.text((4, 22 + r * 66 + 24), leg, fill=0, font=f_)
            for c, (s, e) in enumerate(colunas):
                im.paste(Image.open(caminho(rot, t, s, e)), (LEG + c * 258, 22 + r * 66))
        im.save(os.path.join(a.saida, nome))

    for p in ACENTUADAS:
        folha(f"{esqueleto(p)}.png", f"{p} / {esqueleto(p)}",
              [(f"{rot}: {t}", rot, t) for rot, _ in modelos for t in (p, esqueleto(p))])
    for p in SEM_ACENTO:
        folha(f"sem_acento_{p}.png", p, [(f"{rot}: {p}", rot, p) for rot, _ in modelos])


if __name__ == "__main__":
    main()
