"""
anotacao.py -- Passo 6: validacao do instrumento contra anotacao humana.

    preparar : sorteia uma amostra ESTRATIFICADA pelas quatro categorias da
               metrica e escreve uma planilha para anotar, mais uma folha de
               contato com as imagens numeradas.
    kappa    : le a planilha preenchida e calcula o Cohen's kappa entre a
               metrica e a pessoa.

Por que estratificar pelas categorias e nao sortear ao acaso: se 90% das
amostras caem em "degradacao", uma amostra aleatoria quase nao traz exemplos
das outras tres e o kappa fica dominado pela classe maior -- alto por
concordancia trivial, sem dizer nada sobre as decisoes dificeis.

ATENCAO ao ler o kappa: a estratificacao muda as prevalencias, entao o valor
obtido NAO e o kappa que se obteria na distribuicao natural. Ele mede a
concordancia condicionada a esse desenho, que e o que interessa aqui
(a metrica acerta em cada categoria?), e deve ser reportado assim.

A pessoa anota DOIS eixos separados, nunca a categoria pronta:
  acento_presente : 1 se da para ver o diacritico, 0 se nao, ? se nao da
  base_legivel    : 1 se da para ler a palavra ignorando o acento, 0 se nao
Anotar a categoria diretamente induziria a pessoa a reproduzir a logica da
metrica em vez de julgar a imagem.
"""
import argparse
import csv
import os
import random
import sys

import numpy as np

CATEGORIAS = ["acerto", "omissao_do_acento",
              "degradacao_com_diacritico", "degradacao_sem_diacritico"]


def preparar(a):
    with open(a.csv, encoding="utf-8") as f:
        linhas = [r for r in csv.DictReader(f)]
    if not linhas or not linhas[0].get("categoria"):
        sys.exit("o CSV nao tem coluna 'categoria' preenchida: rode o "
                 "avaliar.py com --limiar-e1 e --com-e2")
    por_cat = {}
    for r in linhas:
        por_cat.setdefault(r["categoria"], []).append(r)
    rng = random.Random(a.seed)
    amostra = []
    for c in CATEGORIAS:
        v = por_cat.get(c, [])
        rng.shuffle(v)
        amostra += v[:a.por_categoria]
        print(f"{c:28s} disponiveis={len(v):5d} sorteadas={len(v[:a.por_categoria])}")
    rng.shuffle(amostra)   # embaralha para a pessoa nao ver blocos por categoria

    campos = ["id", "arquivo", "palavra_alvo", "marca",
              "acento_presente", "base_legivel", "observacao",
              "_metrica_categoria", "_metrica_e1_escore", "_metrica_e2_cer"]
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        for i, r in enumerate(amostra):
            w.writerow({
                "id": i, "arquivo": r["arquivo"], "palavra_alvo": r["palavra"],
                "marca": r["marca"], "acento_presente": "", "base_legivel": "",
                "observacao": "",
                "_metrica_categoria": r["categoria"],
                "_metrica_e1_escore": r["e1_escore"],
                "_metrica_e2_cer": r.get("e2_cer", ""),
            })
    print(f"\nplanilha: {a.out}  ({len(amostra)} linhas)")
    print("as colunas _metrica_* existem so para o calculo do kappa depois.")
    print("QUEM ANOTA NAO DEVE VE-LAS: use a folha de contato, que mostra so")
    print("a imagem e o id.")

    if a.contato:
        from PIL import Image, ImageDraw, ImageFont
        # A fonte padrao do PIL nao tem os glifos acentuados, e sem eles o
        # rotulo mostra "elucida##o" -- justamente a informacao de que quem
        # anota precisa. Procura uma TTF com Latin-1; se nao achar, cai na
        # fonte padrao e escreve o alvo sem acento, dizendo qual e a marca.
        fonte, tem_acento = None, False
        for cam in ("/run/current-system/sw/share/X11/fonts/DejaVuSans.ttf",
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                    "/usr/share/fonts/TTF/DejaVuSans.ttf"):
            if os.path.exists(cam):
                fonte, tem_acento = ImageFont.truetype(cam, 12), True
                break
        if fonte is None:
            import glob as _g
            achados = _g.glob("/nix/store/*/share/fonts/truetype/DejaVuSans.ttf")
            if achados:
                fonte, tem_acento = ImageFont.truetype(achados[0], 12), True
        if fonte is None:
            fonte = ImageFont.load_default()
        cols, lado, alt = 4, 256, 64
        rot = 16
        linhas_n = (len(amostra) + cols - 1) // cols
        folha = Image.new("L", (cols * (lado + 6), linhas_n * (alt + rot + 6)), 255)
        d = ImageDraw.Draw(folha)
        for i, r in enumerate(amostra):
            im = Image.open(os.path.join(a.dir, r["arquivo"])).convert("L")
            x = (i % cols) * (lado + 6)
            y = (i // cols) * (alt + rot + 6)
            folha.paste(im, (x, y + rot))
            import metrica as _M
            alvo = r["palavra"] if tem_acento else _M.sem_acento(r["palavra"])
            d.text((x + 2, y + 2),
                   f"id={i}  alvo={alvo}  ({r['marca']})", fill=0, font=fonte)
        folha.save(a.contato)
        print("folha de contato:", a.contato)


def kappa_cohen(a1, a2):
    """Cohen's kappa nao ponderado."""
    cats = sorted(set(a1) | set(a2))
    idx = {c: i for i, c in enumerate(cats)}
    n = len(a1)
    m = np.zeros((len(cats), len(cats)))
    for x, y in zip(a1, a2):
        m[idx[x], idx[y]] += 1
    po = np.trace(m) / n
    pe = float((m.sum(axis=0) / n * (m.sum(axis=1) / n)).sum())
    k = (po - pe) / (1 - pe) if pe != 1 else float("nan")
    return k, po, pe, cats, m


def calcular_kappa(a):
    with open(a.csv, encoding="utf-8") as f:
        linhas = [r for r in csv.DictReader(f)]
    usaveis = [r for r in linhas
               if r["acento_presente"] in ("0", "1")
               and r["base_legivel"] in ("0", "1")]
    print(f"{len(usaveis)} de {len(linhas)} linhas anotadas nos dois eixos")
    if not usaveis:
        sys.exit("nenhuma linha anotada")

    def cat_humana(r):
        p = r["acento_presente"] == "1"
        b = r["base_legivel"] == "1"
        return ("acerto" if (p and b) else
                "omissao_do_acento" if (not p and b) else
                "degradacao_com_diacritico" if p else
                "degradacao_sem_diacritico")

    hum = [cat_humana(r) for r in usaveis]
    met = [r["_metrica_categoria"] for r in usaveis]
    k, po, pe, cats, m = kappa_cohen(hum, met)
    print(f"\nCATEGORIA (4 classes): kappa = {k:.3f}  "
          f"concordancia observada = {po:.3f}  esperada ao acaso = {pe:.3f}")
    print("\nmatriz de confusao (linha = humano, coluna = metrica):")
    larg = max(len(c) for c in cats) + 2
    print(" " * larg + "".join(f"{c[:12]:>14s}" for c in cats))
    for i, c in enumerate(cats):
        print(f"{c:{larg}s}" + "".join(f"{int(m[i,j]):14d}" for j in range(len(cats))))

    # os dois eixos separados dizem ONDE esta a discordancia
    for nome, col, chave in (("E1 (presenca do acento)", "acento_presente", "e1"),
                             ("E2 (base legivel)", "base_legivel", "e2")):
        h = [r[col] for r in usaveis]
        if chave == "e1":
            mm = ["1" if r["_metrica_categoria"] in
                  ("acerto", "degradacao_com_diacritico") else "0" for r in usaveis]
        else:
            mm = ["1" if r["_metrica_categoria"] in
                  ("acerto", "omissao_do_acento") else "0" for r in usaveis]
        k2, po2, pe2, _, _ = kappa_cohen(h, mm)
        print(f"\n{nome}: kappa = {k2:.3f}  concordancia = {po2:.3f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("preparar")
    p1.add_argument("--csv", required=True, help="saida do avaliar.py")
    p1.add_argument("--dir", required=True, help="pasta das imagens")
    p1.add_argument("--out", required=True)
    p1.add_argument("--contato", default=None)
    p1.add_argument("--por-categoria", type=int, default=25)
    p1.add_argument("--seed", type=int, default=42)
    p1.set_defaults(func=preparar)
    p2 = sub.add_parser("kappa")
    p2.add_argument("--csv", required=True, help="planilha preenchida")
    p2.set_defaults(func=calcular_kappa)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
