"""O reforco contra o acento funciona com um contraste INVENTADO?

Em diag_quando_e_reforco.py o vazamento caiu pela metade usando
    eA + w (eE - eA)
onde eA e a previsao com a palavra acentuada verdadeira ("provável") e eE com
o texto pedido ("provavel"). Uma palavra como "tarde" nao tem versao
acentuada, entao aqui o contraste e fabricado a partir do proprio texto:

  1a vogal   agudo so na primeira vogal          tarde -> tárde
  todas      agudo em todas as vogais            tarde -> tárdé
  misto      a->ã, e->ê, o->ó, u->ú em todas     tarde -> tãrdê

Palavras sem acento nenhum e, para comparar com o contraste verdadeiro, dois
esqueletos de palavras acentuadas.

    python diagnostico/diag_reforco_contraste.py --ckpt model_iam_pt_peso5/models/ema_bloco_16ep.pt \\
        --saida diagnostico/resultados/reforco_contraste
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
from diag_casas_trocadas import TextoFixo, fonte  # noqa: E402

SEM_ACENTO = ["tarde", "bolo", "campo", "mundo"]
ESQUELETOS = {"provavel": "provável", "modulo": "módulo"}      # texto pedido -> acentuada verdadeira
AGUDO = {"a": "á", "e": "é", "o": "ó", "u": "ú"}
MISTO = {"a": "ã", "e": "ê", "o": "ó", "u": "ú"}


def contrastes(p):
    vogais = [i for i, c in enumerate(p) if c in AGUDO]
    c = {"1a vogal": p[:vogais[0]] + AGUDO[p[vogais[0]]] + p[vogais[0] + 1:],
         "todas": "".join(AGUDO.get(x, x) for x in p),
         "misto": "".join(MISTO.get(x, x) for x in p)}
    if p in ESQUELETOS:
        c["verdadeiro"] = ESQUELETOS[p]
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--style", default="DiffusionPen/style_models/iam_style_diffusionpen.pth")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--escritores", type=int, default=4)
    ap.add_argument("--sementes", type=int, nargs="+", default=[42, 43])
    ap.add_argument("--reforco", type=float, default=3.0)
    ap.add_argument("--palavras", nargs="+", default=SEM_ACENTO + list(ESQUELETOS))
    a = ap.parse_args()

    from PIL import Image, ImageDraw
    from comum.diffusionpen import EscritoresIAM, GeradorDiffusionPen
    from medir_marcas import marcas

    g = GeradorDiffusionPen(a.ckpt, a.style, "cpu")
    codificador = g.ema.module.text_encoder
    troca = TextoFixo()
    g.ema.module.text_encoder = troca

    teste = EscritoresIAM("iam_test.txt")
    completos = [e for e in teste.escritores if all(os.path.isfile(c) for c, _ in teste.por_escritor[e])]
    escritores = random.Random(0).sample(completos, a.escritores)
    colunas = [(s, e) for s in a.sementes for e in escritores]
    n = len(colunas)
    ruido = torch.cat([torch.randn(len(escritores), *g.lat_shape, generator=torch.Generator().manual_seed(s))
                       for s in a.sementes])
    with torch.no_grad():
        estilo = g.feat(torch.stack([t for s in a.sementes for j, e in enumerate(escritores)
                                     for t in teste.referencias(e, random.Random(s * 1000 + j))]))
    y = torch.zeros(n, dtype=torch.long)
    tk = g.tokenizer(["x"], padding="max_length", truncation=True, return_tensors="pt", max_length=g.texto_max_len)

    def vetores(p):
        t = g.tokenizer([p], padding="max_length", truncation=True, return_tensors="pt", max_length=g.texto_max_len)
        with torch.no_grad():
            return codificador(**t).last_hidden_state.repeat(n, 1, 1)

    @torch.no_grad()
    def gerar(H, Hc=None):
        x = ruido.clone()
        for t in g.ddim.timesteps:
            ts = t.repeat(n).long()
            troca.fixo = H
            e = g.ema(x, timesteps=ts, context=tk, y=y, style_extractor=estilo)
            if Hc is not None:
                troca.fixo = Hc
                ec = g.ema(x, timesteps=ts, context=tk, y=y, style_extractor=estilo)
                e = ec + a.reforco * (e - ec)
            x = g.ddim.step(e, t, x).prev_sample
        img = g.vae.decode(x / 0.18215).sample
        return ((img.clamp(-1, 1) + 1) / 2).float()

    os.makedirs(a.saida, exist_ok=True)
    linhas = ["palavra\tcontraste\ttexto_contraste\tescritor\tsemente\tmarca_acima"]
    conta = {}
    for p in a.palavras:
        H = vetores(p)
        versoes = [("normal", None)] + list(contrastes(p).items())
        tiras = []
        for nome, texto_c in versoes:
            imgs = gerar(H, vetores(texto_c) if texto_c else None)
            paineis = []
            for (s, e), img in zip(colunas, imgs):
                cinza = (img.mean(0).numpy() * 255).astype(np.uint8)
                m = int(marcas(cinza)[0] > 0)
                linhas.append(f"{p}\t{nome}\t{texto_c or ''}\t{e}\t{s}\t{m}")
                conta.setdefault((p in ESQUELETOS, nome), []).append(m)
                paineis.append(Image.fromarray(cinza))
            tiras.append(paineis)
            print(f"{p:<9} {nome:<11} {texto_c or '':<10} marca em {sum(conta[(p in ESQUELETOS, nome)][-n:])}/{n}", flush=True)

        LEG, CAB = 250, 22
        folha = Image.new("L", (LEG + n * 258, len(tiras) * 66 + CAB), 255)
        d, f = ImageDraw.Draw(folha), fonte()
        d.text((4, 3), f"pedido: {p}    mesma coluna = mesmo escritor do iam_test e mesmo ruido", fill=0, font=f)
        for r, ((nome, texto_c), paineis) in enumerate(zip(versoes, tiras)):
            d.text((4, CAB + r * 66 + 24), f"texto: {p}" if not texto_c
                   else f"reforco {a.reforco:g}x contra {texto_c}", fill=0, font=f)
            for c, im in enumerate(paineis):
                folha.paste(im, (LEG + c * 258, CAB + r * 66))
        folha.save(os.path.join(a.saida, f"{p}.png"))

    with open(os.path.join(a.saida, "paineis.tsv"), "w", encoding="utf-8") as f:
        f.write("\n".join(linhas) + "\n")
    for grupo, titulo in ((False, "palavras sem acento nenhum"), (True, "esqueletos de palavras acentuadas")):
        print(f"\nmarca acima do corpo -- {titulo}")
        for (gr, nome), v in conta.items():
            if gr == grupo:
                print(f"  {nome:<11} {np.mean(v):.0%}  ({sum(v)}/{len(v)})")
    print("FIM")


if __name__ == "__main__":
    main()
