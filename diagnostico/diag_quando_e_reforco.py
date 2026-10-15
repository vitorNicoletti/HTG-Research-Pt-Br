"""Dois testes de geracao, sem treino, sobre o acento que ignora o texto.

Em diag_casas_trocadas.py a maioria das amostras saiu igual com "módulo" e com
"modulo": quem decide o acento parece ser o par escritor+ruido, nao o texto.

TESTE 1 -- QUANDO o acento e decidido. A geracao tem 50 passos, do ruido puro
ate a imagem. Troca-se o texto no meio do caminho:
    A>E@K  texto acentuado nos K primeiros passos, esqueleto no resto
    E>A@K  o contrario
Se a marca segue o texto do comeco, a decisao e tomada em ruido alto.

TESTE 2 -- REFORCO do texto na geracao. O modelo preve o ruido com o texto
acentuado (eA) e com o esqueleto (eE). Em vez de usar uma previsao so, usa-se
    pedindo acento:      eE + w (eA - eE)
    pedindo sem acento:  eA + w (eE - eA)
w = 1 e a geracao normal; w > 1 exagera a parte da previsao que depende do
diacritico. Se o modelo sabe algo sobre o acento e so usa pouco, a marca passa
a seguir o texto. Custa duas chamadas do modelo por passo.

Mesmo escritor e mesmo ruido em todas as versoes de uma coluna.

    python diagnostico/diag_quando_e_reforco.py --ckpt model_iam_pt_peso5/models/ema_bloco_16ep.pt \\
        --saida diagnostico/resultados/quando_e_reforco
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
from diag_casas_trocadas import TextoFixo, esqueleto, fonte  # noqa: E402

PALAVRAS = ["haverá", "módulo", "provável", "câmera", "fogão"]
# (rotulo, tipo, parametro, o texto pede acento?)
VERSOES = [("normal", "fixo", "A", True), ("normal", "fixo", "E", False),
           ("troca no passo 10", "troca", ("A", "E", 10), None), ("troca no passo 10", "troca", ("E", "A", 10), None),
           ("troca no passo 25", "troca", ("A", "E", 25), None), ("troca no passo 25", "troca", ("E", "A", 25), None),
           ("reforco 3", "reforco", ("A", 3.0), True), ("reforco 6", "reforco", ("A", 6.0), True),
           ("reforco 3", "reforco", ("E", 3.0), False), ("reforco 6", "reforco", ("E", 6.0), False)]


def legenda(v, palavra, esq):
    rot, tipo, par, _ = VERSOES[v]
    nome = {"A": palavra, "E": esq}
    if tipo == "fixo":
        return f"texto: {nome[par]}", ""
    if tipo == "troca":
        a, b, k = par
        return f"{nome[a]} nos {k} primeiros passos", f"   depois {nome[b]}"
    alvo, w = par
    return f"pede {nome[alvo]}", f"   reforco {w:g}x contra {nome['E' if alvo == 'A' else 'A']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--style", default="DiffusionPen/style_models/iam_style_diffusionpen.pth")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--escritores", type=int, default=4)
    ap.add_argument("--sementes", type=int, nargs="+", default=[42, 43])
    ap.add_argument("--palavras", nargs="+", default=PALAVRAS)
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
    escritores = random.Random(0).sample(completos, a.escritores)     # os mesmos de diag_casas_trocadas.py
    colunas = [(s, e) for s in a.sementes for e in escritores]
    n = len(colunas)
    print(f"escritores do iam_test: {escritores}; {n} colunas por versao", flush=True)

    # ruido e referencias de estilo: iguais para todas as palavras e versoes
    ruido = torch.cat([torch.randn(len(escritores), *g.lat_shape, generator=torch.Generator().manual_seed(s))
                       for s in a.sementes])
    with torch.no_grad():
        estilo = g.feat(torch.stack([t for s in a.sementes for j, e in enumerate(escritores)
                                     for t in teste.referencias(e, random.Random(s * 1000 + j))]))
    y = torch.zeros(n, dtype=torch.long)
    passos = list(g.ddim.timesteps)

    def vetores(p):
        t = g.tokenizer([p], padding="max_length", truncation=True, return_tensors="pt", max_length=g.texto_max_len)
        with torch.no_grad():
            return codificador(**t).last_hidden_state.repeat(n, 1, 1), t

    @torch.no_grad()
    def prever(x, t, H, tk):
        troca.fixo = H
        return g.ema(x, timesteps=t.repeat(n).long(), context=tk, y=y, style_extractor=estilo)

    @torch.no_grad()
    def gerar(tipo, par, H, tk):
        x = ruido.clone()
        for i, t in enumerate(passos):
            if tipo == "fixo":
                e = prever(x, t, H[par], tk)
            elif tipo == "troca":
                e = prever(x, t, H[par[0] if i < par[2] else par[1]], tk)
            else:
                alvo, w = par
                outro = "E" if alvo == "A" else "A"
                e_alvo, e_outro = prever(x, t, H[alvo], tk), prever(x, t, H[outro], tk)
                e = e_outro + w * (e_alvo - e_outro)
            x = g.ddim.step(e, t, x).prev_sample
        img = g.vae.decode(x / 0.18215).sample
        return ((img.clamp(-1, 1) + 1) / 2).float()

    os.makedirs(a.saida, exist_ok=True)
    linhas = ["palavra\tversao\ttipo\tparametro\tescritor\tsemente\tmarca_acima\tx_marcas"]
    conta = {}
    for palavra in a.palavras:
        esq = esqueleto(palavra)
        (HA, tk), (HE, _) = vetores(palavra), vetores(esq)
        H = {"A": HA, "E": HE}
        pasta = os.path.join(a.saida, "paineis", esq)
        os.makedirs(pasta, exist_ok=True)
        for v, (rot, tipo, par, _) in enumerate(VERSOES):
            imgs = gerar(tipo, par, H, tk)
            for (s, e), img in zip(colunas, imgs):
                cinza = (img.mean(0).numpy() * 255).astype(np.uint8)
                acima, _, caixas, geo = marcas(cinza)
                xs = []
                if geo is not None:
                    x0, x1 = geo.caixa[0], geo.caixa[1]
                    xs = [round((x + w / 2 - x0) / max(1, x1 - x0), 2) for x, yy, w, h, onde in caixas if onde == "acima"]
                linhas.append(f"{palavra}\t{v + 1}\t{tipo}\t{par}\t{e}\t{s}\t{int(acima > 0)}\t{xs}")
                conta.setdefault(v, []).append(int(acima > 0))
                Image.fromarray(cinza).save(os.path.join(pasta, f"{v + 1}_{e}_{s}.png"))
            print(f"{palavra:<10} {' / '.join(x for x in legenda(v, palavra, esq) if x):<60} "
                  f"marca em {sum(conta[v][-n:])}/{n}", flush=True)

        # duas folhas por palavra: (1) quando, (2) reforco; as linhas 1 e 2 entram nas duas
        for nome, idx in (("quando", [0, 1, 2, 3, 4, 5]), ("reforco", [0, 6, 7, 1, 8, 9])):
            LEG, CAB = 290, 22
            folha = Image.new("L", (LEG + n * 258, len(idx) * 66 + CAB), 255)
            d, f = ImageDraw.Draw(folha), fonte()
            d.text((4, 3), f"{palavra} / {esq}    mesma coluna = mesmo escritor do iam_test e mesmo ruido", fill=0, font=f)
            for r, v in enumerate(idx):
                l1, l2 = legenda(v, palavra, esq)
                d.text((4, CAB + r * 66 + (16 if l2 else 24)), l1, fill=0, font=f)
                d.text((4, CAB + r * 66 + 32), l2, fill=0, font=f)
                for c, (s, e) in enumerate(colunas):
                    folha.paste(Image.open(os.path.join(pasta, f"{v + 1}_{e}_{s}.png")), (LEG + c * 258, CAB + r * 66))
            folha.save(os.path.join(a.saida, f"{nome}_{esq}.png"))

    with open(os.path.join(a.saida, "paineis.tsv"), "w", encoding="utf-8") as f:
        f.write("\n".join(linhas) + "\n")
    print("\nmarca acima do corpo, todas as palavras")
    for v in range(len(VERSOES)):
        print(f"  {' / '.join(x.strip() for x in legenda(v, 'ACENTUADA', 'esqueleto') if x):<62} "
              f"{np.mean(conta[v]):.0%}  ({sum(conta[v])}/{len(conta[v])})")
    print("FIM")


if __name__ == "__main__":
    main()
