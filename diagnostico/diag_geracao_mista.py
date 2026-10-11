"""O espaco do acento e decidido nos primeiros passos? Dois modelos numa geracao.

O treino com --professor_fora_mascara limpa a letra das palavras acentuadas mas
perde o acento nas curtas, que saem grandes como no modelo original e sem lugar
para o sinal. Hipotese: o tamanho da palavra e decidido nos primeiros passos
(ruido alto), e ali esse modelo decide como o original.

Teste, sem treinar: gerar com o modelo A nos K primeiros dos 50 passos e com o
modelo B no resto, a partir do mesmo ruido e com o mesmo estilo.

    A        so o modelo A
    B        so o modelo B
    A>B@K    A nos K primeiros passos, B no resto
    B>A@K    o contrario, de controle

Com A = professor e B = professor fora da mascara: se "A>B" devolver o acento
das curtas com a letra limpa de B, a hipotese se sustenta.

Mede a marca solta (scripts/medir_marcas.py), mas quem decide e a folha: uma
por palavra, uma fileira por variante, uma coluna por escritor.

    python diagnostico/diag_geracao_mista.py --saida diagnostico/resultados/geracao_mista \\
        --a professor=model_iam_pt_separado_professor/models/ema_ckpt.pt \\
        --b fora=model_iam_pt_professor_fora/models/ema_ckpt.pt --bases iam_pt_alinhado
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

# Do protocolo de validacao (scripts/avaliar_pt.py). As quatro primeiras sao as
# que mais perderam marca com o professor fora da mascara; as duas ultimas
# quase nao mudaram e servem de comparacao.
PALAVRAS = ["rapé", "pás", "pôr", "pensámos", "terço", "sofrerá", "reclusão"]
ESCRITORES = ["183", "201", "280", "305", "318", "321", "510", "517"]
# (primeiro modelo, segundo modelo, passos do primeiro)
VARIANTES = [("A", "A", 50), ("B", "B", 50), ("A", "B", 5), ("A", "B", 10), ("A", "B", 20), ("A", "B", 30), ("B", "A", 20)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="rotulo=ckpt do modelo A")
    ap.add_argument("--b", required=True, help="rotulo=ckpt do modelo B")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--bases", nargs="*", default=[], help="bases de treino, para conferir que os textos nao estao nelas")
    ap.add_argument("--style", default="DiffusionPen/style_models/iam_style_diffusionpen.pth")
    ap.add_argument("--palavras", nargs="+", default=PALAVRAS)
    ap.add_argument("--escritores", nargs="+", default=ESCRITORES)
    ap.add_argument("--semente", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    from PIL import Image, ImageDraw
    from comum.diffusionpen import EscritoresIAM, GeradorDiffusionPen
    from medir_marcas import marcas

    grupos = {esqueleto(p) for p in a.palavras}
    for b in a.bases:
        with open(os.path.join(b, "split.txt"), encoding="utf-8") as f:
            vistos = {esqueleto(l.rstrip("\n").split(",", 2)[2]) for l in f if l.count(",") >= 2}
        if grupos & vistos:
            raise SystemExit(f"VAZAMENTO: textos avaliados presentes na base {b}: {sorted(grupos & vistos)}")

    teste = EscritoresIAM("iam_test.txt")
    for e in a.escritores:
        if e not in teste.por_escritor:
            raise SystemExit(f"escritor {e} fora do iam_test")
        if not all(os.path.isfile(c) for c, _ in teste.por_escritor[e]):
            raise SystemExit(f"escritor {e} sem todas as imagens do IAM nesta maquina")
    refs = [teste.referencias(e, random.Random(0)) for e in a.escritores]

    (rot_a, ck_a), (rot_b, ck_b) = a.a.split("=", 1), a.b.split("=", 1)
    g = {"A": GeradorDiffusionPen(ck_a, a.style, a.device), "B": GeradorDiffusionPen(ck_b, a.style, a.device)}
    nome = {"A": rot_a, "B": rot_b}
    passos = list(g["A"].ddim.timesteps)
    n = len(a.escritores)

    def rotulo(v):
        p, s, k = v
        return nome[p] if p == s else f"{nome[p]} nos {k} primeiros, depois {nome[s]}"

    @torch.no_grad()
    def gerar(palavra, v):
        p, s, k = v
        ga = g["A"]
        x = torch.randn(n, *ga.lat_shape, generator=torch.Generator(device="cpu").manual_seed(a.semente)).to(a.device)
        estilo = ga.feat(torch.stack([t for r in refs for t in r]).to(a.device))   # mesmo extrator nos dois
        texto = {m: g[m].tokenizer([palavra] * n, padding="max_length", truncation=True, return_tensors="pt",
                                   max_length=g[m].texto_max_len).to(a.device) for m in g}
        y = torch.zeros(n, dtype=torch.long, device=a.device)
        for i, t in enumerate(passos):
            m = p if i < k else s
            ruido = g[m].ema(x, timesteps=t.repeat(n).to(a.device).long(), context=texto[m], y=y, style_extractor=estilo)
            x = ga.ddim.step(ruido, t, x).prev_sample
        img = ga.vae.decode(x / 0.18215).sample
        return ((img.clamp(-1, 1) + 1) / 2).float().cpu()

    os.makedirs(a.saida, exist_ok=True)
    f_ = fonte()
    LEG = 300
    linhas = ["palavra\tvariante\tescritor\tmarca"]
    tab = {}
    for palavra in a.palavras:
        folha = Image.new("RGB", (LEG + n * 258, 22 + len(VARIANTES) * 70), "white")
        d = ImageDraw.Draw(folha)
        d.text((4, 3), f"{palavra}    semente {a.semente}; mesma coluna = mesmo escritor e mesmo ruido; barra verde = marca detectada",
               fill="black", font=f_)
        for r, v in enumerate(VARIANTES):
            d.text((4, 22 + r * 70 + 24), rotulo(v), fill="black", font=f_)
            for c, (e, img) in enumerate(zip(a.escritores, gerar(palavra, v))):
                cinza = (img.mean(0).numpy() * 255).astype(np.uint8)
                m = int(marcas(cinza)[0] > 0)
                tab.setdefault(v, []).append(m)
                linhas.append(f"{palavra}\t{rotulo(v)}\t{e}\t{m}")
                x, y = LEG + c * 258, 22 + r * 70
                folha.paste(Image.fromarray(cinza).convert("RGB"), (x, y))
                if m:
                    d.rectangle([x, y + 64, x + 255, y + 67], fill=(0, 150, 0))
        folha.save(os.path.join(a.saida, f"{esqueleto(palavra)}.png"))
        print(f"{palavra}: " + ", ".join(f"{rotulo(v)} {sum(tab[v][-n:])}/{n}" for v in VARIANTES), flush=True)

    with open(os.path.join(a.saida, "paineis.tsv"), "w", encoding="utf-8") as f:
        f.write("\n".join(linhas) + "\n")
    txt = [f"Imagens com marca solta, de {n} escritores, semente {a.semente}. A = {rot_a}, B = {rot_b}.", "",
           "| variante | " + " | ".join(a.palavras) + " |", "|---|" + "---|" * len(a.palavras)]
    for v in VARIANTES:
        txt.append(f"| {rotulo(v)} | " + " | ".join(str(sum(tab[v][i * n:(i + 1) * n])) for i in range(len(a.palavras))) + " |")
    with open(os.path.join(a.saida, "resumo.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(txt) + "\n")
    print("\n".join(txt))


if __name__ == "__main__":
    main()
