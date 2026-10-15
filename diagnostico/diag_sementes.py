"""A marca depende do estilo ou do sorteio? Mesma palavra, mesmo estilo, muitas sementes.

Para cada modelo gera cada texto com cada escritor N vezes. As imagens de
referencia do escritor ficam FIXAS; entre uma geracao e outra so muda o ruido
inicial. Se a marca indevida fosse do estilo, um escritor daria marca em quase
todas as sementes e outro em quase nenhuma. Se fosse do sorteio, todos dariam
a mesma fracao.

Mede, por texto e escritor, a fracao das sementes com marca solta acima do
corpo (scripts/medir_marcas.py). Os textos sao acentuadas da validacao, os
esqueletos delas e palavras sem acento. As folhas mostram todas as sementes,
com barra verde onde o detector viu marca, para conferir no olho ONDE ela caiu.

    python diagnostico/diag_sementes.py --saida diagnostico/resultados/sementes \\
        --modelo professor=model_iam_pt_separado_professor/models/ema_ckpt.pt \\
        --bases iam_pt_alinhado --device cuda:0
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

# Todas do protocolo de validacao (scripts/avaliar_pt.py).
ACENTUADAS = ["sofrerá", "estarás", "reclusão", "relâmpagos", "caçadora"]
SEM_ACENTO = ["sultan", "travas", "central"]
# Escritores do iam_test: os tres com mais e os tres com menos marca indevida
# no modelo peso 5 (saidas/diffusionpen/fine_tune_iam_pt/avaliacao_val).
ESCRITORES = ["537", "517", "583", "318", "555", "547"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", action="append", required=True, help="rotulo=ckpt")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--bases", nargs="*", default=[], help="bases de treino, para conferir que os textos nao estao nelas")
    ap.add_argument("--style", default="DiffusionPen/style_models/iam_style_diffusionpen.pth")
    ap.add_argument("--escritores", nargs="+", default=ESCRITORES)
    ap.add_argument("--acentuadas", nargs="*", default=ACENTUADAS)
    ap.add_argument("--sem_acento", nargs="*", default=SEM_ACENTO)
    ap.add_argument("--sementes", type=int, default=50, help="quantas; vao de 1000 a 1000+N-1")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    from PIL import Image, ImageDraw
    from comum.diffusionpen import EscritoresIAM, GeradorDiffusionPen
    from medir_marcas import marcas

    textos = [(p, "acentuada") for p in a.acentuadas] + [(esqueleto(p), "esqueleto") for p in a.acentuadas] + \
             [(p, "sem_acento") for p in a.sem_acento]
    grupos = {esqueleto(t) for t, _ in textos}
    for b in a.bases:
        with open(os.path.join(b, "split.txt"), encoding="utf-8") as f:
            vistos = {esqueleto(l.rstrip("\n").split(",", 2)[2]) for l in f if l.count(",") >= 2}
        if grupos & vistos:
            raise SystemExit(f"VAZAMENTO: textos avaliados presentes na base {b}: {sorted(grupos & vistos)}")

    teste = EscritoresIAM("iam_test.txt")
    nomes = {str(e): e for e in teste.escritores}
    falta = [e for e in a.escritores if e not in nomes]
    if falta:
        raise SystemExit(f"escritores fora do iam_test: {falta}")
    escritores = [nomes[e] for e in a.escritores]
    for e in escritores:
        if not all(os.path.isfile(c) for c, _ in teste.por_escritor[e]):
            raise SystemExit(f"escritor {e} sem todas as imagens do IAM nesta maquina")
    # referencias fixas: o estilo nao muda entre as sementes
    refs = {e: teste.referencias(e, random.Random(0)) for e in escritores}
    sementes = list(range(1000, 1000 + a.sementes))
    modelos = [m.split("=", 1) for m in a.modelo]

    def caminho(rot, texto, e, s):
        return os.path.join(a.saida, "paineis", rot, f"{texto}_{e}_{s}.png")

    lote = [(t, e) for t, _ in textos for e in escritores]
    for rot, ckpt in modelos:
        falta = [s for s in sementes if not all(os.path.isfile(caminho(rot, t, e, s)) for t, e in lote)]
        if not falta:
            continue
        os.makedirs(os.path.join(a.saida, "paineis", rot), exist_ok=True)
        g = GeradorDiffusionPen(ckpt, a.style, a.device)   # reconhece checkpoints --acento_separado
        for s in falta:
            # um lote por semente; cada (texto, escritor) recebe a sua fatia do ruido
            for (t, e), img in zip(lote, g.gerar([t for t, _ in lote], [refs[e] for _, e in lote], s)):
                Image.fromarray((img.mean(0).cpu().numpy() * 255).astype(np.uint8)).save(caminho(rot, t, e, s))
            print(f"{rot}: semente {s}", flush=True)
        del g

    linhas = ["modelo\ttexto\ttipo\tescritor\tsemente\tmarca"]
    marca = {}
    for rot, _ in modelos:
        for t, tipo in textos:
            for e in escritores:
                for s in sementes:
                    m = int(marcas(np.asarray(Image.open(caminho(rot, t, e, s)).convert("L")))[0] > 0)
                    marca[(rot, t, e, s)] = m
                    linhas.append(f"{rot}\t{t}\t{tipo}\t{e}\t{s}\t{m}")
    with open(os.path.join(a.saida, "paineis.tsv"), "w", encoding="utf-8") as f:
        f.write("\n".join(linhas) + "\n")

    n = len(sementes)
    txt = [f"Fracao das {n} sementes com marca solta, por texto e escritor. Referencias de estilo fixas.", ""]
    for rot, _ in modelos:
        txt += [f"Modelo {rot}.", "", "| texto | tipo | " + " | ".join(str(e) for e in escritores) + " | todos |",
                "|---|---|" + "---|" * (len(escritores) + 1)]
        for t, tipo in textos:
            fr = [np.mean([marca[(rot, t, e, s)] for s in sementes]) for e in escritores]
            txt.append(f"| {t} | {tipo} | " + " | ".join(f"{x:.0%}" for x in fr) + f" | {np.mean(fr):.0%} |")
        for tipo in ("acentuada", "esqueleto", "sem_acento"):
            ts = [t for t, k in textos if k == tipo]
            if not ts:
                continue
            fr = [np.mean([marca[(rot, t, e, s)] for t in ts for s in sementes]) for e in escritores]
            txt.append(f"| todas | {tipo} | " + " | ".join(f"{x:.0%}" for x in fr) + f" | {np.mean(fr):.0%} |")
        txt.append("")
    with open(os.path.join(a.saida, "resumo.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(txt) + "\n")
    print("\n".join(txt))

    # uma folha por modelo e texto: um bloco por escritor, 10 sementes por fileira
    f_ = fonte()
    POR, LEG = 10, 70
    fil = (n + POR - 1) // POR
    for rot, _ in modelos:
        for t, tipo in textos:
            im = Image.new("RGB", (LEG + POR * 258, 22 + len(escritores) * (fil * 70 + 8)), "white")
            d = ImageDraw.Draw(im)
            d.text((4, 3), f"{rot}: {t} ({tipo})    cada imagem e uma semente; barra verde = marca detectada", fill="black", font=f_)
            for i, e in enumerate(escritores):
                y0 = 22 + i * (fil * 70 + 8)
                d.text((4, y0 + 24), str(e), fill="black", font=f_)
                for k, s in enumerate(sementes):
                    x, y = LEG + (k % POR) * 258, y0 + (k // POR) * 70
                    im.paste(Image.open(caminho(rot, t, e, s)).convert("RGB"), (x, y))
                    if marca[(rot, t, e, s)]:
                        d.rectangle([x, y + 64, x + 255, y + 67], fill=(0, 150, 0))
            im.save(os.path.join(a.saida, f"{rot}_{tipo}_{esqueleto(t)}.png"))


if __name__ == "__main__":
    main()
