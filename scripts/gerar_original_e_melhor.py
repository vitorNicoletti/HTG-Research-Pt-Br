"""Gera, na CPU, algumas palavras do protocolo de validacao com o modelo
original do IAM e com o melhor modelo (zona vogais, 40 epocas), com os mesmos
20 escritores do iam_test, as mesmas referencias e a mesma semente do
scripts/avaliar_pt.py. Alimenta a figura lado a lado de
scripts/figuras_artigo.py.

So palavras de validacao; nunca o split de teste. Lote de 20 escritores por
chamada, como no protocolo, para o ruido inicial ser o mesmo.

    python scripts/gerar_original_e_melhor.py --saida arquivo/figuras_artigo
"""
import argparse
import os
import random
import sys

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "scripts"))

from avaliar_pt import N_ESCRITORES  # noqa: E402

# uma palavra por classe de sinal, todas do split de validacao
PALAVRAS = ["haverá", "opressão", "relâmpagos", "caçadora", "prestação",
            "havera", "opressao", "relampagos", "cacadora", "prestacao"]
MODELOS = {
    "original": os.path.join(RAIZ, "DiffusionPen/diffusionpen_iam_model_path/models/ema_ckpt.pt"),
    "zona_vogais_40ep": os.path.join(RAIZ, "modelos/zona_vogais_40ep.pt"),
}
SEMENTE = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", required=True)
    ap.add_argument("--style", default=os.path.join(RAIZ, "DiffusionPen/style_models/iam_style_diffusionpen.pth"))
    a = ap.parse_args()

    import torch
    from PIL import Image
    from acentos_sinteticos.vocabulario import Vocabulario
    from comum.diffusionpen import EscritoresIAM, GeradorDiffusionPen

    Vocabulario().conferir(PALAVRAS, "val", "figura do artigo")
    teste = EscritoresIAM("iam_test.txt")
    escritores = random.Random(0).sample(teste.escritores, N_ESCRITORES)
    torch.set_num_threads(os.cpu_count())
    for rot, ckpt in MODELOS.items():
        ger = GeradorDiffusionPen(ckpt, a.style, "cpu")
        pasta = os.path.join(a.saida, rot)
        os.makedirs(pasta, exist_ok=True)
        for palavra in PALAVRAS:
            if all(os.path.exists(os.path.join(pasta, f"{palavra}_{e}_{SEMENTE}.png")) for e in escritores):
                continue
            refs = [teste.referencias(e, random.Random(SEMENTE * 1000 + j)) for j, e in enumerate(escritores)]
            imgs = ger.gerar([palavra] * len(escritores), refs, SEMENTE)
            for e, img in zip(escritores, imgs):
                g = (img.mean(0).numpy() * 255).astype(np.uint8)
                Image.fromarray(g).save(os.path.join(pasta, f"{palavra}_{e}_{SEMENTE}.png"))
            print(f"  {rot}: {palavra}", flush=True)
        del ger


if __name__ == "__main__":
    main()
