"""Palavras do IAM: lista (caminho, escritor, texto) e leitura das imagens."""

import os
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class PalavraIAM:
    caminho: str     # caminho absoluto do PNG
    escritor: str    # id do escritor no IAM (ex.: "000")
    texto: str


def listar(clone, split="iam_train_val.txt"):
    """Palavras de utils/splits_words/<split> do clone do DiffusionPen.

    Cada linha e "a01/a01-000u/a01-000u-00-00.png,000,A".
    """
    arq = os.path.join(clone, "utils", "splits_words", split)
    raiz = os.path.join(clone, "iam_data", "words")
    saida = []
    with open(arq, encoding="utf-8") as f:
        for l in f:
            partes = l.rstrip("\n").split(",")
            if len(partes) < 3:
                continue
            rel, escritor, texto = partes[0], partes[1], ",".join(partes[2:])
            saida.append(PalavraIAM(os.path.join(raiz, rel), escritor, texto))
    return saida


def carregar_cinza(caminho):
    """PNG -> array float32 0..255 (0 = tinta preta)."""
    return np.asarray(Image.open(caminho).convert("L"), dtype=np.float32)
