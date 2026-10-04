"""Base de acentos sinteticos do IAM (scripts/gerar_base_acentos.py) para o train.py.

Cada amostra de treino e:
  - uma palavra ACENTUADA da base (imagens/NNNNNN.png, rotulo com diacritico), ou
  - uma palavra ORIGINAL do IAM (utils/splits_words/iam_train_val.txt), sem
    acento -- a fracao args.iam_originais delas entra junto. Mantem a
    distribuicao em que o modelo foi treinado (menos esquecimento) e evita que
    ele aprenda "sempre ponha algum acento". Como cada palavra acentuada saiu
    de uma original, o par (sem acento, com acento) da mesma imagem e o
    contraste mais direto possivel.

As 5 referencias de estilo saem sempre de palavras ORIGINAIS do mesmo escritor,
com mais de 3 letras (mesmo criterio do IAMDataset): o extrator de estilo ve a
caligrafia, e o acento so pode vir do texto.

O pre-processamento e o do utils/iam_dataset.py, copiado sem mudanca: altura
64 preservando o aspecto, centralizado em 256 (ou encolhido de 20 em 20 px ate
caber). Sem normalizacao de contraste -- e o que o modelo viu no treino do IAM.
"""

import json
import os
import random
import string
import sys

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from utils.auxilary_functions import image_resize_PIL, centered_PIL

NUM_STYLE_IMGS = 5
CLONE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_ORIGINAIS = os.path.join(CLONE, "utils", "splits_words", "iam_train_val.txt")
WRITERS_DICT = os.path.join(CLONE, "writers_dict_train.json")
SEMENTE_ORIGINAIS = 0   # sorteio da fracao de originais (reprodutivel)


def preprocessar_iam(img, transcr):
    """Exatamente o laco de utils/iam_dataset.py (main_loader)."""
    if transcr in string.punctuation:
        return centered_PIL(img, (64, 256), border_value=255.0)
    (w, h) = img.size
    img = img.resize((int(w * 64 / h), 64))
    (w, h) = img.size
    if w < 256:
        return ImageOps.pad(img, size=(256, 64), color="white")
    while w > 256:
        img = image_resize_PIL(img, width=w - 20)
        (w, h) = img.size
    return centered_PIL(img, (64, 256), border_value=255.0)


class IAMAcentuadoDataset(Dataset):
    def __init__(self, basefolder, transforms=None, args=None):
        self.transforms = transforms
        self.args = args
        self.raiz_iam = os.environ.get("IAM_IMAGES", os.path.join(CLONE, "iam_data", "words"))
        fracao = float(getattr(args, "iam_originais", 1.0))
        if not 0.0 <= fracao <= 1.0:
            raise ValueError(f"iam_originais fora de [0, 1]: {fracao}")

        # amostras acentuadas
        sinteticas = []
        with open(os.path.join(basefolder, "split.txt"), encoding="utf-8") as f:
            for l in f:
                p = l.rstrip("\n").split(",")
                if len(p) >= 3:
                    sinteticas.append((os.path.join(basefolder, p[0]), p[1], ",".join(p[2:])))

        # Base portuguesa: o resumo.json aponta o vocabulario particionado.
        # Todo rotulo tem de ser do split de treino -- palavra de validacao ou
        # teste aqui e vazamento, e o treino para em vez de seguir.
        caminho_resumo = os.path.join(basefolder, "resumo.json")
        if os.path.isfile(caminho_resumo):
            with open(caminho_resumo, encoding="utf-8") as f:
                vocab = json.load(f).get("vocabulario")
            if vocab:
                raiz = os.path.dirname(CLONE)
                sys.path.insert(0, raiz)
                from acentos_sinteticos.vocabulario import Vocabulario
                Vocabulario(os.path.join(raiz, vocab)).conferir(
                    [t for _, _, t in sinteticas], "treino", f"leitor da base {basefolder}")
                print(f"IAM acentuado: {len(sinteticas)} rotulos conferidos contra {vocab} (so treino)")

        # originais do IAM: todas servem de referencia de estilo; a fracao
        # pedida tambem entra como amostra de treino
        originais = []
        with open(SPLIT_ORIGINAIS, encoding="utf-8") as f:
            for l in f:
                p = l.rstrip("\n").split(",")
                if len(p) >= 3:
                    originais.append((os.path.join(self.raiz_iam, p[0]), p[1], ",".join(p[2:])))
        k = round(fracao * len(originais))
        treino_orig = random.Random(SEMENTE_ORIGINAIS).sample(originais, k) if k < len(originais) else originais

        # classe do escritor: o mesmo indice do treino do IAM (0..338). Com o
        # extrator de estilo ativo o train.py nao usa essa classe, mas ela
        # tem de ser um int valido.
        if os.path.isfile(WRITERS_DICT):
            with open(WRITERS_DICT, encoding="utf-8") as f:
                self.wid = {k: int(v) for k, v in json.load(f).items()}
        else:
            self.wid = {w: i for i, w in enumerate(sorted({o[1] for o in originais}))}

        self.referencias = {}
        for caminho, escritor, texto in originais:
            if len(texto) > 3:
                self.referencias.setdefault(escritor, []).append(caminho)
        for caminho, escritor, texto in originais:     # escritores so com palavras curtas
            if escritor not in self.referencias:
                self.referencias.setdefault(escritor, []).append(caminho)

        dados = [(c, e, t, True) for c, e, t in sinteticas] + \
                [(c, e, t, False) for c, e, t in treino_orig]
        dados = [d for d in dados if d[1] in self.wid and d[1] in self.referencias]
        max_amostras = getattr(args, "max_samples", 0) or 0
        if max_amostras and len(dados) > max_amostras:
            dados = random.Random(SEMENTE_ORIGINAIS).sample(dados, max_amostras)
        self.data = dados
        self.n_sinteticas = sum(1 for d in dados if d[3])
        self.n_originais = len(dados) - self.n_sinteticas
        print(f"IAM acentuado: {self.n_sinteticas} acentuadas + {self.n_originais} originais "
              f"(fracao {fracao}) = {len(dados)} amostras, "
              f"{len({d[1] for d in dados})} escritores")

    def __len__(self):
        return len(self.data)

    def carregar(self, caminho, transcr):
        # Sem try/except amplo: imagem que nao abre e erro, nao uma imagem
        # branca silenciosa no treino.
        img = Image.open(caminho).convert("RGB")
        img = preprocessar_iam(img, transcr)
        return self.transforms(img) if self.transforms else img

    def __getitem__(self, index):
        caminho, escritor, transcr, _ = self.data[index]
        img = self.carregar(caminho, transcr)
        refs = self.referencias[escritor]
        if len(refs) >= NUM_STYLE_IMGS:
            escolhidas = random.sample(refs, NUM_STYLE_IMGS)
        else:
            escolhidas = random.choices(refs, k=NUM_STYLE_IMGS)
        estilos = torch.stack([self.carregar(r, "palavra") for r in escolhidas])
        cor = self.carregar(random.choice(refs), "palavra")
        # Mesmas posicoes do IAMDataset/BRESSAY_Dataset:
        # 0 imagem, 1 texto, 2 classe do escritor, 3 estilos [5,3,64,256],
        # 4 caminho, 5 cor_im
        return img, transcr, self.wid[escritor], estilos, caminho, cor
