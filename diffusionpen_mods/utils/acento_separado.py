"""Letra e diacritico separados no condicionamento de texto.

No DiffusionPen o texto passa inteiro pelo CANINE-C. O CANINE e contextual:
trocar "a" por "á" muda o vetor daquela posicao (tanto quanto trocar a letra),
mas tambem o das letras vizinhas (~0,4 de distancia relativa) e o das posicoes
de preenchimento (~0,8) -- diagnostico/diag_texto_por_posicao.py. Uma palavra
acentuada chega ao UNet como uma entrada diferente em todas as 40 posicoes.

Aqui o CANINE recebe sempre o ESQUELETO ("implantacao"), a mesma entrada que o
modelo do IAM conhece, e o diacritico entra por fora: um vetor por tipo de
sinal, somado so na posicao da letra que o leva. O vetor comeca em zero, entao
no passo zero o modelo e exatamente o original pedindo a palavra sem acento.

    palavra      i m p l a n t a ç ã o
    CANINE ve    i m p l a n t a c a o
    acentos      0 0 0 0 0 0 0 0 5 4 0      (0 nada, 1 agudo, 2 grave,
                                             3 circunflexo, 4 til, 5 cedilha)

Usado pelo train.py com --acento_separado e pelos scripts de geracao que
carregam um checkpoint treinado assim (a chave CHAVE existe no state_dict).
"""
import types
import unicodedata

import torch
import torch.nn as nn

SINAIS = {"́": 1, "̀": 2, "̂": 3, "̃": 4, "̧": 5}
N_SINAIS = 6
CHAVE = "module.text_encoder.acento.weight"


def separar(texto):
    """'ação' -> ('acao', [0, 5, 4, 0]): esqueleto e sinal de cada letra."""
    letras, sinais = [], []
    for c in unicodedata.normalize("NFD", texto):
        if unicodedata.category(c) == "Mn":
            if c in SINAIS and sinais:
                sinais[-1] = SINAIS[c]
        else:
            letras.append(c)
            sinais.append(0)
    return "".join(letras), sinais


def tokenizar(tokenizer, textos, max_len):
    """Tokeniza os esqueletos e acrescenta 'acentos' (B, max_len) ao resultado."""
    pares = [separar(t) for t in textos]
    saida = tokenizer([e for e, _ in pares], padding="max_length", truncation=True,
                      return_tensors="pt", max_length=max_len)
    ids = torch.zeros(len(textos), max_len, dtype=torch.long)
    for k, (_, s) in enumerate(pares):
        s = s[:max_len - 2]                                  # posicao 0 e o [CLS]; sobra o [SEP]
        ids[k, 1:1 + len(s)] = torch.tensor(s, dtype=torch.long)
    saida["acentos"] = ids
    return saida


class TextoComAcento(nn.Module):
    """No lugar do codificador de texto do UNet.

    O CANINE fica em `self.module`, entao as chaves dele no state_dict nao
    mudam (text_encoder.module.*) e os pesos do IAM carregam como antes. A
    unica chave nova e text_encoder.acento.weight.
    """

    def __init__(self, canine):
        super().__init__()
        self.module = canine
        self.acento = nn.Embedding(N_SINAIS, canine.config.hidden_size)
        nn.init.zeros_(self.acento.weight)

    def forward(self, acentos=None, **entradas):
        h = self.module(**entradas).last_hidden_state
        if acentos is not None:
            h = h + self.acento(acentos)
        return types.SimpleNamespace(last_hidden_state=h)


def instalar(unet):
    """Troca o text_encoder do UNet (embrulhado ou nao) e devolve o modulo novo."""
    alvo = unet.module if hasattr(unet, "module") else unet
    atual = alvo.text_encoder
    canine = atual.module if hasattr(atual, "module") else atual
    alvo.text_encoder = TextoComAcento(canine).to(next(canine.parameters()).device)
    return alvo.text_encoder


class Tokenizador:
    """Embrulha o tokenizador para os geradores: mesma chamada, esqueleto + 'acentos'."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, textos, max_length=40, **_):
        return tokenizar(self.tokenizer, list(textos), max_length)
