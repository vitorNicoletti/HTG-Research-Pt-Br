"""
reconhecedor.py -- eixo E2: le a imagem com TrOCR e devolve o CER.

O TrOCR (microsoft/trocr-base-handwritten) e treinado em ingles manuscrito e
nao le diacriticos do portugues. Isso nao e um defeito para este uso: o eixo
E2 mede INTEGRIDADE DA BASE, e a base e justamente a palavra sem acento. Por
isso alvo e predicao sao dobrados para ASCII antes do CER -- ver dobra_ascii
em metrica.py.

Um numero de E2 so significa alguma coisa ao lado do controle: o mesmo
reconhecedor aplicado a recortes REAIS do BRESSAY. Sem esse piso nao da para
dizer se um CER alto e falha do gerador ou do leitor.
"""

import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrica as M  # noqa: E402

MODELO = "microsoft/trocr-base-handwritten"

# O repo do TrOCR no Hub so tem vocab.json + merges.txt, e o transformers 5.x
# exige um tokenizer "fast" ja serializado -- converter na hora pediria
# sentencepiece/tiktoken, que nao estao neste ambiente. Entao o tokenizer.json
# foi construido uma vez (ByteLevel BPE, igual ao RoBERTa) e versionado aqui.
# Conferido: "nacao" -> [0, 282, 1043, 3853, 2] -> "nacao", e bos/eos/pad
# (0/2/1) batem com o config.json do modelo.
TOKENIZER_LOCAL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "trocr_tokenizer")


class Reconhecedor:
    def __init__(self, device="cpu", modelo=MODELO, max_novos=16):
        from transformers import (AutoImageProcessor, AutoTokenizer,
                                  VisionEncoderDecoderModel)
        self.img_proc = AutoImageProcessor.from_pretrained(modelo)
        self.tok = AutoTokenizer.from_pretrained(TOKENIZER_LOCAL)
        self.model = VisionEncoderDecoderModel.from_pretrained(modelo).to(device)
        self.model.eval()
        self.device = device
        self.max_novos = max_novos

    @torch.no_grad()
    def ler(self, caminhos, batch=8):
        """Lista de caminhos -> lista de strings lidas."""
        saida = []
        for i in range(0, len(caminhos), batch):
            imgs = [Image.open(c).convert("RGB") for c in caminhos[i:i + batch]]
            px = self.img_proc(images=imgs, return_tensors="pt").pixel_values.to(self.device)
            ids = self.model.generate(px, max_new_tokens=self.max_novos)
            saida += self.tok.batch_decode(ids, skip_special_tokens=True)
        return saida

    def cer_ascii(self, caminhos, alvos, batch=8):
        """(lista de leituras, lista de CER dobrado para ASCII)."""
        lidos = self.ler(caminhos, batch)
        cers = [M.cer(M.dobra_ascii(a), M.dobra_ascii(l))
                for a, l in zip(alvos, lidos)]
        return lidos, cers
