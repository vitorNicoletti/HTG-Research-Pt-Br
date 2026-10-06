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

Peso no acento (args.peso_acento != 1): cada amostra ganha um 7o elemento, a
mascara (8, 32) do acento no latente. Ela sai da diferenca entre a acentuada e
o seu par (a mesma imagem antes dos sinais), por isso exige base com pares
alinhados (resumo.json com pares_alinhados). A acentuada e o par recebem a
MESMA mascara: na acentuada o peso cobra desenhar o sinal, no par cobra nao
desenhar. Amostras sem par (sem_acento, originais do IAM) recebem zeros.

Peso na zona vazia (args.peso_zona != 1): 8o elemento, a mascara (8, 32) das
celulas vazias acima/abaixo do corpo da palavra (mascara_zona), onde tinta seria
um acento que o texto nao pede ("havera" com acento no primeiro a). Calculada
na imagem SEM sinal: o par para acentuada e par (as duas recebem a mesma, sem
as celulas do acento, que ja tem o peso_acento), a propria imagem para
sem_acento e para os originais do IAM -- na base de acentos sobre o IAM real
as palavras sem acento sao justamente os originais.
"""

import json
import os
import random
import string
import sys

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from utils.auxilary_functions import image_resize_PIL, centered_PIL

NUM_STYLE_IMGS = 5
CLONE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_ORIGINAIS = os.path.join(CLONE, "utils", "splits_words", "iam_train_val.txt")
WRITERS_DICT = os.path.join(CLONE, "writers_dict_train.json")
SEMENTE_ORIGINAIS = 0   # sorteio da fracao de originais (reprodutivel)
LIMIAR_PIXEL = 40       # diferenca de cinza (0..255) que conta como acento (= diag_peso_acento.py)
FORMA_LATENTE = (8, 32) # 64x256 pelo VAE do SD (fator 8)
TOL_CORPO = 0.15        # folga do corpo na zona vazia, em alturas-x (= TOL de scripts/medir_marcas.py)
FOLGA_TINTA = 2         # px de dilatacao da tinta: celula vizinha de traco nao e zona vazia
ALCANCE_ZONA = 1.0      # altura da zona acima/abaixo do corpo, em alturas-x: onde acento e cedilha caem

# acentos_sinteticos (geometria, vocabulario) fica na raiz do repositorio
if os.path.dirname(CLONE) not in sys.path:
    sys.path.insert(0, os.path.dirname(CLONE))


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


def mascara_acento(img_acentuada, img_par):
    """Duas imagens ja pre-processadas (64x256) -> mascara bool (8, 32) do acento.

    Celula do latente com algum pixel que mudou mais que LIMIAR_PIXEL, dilatada
    em 1 celula (o VAE espalha o sinal para as vizinhas)."""
    a = np.asarray(img_acentuada.convert("L"), dtype=np.int16)
    b = np.asarray(img_par.convert("L"), dtype=np.int16)
    if a.shape != b.shape:
        raise ValueError(f"acentuada {a.shape} e par {b.shape} com formas diferentes")
    px = np.abs(a - b) > LIMIAR_PIXEL
    h, w = FORMA_LATENTE
    m = px.reshape(h, px.shape[0] // h, w, px.shape[1] // w).any(axis=(1, 3))
    p = np.pad(m, 1)
    return np.logical_or.reduce([p[dy:dy + h, dx:dx + w] for dy in range(3) for dx in range(3)])


def mascara_zona(img_sem_sinal):
    """Imagem SEM acento ja pre-processada (64x256) -> mascara bool (8, 32) da zona vazia.

    Celulas do latente inteiramente nas faixas logo acima e logo abaixo do
    corpo da palavra (de TOL_CORPO a ALCANCE_ZONA alturas-x alem do topo da
    altura-x / da linha de base), sem tinta (dilatada em FOLGA_TINTA px) e
    dentro das colunas da palavra. E onde um acento ou cedilha que o texto nao
    pede apareceria; hastes e pernas de letras sao tinta e ficam de fora, e o
    fundo longe da palavra tambem. None da geometria (sem tinta) -> vazia."""
    import cv2
    from acentos_sinteticos import geometria
    g = np.asarray(img_sem_sinal.convert("L"), dtype=np.float32)
    h, w = FORMA_LATENTE
    geo = geometria.analisar(g)
    if geo is None:
        return np.zeros(FORMA_LATENTE, dtype=bool)
    H, W = g.shape
    folga, alcance = TOL_CORPO * geo.altura_x, ALCANCE_ZONA * geo.altura_x
    linhas = np.arange(H)
    fora_corpo = ((linhas < geo.topo_x - folga) & (linhas >= geo.topo_x - alcance)) |                  ((linhas > geo.base + folga) & (linhas <= geo.base + alcance))
    tinta = cv2.dilate(geometria.mascara_tinta(g).astype(np.uint8),
                       np.ones((2 * FOLGA_TINTA + 1,) * 2, np.uint8)).astype(bool)
    x0, x1 = geo.caixa[0], geo.caixa[1]
    colunas = np.zeros(W, dtype=bool)
    colunas[max(0, x0 - W // w):min(W, x1 + W // w)] = True
    livre = fora_corpo[:, None] & colunas[None, :] & ~tinta
    return livre.reshape(h, H // h, w, W // w).all(axis=(1, 3))


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
        resumo = {}
        if os.path.isfile(caminho_resumo):
            with open(caminho_resumo, encoding="utf-8") as f:
                resumo = json.load(f)
            vocab = resumo.get("vocabulario")
            if vocab:
                raiz = os.path.dirname(CLONE)
                sys.path.insert(0, raiz)
                from acentos_sinteticos.vocabulario import Vocabulario
                Vocabulario(os.path.join(raiz, vocab)).conferir(
                    [t for _, _, t in sinteticas], "treino", f"leitor da base {basefolder}")
                print(f"IAM acentuado: {len(sinteticas)} rotulos conferidos contra {vocab} (so treino)")

        # peso no acento / na zona vazia: caminho de cada acentuada/par ->
        # (acentuada, par). A zona tambem precisa do par: e calculada na
        # imagem sem sinal e exclui as celulas do acento.
        self.peso_acento = float(getattr(args, "peso_acento", 1.0))
        self.peso_zona = float(getattr(args, "peso_zona", 1.0))
        self.pares = None
        if self.peso_acento != 1.0 or self.peso_zona != 1.0:
            if not resumo.get("pares_alinhados"):
                raise ValueError(f"peso_acento={self.peso_acento}/peso_zona={self.peso_zona} exige base com pares alinhados "
                                 f"(resumo.json de {basefolder} sem pares_alinhados; ver scripts/alinhar_pares.py)")
            with open(os.path.join(basefolder, "manifesto.jsonl"), encoding="utf-8") as f:
                man = [json.loads(l) for l in f]
            por_tarefa = {}
            for x in man:
                if x["tipo"] in ("acentuada", "par"):
                    por_tarefa.setdefault(x["tarefa"], {})[x["tipo"]] = x
            self.pares = {}
            for t, d in por_tarefa.items():
                if set(d) != {"acentuada", "par"}:
                    raise ValueError(f"tarefa {t} sem acentuada ou sem par no manifesto")
                par = ((os.path.join(basefolder, d["acentuada"]["arquivo"]), d["acentuada"]["rotulo"]),
                       (os.path.join(basefolder, d["par"]["arquivo"]), d["par"]["rotulo"]))
                self.pares[par[0][0]] = par
                self.pares[par[1][0]] = par
            sem_par = [c for c, _, t in sinteticas if c not in self.pares and not t.isascii()]
            if sem_par:
                raise ValueError(f"{len(sem_par)} amostras acentuadas sem par: {sem_par[:5]}")
            print(f"IAM acentuado: peso {self.peso_acento} no acento, {self.peso_zona} na zona vazia, "
                  f"{len(por_tarefa)} pares com mascara")

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

    def mascara(self, caminho):
        """Mascara float (8, 32) do acento para a amostra; zeros se ela nao tem par."""
        if caminho not in self.pares:
            return torch.zeros(FORMA_LATENTE)
        (ca, ra), (cp, rp) = self.pares[caminho]
        ia = preprocessar_iam(Image.open(ca).convert("RGB"), ra)
        ip = preprocessar_iam(Image.open(cp).convert("RGB"), rp)
        return torch.from_numpy(mascara_acento(ia, ip).astype(np.float32))

    def zona(self, caminho, transcr, sintetica, acento):
        """Mascara float (8, 32) da zona vazia: da imagem sem sinal (o par, se a
        amostra tem par; senao a propria), sem as celulas do acento."""
        if caminho in self.pares:
            caminho, transcr = self.pares[caminho][1]
        z = mascara_zona(preprocessar_iam(Image.open(caminho).convert("RGB"), transcr))
        return torch.from_numpy(z.astype(np.float32)) * (1 - acento)

    def __getitem__(self, index):
        caminho, escritor, transcr, sintetica = self.data[index]
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
        # 4 caminho, 5 cor_im; com peso no acento ou na zona, 6 mascara do
        # acento (8, 32); com peso na zona, 7 mascara da zona vazia (8, 32)
        saida = (img, transcr, self.wid[escritor], estilos, caminho, cor)
        if self.pares is not None:
            acento = self.mascara(caminho)
            saida += (acento,)
            if self.peso_zona != 1.0:
                saida += (self.zona(caminho, transcr, sintetica, acento),)
        return saida
