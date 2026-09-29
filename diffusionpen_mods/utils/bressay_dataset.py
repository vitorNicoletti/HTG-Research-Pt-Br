import os
import random
import torch
from PIL import Image, ImageOps, ImageFilter
from torch.utils.data import Dataset
from utils.auxilary_functions import image_resize_PIL, centered_PIL
import numpy as np

NUM_STYLE_IMGS = 5
# Normalizacao de contraste: o percentil P_TINTA vira preto, P_FUNDO vira
# branco. Em nivel de modulo para o train.py registrar no config.jsonl.
P_TINTA, P_FUNDO = 3, 40

# Pre-processamento. 'v1' e o original: contraste + enquadramento do recorte
# inteiro (margens, pauta e fundo incluidos). 'v2' tira a pauta, recorta justo
# na tinta e escala como o IAM. Escolhido por args.preproc; o padrao continua
# 'v1' para nao mudar runs antigos nem a calibracao da metrica, que tambem usa
# load_image(). Ver ACHADOS.md, secao 7.
PREPROC_PADRAO = "v1"
PREPROCS = ("v1", "v2")

# Parametros do v2. Pauta: os mesmos de avaliacao_diacriticos/metrica.py,
# calibrados na escala de 64 px de altura -- a pauta e uma faixa de ate 5
# fileiras cobrindo >= 90% da largura da tinta; traco horizontal de letra
# ocupa 5 a 7 fileiras.
ESPESSURA_MAX_PAUTA = 5
COBERTURA_PAUTA = 0.90
MARGEM_RECORTE = 2    # px, na escala de 64, em volta da caixa de tinta
AREA_MIN_MANCHA = 6   # px; componentes menores nao contam para a caixa


def _binarizar(g):
    """Cinza 0..255 -> mascara booleana de tinta (Otsu). Vazia sem contraste."""
    import cv2
    if float(g.max()) - float(g.min()) < 25:
        return np.zeros(g.shape, dtype=bool)
    _, m = cv2.threshold(g.astype(np.uint8), 0, 255,
                         cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return m.astype(bool)


def _caixa(mask):
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1


def _remover_pauta(g, mask):
    """Apaga a linha pautada de g (cinza 0..255, altura 64). Devolve (g, mask, n).

    Deteccao igual a da metrica. A remocao e mais cuidadosa: a metrica zera as
    fileiras inteiras, o que aqui cortaria toda letra que cruza a pauta -- e a
    cedilha fica justamente nessa regiao. So se apagam as colunas em que a
    tinta NAO continua logo acima E logo abaixo da faixa; o traco que atravessa
    a pauta sobrevive.
    """
    cx = _caixa(mask)
    if cx is None:
        return g, mask, 0
    x0, x1, _, _ = cx
    cheia = mask[:, x0:x1].mean(axis=1) > COBERTURA_PAUTA
    g, mask = g.copy(), mask.copy()
    n, i, H = 0, 0, len(cheia)
    while i < H:
        if not cheia[i]:
            i += 1
            continue
        j = i
        while j + 1 < H and cheia[j + 1]:
            j += 1
        ini, fim = i, j + 1
        if fim - ini <= ESPESSURA_MAX_PAUTA:
            acima = mask[ini - 1] if ini > 0 else np.zeros(mask.shape[1], bool)
            abaixo = mask[fim] if fim < H else np.zeros(mask.shape[1], bool)
            apagar = ~(acima & abaixo)
            g[ini:fim, apagar] = 255
            mask[ini:fim, apagar] = False
            n += fim - ini
        i = fim
    return g, mask, n


def preprocessar_v2(g):
    """Cinza 0..255 (contraste ja normalizado) -> PIL RGB 256x64.

    1. amplia para 64 px de altura, a escala em que a regra da pauta foi
       calibrada;
    2. apaga a pauta;
    3. recorta justo na tinta (ignorando manchas minusculas do papel);
    4. escala como o IAM: altura 64 preservando o aspecto, cabendo em 256,
       centralizado em fundo branco.
    """
    import cv2
    h, w = g.shape
    g = cv2.resize(g, (max(1, round(w * 64.0 / h)), 64),
                   interpolation=cv2.INTER_CUBIC)
    g = np.clip(g, 0, 255)
    mask = _binarizar(g)
    g, mask, _ = _remover_pauta(g, mask)

    n, rot, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    grandes = [k for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] >= AREA_MIN_MANCHA]
    cx = _caixa(np.isin(rot, grandes)) if grandes else None
    if cx is not None:
        x0, x1, y0, y1 = cx
        m = MARGEM_RECORTE
        g = g[max(0, y0 - m):min(g.shape[0], y1 + m),
              max(0, x0 - m):min(g.shape[1], x1 + m)]
    im = Image.fromarray(g.astype(np.uint8)).convert("RGB")
    return ImageOps.pad(im, size=(256, 64), color="white")


class BRESSAY_Dataset(Dataset):
    def __init__(
        self,
        basefolder,
        subset,
        segmentation_level="word",
        fixed_size=(64, 256),
        tokenizer=None,
        text_encoder=None,
        feat_extractor=None,
        transforms=None,
        args=None,
    ):
        self.basefolder = basefolder
        self.subset = subset
        self.transforms = transforms
        self.fixed_size = fixed_size
        self.tokenizer = tokenizer
        self.args = args

        # Caminhos apontando para o setup local do BRESSAY
        self.tsv_file = os.path.join(basefolder, "splits", f"{subset}.tsv")
        self.images_root = os.environ.get("BRESSAY_IMAGES", "./bressay/data/words")

        self.data = []
        self.writer_indices = {}

        # Limite de amostras por split; 0 = usa o split inteiro. Nao tem a ver
        # com RAM: este dataset e lazy (guarda so os paths e abre a imagem no
        # __getitem__), entao o custo de memoria de um split inteiro e o de uma
        # lista de tuplas de string. O limite existe para poder comparar runs
        # com volumes diferentes de dados.
        max_amostras = getattr(args, "max_samples", 0) or 0

        print(f"BRESSAY [{subset}]: Lendo indice do disco (Otimizado Lazy Load)...")
        with open(self.tsv_file, "r", encoding="utf-8") as f:
            lines = f.read().strip().split("\n")

        for line in lines:
            if max_amostras and len(self.data) >= max_amostras:
                break
            if not line.strip():
                continue

            parts = line.split("\t")
            if len(parts) < 3:
                continue
            img_rel_path, writer_id, transcr = parts[0], parts[1], parts[2]

            try:
                writer_id = int(writer_id)
            except ValueError:
                continue

            full_img_path = os.path.join(self.images_root, img_rel_path)
            self.data.append((full_img_path, transcr, writer_id))

            # CRITICO: o indice guardado tem que ser a posicao em self.data,
            # nao o numero da linha do arquivo. Como linhas vazias/malformadas
            # sao puladas com `continue`, os dois desalinham e o sorteio de
            # estilo passa a devolver imagens de OUTRO escritor -- silenciosamente.
            self.writer_indices.setdefault(writer_id, []).append(len(self.data) - 1)

        self.all_writers = list(self.writer_indices.keys())

        # Mapa writer_id -> 0..N-1, para a posicao 2 do return (o train.py faz
        # .to(device) nela, entao precisa ser int).
        wids = sorted(self.writer_indices.keys())
        self.wid2idx = {w: i for i, w in enumerate(wids)}

        print(f"BRESSAY [{subset}]: {len(wids)} escritores distintos.")
        print(
            f"BRESSAY [{subset}]: Carregamento concluido. "
            f"{len(self.data)} amostras validas de {len(lines)} linhas no split"
            f"{f' (limitado a {max_amostras})' if max_amostras else ' (split inteiro)'}."
        )

    def __len__(self):
        return len(self.data)

    
    
    def load_image(self, img_path):
        preproc = getattr(self.args, "preproc", PREPROC_PADRAO) or PREPROC_PADRAO
        if preproc not in PREPROCS:
            raise ValueError(f"preproc desconhecido: {preproc!r} (use {PREPROCS})")
        if preproc == "v2":
            # Fora do try/except de baixo de proposito: la qualquer erro vira
            # imagem branca em silencio, e um defeito no v2 faria o modelo
            # treinar sobre imagens vazias sem ninguem notar. So o arquivo
            # ilegivel vira branco aqui.
            try:
                im = Image.open(img_path).convert("L")
            except OSError:
                return Image.new("RGB", (256, 64), color="white")
            g = np.asarray(im, dtype=np.float32)
            lo, hi = np.percentile(g, P_TINTA), np.percentile(g, P_FUNDO)
            if hi - lo >= 8:
                g = np.clip((g - lo) / (hi - lo), 0, 1) * 255
            return preprocessar_v2(g)

        try:
            im = Image.open(img_path).convert("RGB")

            # normalizacao de contraste por percentis
            g = np.asarray(im.convert("L"), dtype=np.float32)
            lo = np.percentile(g, P_TINTA)
            hi = np.percentile(g, P_FUNDO)
            if hi - lo >= 8:
                g = np.clip((g - lo) / (hi - lo), 0, 1) * 255
                im = Image.fromarray(g.astype(np.uint8)).convert("RGB")

            # resize para altura 64
            w, h = im.size
            # im = im.resize((max(1, int(w * 64 / h)), 64), Image.BICUBIC)
    
            # ajuste para largura 256
            if im.size[0] < 256:
                im = ImageOps.pad(im, size=(256, 64), color="white")
            else:
                im = im.resize((256, 64), Image.BICUBIC)
    
            return im
        except Exception:
            return Image.new("RGB", (256, 64), color="white")


    def __getitem__(self, index):
        img_path, transcr, wid = self.data[index]

        # 1. Imagem principal
        img_pil = self.load_image(img_path)
        img_tensor = self.transforms(img_pil) if self.transforms else img_pil

        # 2. Imagens de estilo: NUM_STYLE_IMGS amostras do mesmo escritor.
        # random.choices sorteia com reposicao, entao funciona mesmo para
        # escritores com poucas amostras.
        sampled_indices = random.choices(self.writer_indices[wid], k=NUM_STYLE_IMGS)
        style_tensors = []
        for idx in sampled_indices:
            s_path, _, _ = self.data[idx]
            s_pil = self.load_image(s_path)
            s_tensor = self.transforms(s_pil) if self.transforms else s_pil
            style_tensors.append(s_tensor)

        style_images = torch.stack(style_tensors)  # [NUM_STYLE_IMGS, 3, 64, 256]

        # Posicoes lidas pelo train.py:
        #   loop de treino (linha ~456): 0, 1, 2, 3
        #   loop de amostragem (linha ~194): 0, 1, 2, 3, 4, 5
        return (
            img_tensor,  # 0: image      [3, 64, 256]
            transcr,  # 1: transcript (str)
            self.wid2idx[wid],  # 2: s_id       (int)
            style_images,  # 3: style      [NUM_STYLE_IMGS, 3, 64, 256]
            img_path,  # 4: img_path   (str)
            img_tensor,  # 5: cor_im     [3, 64, 256]
        )
    
