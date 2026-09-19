import os
import random
import torch
from PIL import Image, ImageOps, ImageFilter
from torch.utils.data import Dataset
from utils.auxilary_functions import image_resize_PIL, centered_PIL
import numpy as np

NUM_STYLE_IMGS = 5


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
        try:
            P_TINTA, P_FUNDO = 3, 40

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
    
