"""DiffusionPen carregado uma vez, para gerar muitas palavras em lote.

Mesma montagem de scripts/gerar_amostras.py (CANINE, UNet com DataParallel,
VAE do SD 1.5, extrator de estilo MobileNetV2, DDIM), mas como objeto: os
scripts de base e de avaliacao geram milhares de palavras com escritores e
sementes controlados por eles.

As referencias de estilo usam o pre-processamento do IAMDataset
(utils/iam_acentuado_dataset.preprocessar_iam), o mesmo do treino.
"""

import copy
import os
import random
import sys

import torch
from PIL import Image
from torchvision import transforms

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLONE = os.path.join(RAIZ, "DiffusionPen")
sys.path.insert(0, CLONE)
sys.path.insert(0, os.path.join(RAIZ, "scripts"))

NUM_STYLE_IMGS = 5
STYLE_CLASSES = 339
VOCAB_SIZE = 79
TRANSFORM = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


class EscritoresIAM:
    """Palavras do IAM por escritor, de um split de utils/splits_words."""

    def __init__(self, split="iam_train_val.txt"):
        from utils.iam_acentuado_dataset import preprocessar_iam
        self._prep = preprocessar_iam
        raiz = os.path.join(CLONE, "iam_data", "words")
        self.por_escritor = {}
        with open(os.path.join(CLONE, "utils", "splits_words", split), encoding="utf-8") as f:
            for l in f:
                c = l.rstrip("\n").split(",")
                if len(c) >= 3:
                    self.por_escritor.setdefault(c[1], []).append((os.path.join(raiz, c[0]), ",".join(c[2:])))
        self.escritores = sorted(self.por_escritor)

    def referencias(self, escritor, rnd):
        """5 recortes do escritor (palavras com mais de 3 letras, como no IAMDataset)."""
        cand = [x for x in self.por_escritor[escritor] if len(x[1]) > 3] or self.por_escritor[escritor]
        escolha = rnd.sample(cand, NUM_STYLE_IMGS) if len(cand) >= NUM_STYLE_IMGS \
            else rnd.choices(cand, k=NUM_STYLE_IMGS)
        return [TRANSFORM(self._prep(Image.open(c).convert("RGB"), "palavra")) for c, _ in escolha]


class GeradorDiffusionPen:
    def __init__(self, ckpt, style_path, device="cuda:0", steps=50, texto_max_len=40,
                 stable_dif=None):
        import gerar_amostras as ga   # build_args e embrulha: mesmos do gerar_amostras.py
        from diffusers import AutoencoderKL, DDIMScheduler
        from transformers import CanineModel, CanineTokenizer
        from unet import UNetModel
        from feature_extractor import ImageEncoder

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise SystemExit("ERRO: GPU pedida mas indisponivel (sem fallback para CPU)")
        stable_dif = stable_dif or os.environ.get("SD", "stable-diffusion-v1-5/stable-diffusion-v1-5")
        args = ga.build_args(device, style_path)
        self.device, self.texto_max_len = device, texto_max_len
        ids = [int("".join(filter(str.isdigit, device)))] if device != "cpu" else []
        na_gpu = device != "cpu"

        self.tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
        enc = ga.embrulha(CanineModel.from_pretrained("google/canine-c"), na_gpu, ids).to(device).eval()
        unet = UNetModel(image_size=args.img_size, in_channels=args.channels,
                         model_channels=args.emb_dim, out_channels=args.channels,
                         num_res_blocks=args.num_res_blocks, attention_resolutions=(1, 1),
                         channel_mult=(1, 1), num_heads=args.num_heads,
                         num_classes=STYLE_CLASSES, context_dim=args.emb_dim,
                         vocab_size=VOCAB_SIZE, text_encoder=enc, args=args)
        unet = ga.embrulha(unet, na_gpu, ids).to(device)
        self.ema = copy.deepcopy(unet).eval().requires_grad_(False)
        self.ema.load_state_dict(torch.load(ckpt, map_location=device))
        self.ema.eval()
        vae = ga.embrulha(AutoencoderKL.from_pretrained(stable_dif, subfolder="vae"), na_gpu, ids).to(device)
        vae.requires_grad_(False)
        self.vae = vae.module
        feat = ImageEncoder(model_name="mobilenetv2_100", num_classes=0, pretrained=True, trainable=True)
        sd = torch.load(style_path, map_location=device)
        md = feat.state_dict()
        md.update({k: v for k, v in sd.items() if k in md and md[k].shape == v.shape})
        feat.load_state_dict(md)
        self.feat = ga.embrulha(feat, na_gpu, ids).to(device).eval().requires_grad_(False)
        self.ddim = DDIMScheduler.from_pretrained(stable_dif, subfolder="scheduler")
        self.ddim.set_timesteps(steps)
        with torch.no_grad():
            self.lat_shape = self.vae.encode(torch.zeros(1, 3, 64, 256, device=device)).latent_dist.sample().shape[1:]

    @torch.no_grad()
    def gerar(self, palavras, referencias, semente):
        """palavras: N textos; referencias: N listas de 5 tensores; -> (N,3,64,256) em 0..1."""
        n = len(palavras)
        g = torch.Generator(device="cpu").manual_seed(semente)
        x = torch.randn(n, *self.lat_shape, generator=g).to(self.device)
        estilo = self.feat(torch.stack([t for r in referencias for t in r]).to(self.device))
        texto = self.tokenizer(list(palavras), padding="max_length", truncation=True,
                               return_tensors="pt", max_length=self.texto_max_len).to(self.device)
        y = torch.zeros(n, dtype=torch.long, device=self.device)   # ignorado com estilo
        for t in self.ddim.timesteps:
            ts = t.repeat(n).to(self.device).long()
            ruido = self.ema(x, timesteps=ts, context=texto, y=y, style_extractor=estilo)
            x = self.ddim.step(ruido, t, x).prev_sample
        img = self.vae.decode(x / 0.18215).sample
        return ((img.clamp(-1, 1) + 1) / 2).float().cpu()
