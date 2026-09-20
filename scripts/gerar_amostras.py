"""
gerar_amostras.py -- geracao standalone a partir de um checkpoint

    python gerar_amostras.py --ckpt <arquivo .pt> --out <pasta>

Exemplo:
    python gerar_amostras.py --ckpt ./meu_modelo/models/ema_ckpt.pt --out ./amostras

--ckpt aponta para o ARQUIVO, direto. Semente e escritores sao fixos por
padrao, entao rodar em checkpoints diferentes produz amostras comparaveis.

Nao toca no train.py nem no treino em andamento. Mas USA A MESMA GPU: se o
treino estiver rodando pode dar OOM, entao rode com a GPU livre.
"""

import os
import sys
import copy
import random
import argparse

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CanineModel, CanineTokenizer
from torch.nn import DataParallel

# O train.py faz "from unet import UNetModel", entao o diretorio dele
# precisa estar no path.
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.join(RAIZ, "DiffusionPen")
sys.path.insert(0, REPO)

from unet import UNetModel
from feature_extractor import ImageEncoder
from utils.word_dataset import WordLineDataset
from utils.bressay_dataset import BRESSAY_Dataset

# ============================ CONFIGURACAO ============================
# Caminhos e saida vem da linha de comando (--ckpt / --out). Aqui ficam so as
# constantes que precisam casar com o treino.
STYLE_PADRAO = os.path.join(RAIZ, "style_models", "mixed_bressay_mobilenetv2_100.pth")
STABLE_DIF = "runwayml/stable-diffusion-v1-5"
DATASET_FOLDER = os.path.join(RAIZ, "bressay_split")

# Palavras de teste: as cinco primeiras exercitam os diacriticos (til, cedilha,
# agudo, circunflexo), que sao o objetivo do trabalho; "text" e controle ASCII.
PALAVRAS_PADRAO = ["ação", "coração", "português", "não", "avó", "text"]

STYLE_CLASSES = 339   # tem que bater com o valor usado no treino
VOCAB_SIZE = 79
NUM_STYLE_IMGS = 5    # cravado no unet.py (~1309): y.reshape(b, 5, -1)
# ======================================================================


def parse_cli():
    p = argparse.ArgumentParser(
        description="Gera amostras de manuscrito a partir de um checkpoint."
    )
    p.add_argument("--ckpt", required=True,
                   help="caminho do ARQUIVO .pt (ex.: ./meu_modelo/models/ema_ckpt.pt)")
    p.add_argument("--out", required=True, help="pasta onde salvar os PNG")
    p.add_argument("--palavras", nargs="+", default=PALAVRAS_PADRAO,
                   help="palavras a gerar (padrao: as de teste de diacritico)")
    p.add_argument("--styles", type=int, default=4,
                   help="quantos escritores diferentes por palavra")
    p.add_argument("--steps", type=int, default=50, help="passos do DDIM")
    p.add_argument("--style", type=str, default=STYLE_PADRAO,
                   help="extrator de estilo; TEM que ser o mesmo usado no treino")
    p.add_argument("--device", type=str, default="cuda:0",
                   help="cpu para tirar a GPU da equacao")
    p.add_argument("--seed", type=int, default=42,
                   help="mesma semente = amostras comparaveis entre checkpoints")
    p.add_argument("--estilo_de", choices=("bressay", "iam"), default="bressay",
                   help="de onde vem as 5 imagens de referencia que o extrator "
                        "le. 'iam' usa recortes do IAM, com o pre-processamento "
                        "do IAM (aspecto preservado, sem normalizacao por "
                        "percentis) -- passar imagem do IAM pelo load_image do "
                        "BRESSAY mediria a coisa errada.")
    p.add_argument("--em_lote", action="store_true",
                   help="gera os N estilos num lote so. NAO use na RX 6600 XT: "
                        "lote > 1 corrompe a amostragem nessa placa (NaN e "
                        "colapso para cinza, nao reproduzivel). O padrao gera "
                        "um estilo por vez, que bate com a CPU em 1/255.")
    a = p.parse_args()
    if not os.path.isfile(a.style):
        p.error(
            f"extrator de estilo nao encontrado: {a.style}\n"
            f"       Ele precisa ser TREINADO antes (style_encoder_train.py) e, por\n"
            f"       causa do defeito de hardware documentado em diagnostico/, isso\n"
            f"       tem de ser feito numa GPU validada. Veja o README.")
    if not os.path.isfile(a.ckpt):
        p.error(f"--ckpt nao e um arquivo: {a.ckpt}\n"
                f"       aponte para o .pt direto, nao para a pasta do modelo.")
    return a


class Passa(nn.Module):
    """Prefixo "module." do DataParallel sem CUDA, para rodar na CPU."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *a, **k):
        return self.module(*a, **k)


def embrulha(m, na_gpu, ids):
    return DataParallel(m, device_ids=ids) if na_gpu else Passa(m)


def _prep_iam(im):
    """Pre-processamento do IAM, copiado de utils/iam_dataset.py.

    Preserva o aspecto sempre: altura 64 primeiro, depois encolhe ate caber em
    256 e centraliza. Nao tem normalizacao por percentis -- ela foi calibrada
    para o papel pautado do BRESSAY e nao se aplica aqui.
    """
    from PIL import Image, ImageOps
    w, h = im.size
    im = im.resize((max(1, int(w * 64 / h)), 64))
    w, h = im.size
    if w < 256:
        return ImageOps.pad(im, size=(256, 64), color="white")
    while w > 256:
        w -= 20
        im = im.resize((w, max(1, int(im.size[1] * w / im.size[0]))))
        w, h = im.size
    fundo = Image.new("RGB", (256, 64), (255, 255, 255))
    fundo.paste(im, ((256 - w) // 2, (64 - h) // 2))
    return fundo


def estilos_iam(k, transform):
    """k listas de NUM_STYLE_IMGS recortes do IAM, cada lista de um escritor.

    No IAM uma pasta de formulario (a01/a01-000u/) e um escritor, entao os
    recortes de estilo saem todos da mesma pasta -- e o equivalente ao
    writer_indices do BRESSAY.
    """
    import glob
    from PIL import Image
    raiz = os.path.join(REPO, "iam_data", "words")
    formularios = [d for d in glob.glob(os.path.join(raiz, "*", "*"))
                   if os.path.isdir(d)]
    if not formularios:
        raise SystemExit(f"nenhum formulario do IAM em {raiz}")
    saida = []
    for _ in range(k):
        while True:
            d = random.choice(formularios)
            pngs = glob.glob(os.path.join(d, "*.png"))
            if len(pngs) >= 1:
                break
        escolhidos = random.choices(pngs, k=NUM_STYLE_IMGS)
        saida.append([transform(_prep_iam(Image.open(f).convert("RGB")))
                      for f in escolhidos])
    return saida


def build_args(device="cuda:0", style_path=STYLE_PADRAO):
    """Namespace com os mesmos defaults do parser do train.py."""
    a = argparse.Namespace()
    a.device = device
    a.img_size = (64, 256)
    a.channels = 4
    a.emb_dim = 320
    a.num_heads = 4
    a.num_res_blocks = 1
    a.latent = True
    a.img_feat = True
    a.interpolation = False
    a.mix_rate = None
    a.model_name = "diffusionpen"
    a.dataparallel = False
    a.color = True
    a.level = "word"
    a.unet = "unet_latent"
    a.batch_size = 1
    a.stable_dif_path = STABLE_DIF
    a.style_path = style_path
    a.save_path = ""
    return a


def main():
    cli = parse_cli()
    random.seed(cli.seed)
    torch.manual_seed(cli.seed)

    args = build_args(cli.device, cli.style)
    device = args.device
    na_gpu = device != "cpu"
    device_ids = [int("".join(filter(str.isdigit, device)))] if na_gpu else []
    os.makedirs(cli.out, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # ---------------- texto ----------------
    print("carregando CANINE...")
    tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
    text_encoder = CanineModel.from_pretrained("google/canine-c")
    text_encoder = embrulha(text_encoder, na_gpu, device_ids).to(device)
    text_encoder.eval()

    # ---------------- unet + ema ----------------
    print("construindo unet...")
    unet = UNetModel(
        image_size=args.img_size,
        in_channels=args.channels,
        model_channels=args.emb_dim,
        out_channels=args.channels,
        num_res_blocks=args.num_res_blocks,
        attention_resolutions=(1, 1),
        channel_mult=(1, 1),
        num_heads=args.num_heads,
        num_classes=STYLE_CLASSES,
        context_dim=args.emb_dim,
        vocab_size=VOCAB_SIZE,
        text_encoder=text_encoder,
        args=args,
    )
    # DataParallel e obrigatorio: o checkpoint foi salvo com o prefixo "module."
    unet = embrulha(unet, na_gpu, device_ids).to(device)

    ema_model = copy.deepcopy(unet).eval().requires_grad_(False)
    # Vale tanto para um EMA (ema_*.pt) quanto para o modelo bruto (ckpt.pt):
    # ja aconteceu de o EMA derivar do modelo e gerar lixo, entao comparar os
    # dois e util quando a saida piora sem explicacao.
    ckpt = cli.ckpt
    ema_model.load_state_dict(torch.load(ckpt, map_location=device))
    ema_model.eval()
    print("checkpoint carregado:", ckpt)

    p = dict(ema_model.named_parameters())["module.out.2.bias"]
    print("out.2.bias norma:", p.norm().item())

    # ---------------- vae ----------------
    vae = AutoencoderKL.from_pretrained(STABLE_DIF, subfolder="vae")
    vae = embrulha(vae, na_gpu, device_ids).to(device)
    vae.requires_grad_(False)
    vae.eval()
    vae_model = vae.module

    # ---------------- style extractor ----------------
    feat = ImageEncoder(
        model_name="mobilenetv2_100", num_classes=0, pretrained=True, trainable=True
    )
    sd = torch.load(cli.style, map_location=device)
    md = feat.state_dict()
    sd = {k: v for k, v in sd.items() if k in md and md[k].shape == v.shape}
    md.update(sd)
    feat.load_state_dict(md)
    feat = embrulha(feat, na_gpu, device_ids).to(device)
    feat.requires_grad_(False)
    feat.eval()

    # ---------------- scheduler ----------------
    ddim = DDIMScheduler.from_pretrained(STABLE_DIF, subfolder="scheduler")
    ddim.set_timesteps(cli.steps)

    # ---------------- dataset (so para pegar estilo) ----------------
    ds = BRESSAY_Dataset(DATASET_FOLDER, "train", transforms=transform, args=args)
    escritores = random.sample(list(ds.wid2idx.keys()), min(cli.styles, len(ds.wid2idx)))
    print("escritores escolhidos:", escritores)

    # descobre o shape do latente sem chutar
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 64, 256, device=device)
        lat_shape = vae_model.encode(dummy).latent_dist.sample().shape[1:]
    print("shape do latente:", tuple(lat_shape))

    # ---------------- imagens de referencia de estilo ----------------
    # Fixadas ANTES do laco de palavras: assim todas as palavras de uma
    # execucao veem exatamente os mesmos escritores, e duas execucoes com a
    # mesma semente sao comparaveis palavra a palavra.
    if cli.estilo_de == "iam":
        ref = estilos_iam(len(escritores), transform)
        print(f"estilo: {len(ref)} escritores do IAM "
              f"({NUM_STYLE_IMGS} recortes cada)")
    else:
        ref = [[transform(ds.load_image(ds.data[i][0]))
                for i in random.choices(ds.writer_indices[w], k=NUM_STYLE_IMGS)]
               for w in escritores]
        print(f"estilo: {len(ref)} escritores do BRESSAY "
              f"({NUM_STYLE_IMGS} recortes cada)")

    rotulos = [ds.wid2idx[w] for w in escritores]

    def gera(palavra, indices):
        """Gera `palavra` para os estilos em `indices`, num unico lote."""
        n = len(indices)
        st = [t for i in indices for t in ref[i]]
        style_images = torch.stack(st).to(device)
        with torch.no_grad():
            style_features = feat(style_images)
            text_features = tokenizer(
                [palavra] * n, padding="max_length", truncation=True,
                return_tensors="pt", max_length=40,
            ).to(device)
            labels = torch.tensor([rotulos[i] for i in indices],
                                  device=device).long()
            x = torch.randn(n, *lat_shape, device=device)
            for t in ddim.timesteps:
                ts = t.repeat(n).to(device).long()
                noise_pred = ema_model(x, timesteps=ts, context=text_features,
                                       y=labels, style_extractor=style_features)
                x = ddim.step(noise_pred, t, x).prev_sample
            return vae_model.decode(x / 0.18215).sample

    # ---------------- geracao ----------------
    n = len(escritores)
    for palavra in cli.palavras:
        if cli.em_lote:
            img = gera(palavra, list(range(n)))
        else:
            # Um estilo por vez. A semente e refixada por estilo para que o
            # ruido inicial de cada painel nao dependa de quantos estilos
            # foram pedidos -- sem isso, mudar --styles mudaria as imagens.
            partes = []
            for i in range(n):
                torch.manual_seed(cli.seed + i)
                partes.append(gera(palavra, [i]))
            img = torch.cat(partes, dim=0)

        img = (img.clamp(-1, 1) + 1) / 2
        out = os.path.join(cli.out, f"{palavra}.png")
        save_image(img, out, nrow=n)
        desvios = [f"{float(p.std()):.3f}" for p in img]
        print(f"salvo: {out}  (std por painel: {' '.join(desvios)})")

    print("\npronto. veja", cli.out)


if __name__ == "__main__":
    main()
