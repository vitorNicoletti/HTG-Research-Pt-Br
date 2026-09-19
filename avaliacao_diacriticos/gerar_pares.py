"""
gerar_pares.py -- sonda de pares minimos para a metrica de diacriticos.

Gera, para cada palavra pedida, UMA imagem por (escritor, semente), com a
garantia de que o par minimo ("nacao" / "nacao-com-til") compartilha:

  * as 5 imagens de estilo (dependem so do escritor e de --style-seed);
  * o ruido inicial do DDIM (depende so de (semente, escritor));
  * o scheduler e o numero de passos.

Logo a UNICA variavel entre os gemeos e a string do texto. Isso e o que
permite comparar as duas imagens diretamente (E1 por diferenca) ou, se elas
nao sairem alinhadas, comparar so a coluna do caractere acentuado.

Nao importa nada de train.py: reconstroi a UNet do mesmo jeito que o
gerar_amostras.py da raiz, so que com o sorteio determinístico.

Saida: PNGs 256x64 individuais (um por amostra, NAO um painel) em --out-dir,
mais um manifest.jsonl com os metadados de cada amostra.
"""

import argparse
import copy
import hashlib
import json
import os
import random
import sys
import unicodedata

import torch
import torch.nn as nn
from torch.nn import DataParallel
from torchvision import transforms
from torchvision.utils import save_image
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CanineModel, CanineTokenizer

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.join(RAIZ, "DiffusionPen")
sys.path.insert(0, REPO)

from unet import UNetModel                      # noqa: E402
from feature_extractor import ImageEncoder      # noqa: E402
from utils.bressay_dataset import BRESSAY_Dataset  # noqa: E402

STABLE_DIF = "runwayml/stable-diffusion-v1-5"
STYLE_PATH = os.path.join(RAIZ, "style_models", "mixed_bressay_mobilenetv2_100.pth")
DATASET_FOLDER = os.path.join(RAIZ, "bressay_split")

STYLE_CLASSES = 339
VOCAB_SIZE = 79
NUM_STYLE_IMGS = 5   # cravado no unet.py (~1309): y.reshape(b, 5, -1)


class Envolto(nn.Module):
    """Equivalente CPU do DataParallel: so recria o prefixo "module."."""

    def __init__(self, m):
        super().__init__()
        self.module = m

    def forward(self, *a, **k):
        return self.module(*a, **k)


def sem_acento(s):
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def slug(s):
    """Nome de arquivo estavel e sem acento, mas que nao colide com o gemeo."""
    base = sem_acento(s)
    base = "".join(c if (c.isalnum() or c in "-_") else "x" for c in base)
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()[:6]
    return f"{base}-{h}"


def semente_de(*partes):
    """Semente inteira estavel a partir de qualquer tupla (nao usa hash())."""
    m = hashlib.sha1("|".join(str(p) for p in partes).encode("utf-8")).digest()
    return int.from_bytes(m[:4], "big")


def build_args(save_path):
    a = argparse.Namespace()
    a.device = "cuda:0"
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
    a.style_path = STYLE_PATH
    a.save_path = save_path
    return a


def carregar_palavras(args):
    if args.pares_tsv:
        pares = []
        with open(args.pares_tsv, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip() or line.startswith("#"):
                    continue
                p = line.split("\t")
                pares.append((p[0], p[1]))
        if args.limite:
            pares = pares[: args.limite]
        return pares
    if args.palavras:
        return [(w, sem_acento(w)) for w in args.palavras]
    raise SystemExit("passe --pares-tsv ou --palavras")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-path", required=True,
                    help="pasta do modelo (contem models/<ckpt>)")
    ap.add_argument("--ckpt", default="ema_ckpt.pt")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pares-tsv", default=None,
                    help="TSV: acentuada \\t ascii")
    ap.add_argument("--palavras", nargs="*", default=None)
    ap.add_argument("--limite", type=int, default=0)
    ap.add_argument("--n-styles", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--style-seed", type=int, default=42,
                    help="controla QUAIS escritores e QUAIS imagens de estilo")
    ap.add_argument("--ddim-steps", type=int, default=50)
    ap.add_argument("--batch", type=int, default=1,
                    help="MANTENHA 1. Ver nota sobre corrupcao numerica em lote.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--autoteste-lote", action="store_true",
                    help="so verifica que gerar em lote == gerar uma a uma")
    args_cli = ap.parse_args()

    args = build_args(args_cli.save_path)
    args.device = args_cli.device
    device = args_cli.device
    em_cuda = device.startswith("cuda")
    device_ids = [int("".join(filter(str.isdigit, device)) or 0)] if em_cuda else None

    os.makedirs(args_cli.out_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    def wrap(m):
        # Em CPU nao da para usar DataParallel, mas o checkpoint foi salvo com
        # o prefixo "module." nas chaves; Envolto reproduz esse prefixo.
        if em_cuda:
            return DataParallel(m, device_ids=device_ids).to(device)
        return Envolto(m).to(device)

    print("carregando CANINE...", flush=True)
    tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
    text_encoder = wrap(CanineModel.from_pretrained("google/canine-c")).eval()

    print("construindo unet...", flush=True)
    unet = UNetModel(
        image_size=args.img_size, in_channels=args.channels,
        model_channels=args.emb_dim, out_channels=args.channels,
        num_res_blocks=args.num_res_blocks, attention_resolutions=(1, 1),
        channel_mult=(1, 1), num_heads=args.num_heads,
        num_classes=STYLE_CLASSES, context_dim=args.emb_dim,
        vocab_size=VOCAB_SIZE, text_encoder=text_encoder, args=args,
    )
    unet = wrap(unet)
    ema_model = copy.deepcopy(unet).eval().requires_grad_(False)
    ckpt = os.path.join(args_cli.save_path, "models", args_cli.ckpt)
    ema_model.load_state_dict(torch.load(ckpt, map_location=device))
    ema_model.eval()
    print("ckpt carregado:", ckpt, flush=True)
    print("out.2.bias norma:",
          dict(ema_model.named_parameters())["module.out.2.bias"].norm().item())

    vae = wrap(AutoencoderKL.from_pretrained(STABLE_DIF, subfolder="vae"))
    vae.requires_grad_(False)
    vae.eval()
    vae_model = vae.module

    feat = ImageEncoder(model_name="mobilenetv2_100", num_classes=0,
                        pretrained=True, trainable=True)
    sd = torch.load(STYLE_PATH, map_location=device)
    md = feat.state_dict()
    sd = {k: v for k, v in sd.items() if k in md and md[k].shape == v.shape}
    md.update(sd)
    feat.load_state_dict(md)
    feat = wrap(feat)
    feat.requires_grad_(False)
    feat.eval()

    ddim = DDIMScheduler.from_pretrained(STABLE_DIF, subfolder="scheduler")
    ddim.set_timesteps(args_cli.ddim_steps)

    ds = BRESSAY_Dataset(DATASET_FOLDER, "train", transforms=transform, args=args)

    # --- escolha DETERMINISTICA dos escritores -------------------------------
    todos = sorted(ds.wid2idx.keys())
    rng_est = random.Random(args_cli.style_seed)
    escritores = rng_est.sample(todos, min(args_cli.n_styles, len(todos)))
    print("escritores:", escritores, flush=True)

    # --- imagens de estilo: dependem SO de (style_seed, escritor) ------------
    style_feats = {}
    with torch.no_grad():
        for w in escritores:
            rng = random.Random(semente_de(args_cli.style_seed, "estilo", w))
            idxs = [rng.choice(ds.writer_indices[w]) for _ in range(NUM_STYLE_IMGS)]
            imgs = torch.stack([transform(ds.load_image(ds.data[i][0])) for i in idxs])
            style_feats[w] = feat(imgs.to(device)).detach()   # [5, F]

    with torch.no_grad():
        lat_shape = vae_model.encode(
            torch.zeros(1, 3, 64, 256, device=device)
        ).latent_dist.sample().shape[1:]
    print("shape do latente:", tuple(lat_shape), flush=True)

    pares = carregar_palavras(args_cli)
    # lista plana de (palavra, rotulo_do_par, eh_acentuada)
    itens = []
    for acc, asc in pares:
        par_id = slug(acc)
        itens.append((acc, par_id, True))
        if asc != acc:
            itens.append((asc, par_id, False))
    print(f"{len(pares)} pares -> {len(itens)} palavras x "
          f"{len(escritores)} estilos x {len(args_cli.seeds)} sementes = "
          f"{len(itens) * len(escritores) * len(args_cli.seeds)} imagens",
          flush=True)

    def amostrar(textos, w, base_noise):
        """Roda o DDIM para um lote de palavras com estilo e ruido fixos.

        LOTE = 1, SEMPRE. Nesta GPU (RX 6600 XT / ROCm) a amostragem em
        lote e numericamente corrupta. Medido com a mesma chamada repetida,
        4 palavras, escritor fixo, ruido fixo, std da imagem gerada:

            batch 4, rep 0 : [0.0793, 0.0515, 0.0423, 0.0643]
            batch 4, rep 1 : [nan, nan, nan, nan]
            batch 4, rep 2 : [0.1229, 0.0596, 0.0756, 0.0691]
            batch 2        : [0.1863, 0.0597, 0.0398, 0.0637]
            batch 1, rep 0 : [0.1858, 0.1945, 0.2165, 0.2193]
            batch 1, rep 1 : [0.1809, 0.1945, 0.2165, 0.2193]

        Ou seja: com lote > 1 a saida colapsa para cinza quase uniforme (ou
        vira NaN) e NAO e reproduzivel entre execucoes; com lote 1 e estavel.
        Como a metrica de diacritico mede exatamente presenca de tinta, uma
        imagem colapsada seria lida como "acento omitido" -- o artefato viraria
        resultado. Por isso o padrao e 1 e existe o --autoteste-lote.

        `y` recebe o tensor de rotulos por simetria com o gerar_amostras.py da
        raiz; o unet sobrescreve y com o style_extractor (unet.py ~1288), entao
        ele nao muda a saida com lote 1.
        """
        n = len(textos)
        with torch.no_grad():
            sf = style_feats[w].unsqueeze(0).repeat(n, 1, 1)
            sf = sf.reshape(n * NUM_STYLE_IMGS, -1)
            tf = tokenizer(textos, padding="max_length", truncation=True,
                           return_tensors="pt", max_length=40).to(device)
            labels = torch.tensor([ds.wid2idx[w]] * n, device=device).long()
            x = base_noise.repeat(n, 1, 1, 1).clone()
            for t in ddim.timesteps:
                ts = t.repeat(n).to(device).long()
                eps = ema_model(x, timesteps=ts, context=tf,
                                y=labels, style_extractor=sf)
                x = ddim.step(eps, t, x).prev_sample
            img = vae_model.decode(x / 0.18215).sample
        return (img.clamp(-1, 1) + 1) / 2

    def aquecer(w):
        """Descarta as 2 primeiras amostragens do processo.

        Medido em 5 repeticoes da mesma palavra/estilo/ruido nesta GPU:
        mean|dif| = 0.0078 entre as reps 0-1 e 0.0 entre as reps 2-4. E a
        selecao de kernel do MIOpen na primeira execucao.
        """
        g = torch.Generator(device="cpu").manual_seed(0)
        bn = torch.randn(1, *lat_shape, generator=g).to(device)
        for _ in range(2):
            amostrar([itens[0][0]], w, bn).cpu()

    if args_cli.autoteste_lote:
        # Invariante que a metrica precisa: a imagem gerada nao pode depender
        # do tamanho do lote nem da execucao. Nesta GPU (RX 6600 XT / ROCm) o
        # lote > 1 quebra isso -- ver a nota em amostrar().
        w0 = escritores[0]
        aquecer(w0)
        g = torch.Generator(device="cpu").manual_seed(semente_de(0, "ruido", w0))
        bn = torch.randn(1, *lat_shape, generator=g).to(device)
        palavras_teste = [t[0] for t in itens[:4]]
        b = args_cli.batch

        def stats(t):
            return (f"min={float(t.min()):.3f} max={float(t.max()):.3f} "
                    f"std={float(t.std()):.4f} nan={bool(torch.isnan(t).any())}")

        # .cpu() logo apos gerar: segurar muitos tensores vivos na GPU foi o
        # que produziu saida toda preta nos testes iniciais.
        lote = torch.cat([amostrar(palavras_teste[i:i + b], w0, bn).cpu()
                          for i in range(0, len(palavras_teste), b)])
        uma = torch.cat([amostrar([q], w0, bn).cpu() for q in palavras_teste])
        uma2 = torch.cat([amostrar([q], w0, bn).cpu() for q in palavras_teste])
        d = float((lote - uma).abs().max())
        d2 = float((uma - uma2).abs().max())
        dm = float((lote - uma).abs().mean())
        dm2 = float((uma - uma2).abs().mean())
        for i, pal in enumerate(palavras_teste):
            print(f"   {pal:14s} lote={stats(lote[i])}")
            print(f"   {'':14s} indv={stats(uma[i])}")
        print(f"AUTOTESTE lote(--batch {b})-vs-individual : "
              f"max|dif|={d:.5f} mean|dif|={dm:.6f}")
        print(f"AUTOTESTE individual-vs-individual        : "
              f"max|dif|={d2:.5f} mean|dif|={dm2:.6f}")
        vazias = [palavras_teste[i] for i in range(len(palavras_teste))
                  if float(uma[i].std()) < 0.02]
        if vazias:
            raise SystemExit(f"FALHOU: imagens colapsadas (std<0.02): {vazias}")
        # Limiares: um pixel isolado divergindo nao muda decisao nenhuma da
        # metrica; o que invalidaria a medida e a imagem inteira mudar. Por
        # isso o criterio duro e o erro MEDIO, com um teto frouxo no maximo.
        if dm2 > 0.005 or d2 > 0.10:
            raise SystemExit("FALHOU: geracao individual nao e reproduzivel")
        if dm > 0.005 or d > 0.10:
            raise SystemExit("FALHOU: lote != individual. Rode com --batch 1.")
        print("AUTOTESTE ok")
        return

    aquecer(escritores[0])
    print("aquecimento feito", flush=True)

    manifest = open(os.path.join(args_cli.out_dir, "manifest.jsonl"), "w",
                    encoding="utf-8")
    total = 0
    for seed in args_cli.seeds:
        for w in escritores:
            # ruido inicial: depende SO de (seed, escritor). Identico para
            # todas as palavras, logo identico entre os gemeos do par.
            g = torch.Generator(device="cpu").manual_seed(semente_de(seed, "ruido", w))
            base_noise = torch.randn(1, *lat_shape, generator=g).to(device)

            for i0 in range(0, len(itens), args_cli.batch):
                lote = itens[i0:i0 + args_cli.batch]
                n = len(lote)
                textos = [t[0] for t in lote]

                img = amostrar(textos, w, base_noise)

                for k, (palavra, par_id, eh_acc) in enumerate(lote):
                    nome = f"{par_id}__{'acc' if eh_acc else 'asc'}__w{w}__s{seed}.png"
                    caminho = os.path.join(args_cli.out_dir, nome)
                    # Marca no manifest as imagens que sairam quebradas, para a
                    # metrica poder descarta-las em vez de le-las como ausencia
                    # de tinta.
                    nan = bool(torch.isnan(img[k]).any() or torch.isinf(img[k]).any())
                    std = float(img[k].std()) if not nan else 0.0
                    save_image(torch.nan_to_num(img[k]), caminho)
                    manifest.write(json.dumps({
                        "arquivo": nome, "palavra": palavra, "par_id": par_id,
                        "acentuada": eh_acc, "escritor": int(w), "semente": int(seed),
                        "nan": nan, "std": round(std, 5),
                        "colapsada": bool(nan or std < 0.02),
                        "ckpt": os.path.relpath(ckpt, RAIZ),
                        "ddim_steps": args_cli.ddim_steps,
                        "style_seed": args_cli.style_seed,
                    }, ensure_ascii=False) + "\n")
                    total += 1
                manifest.flush()
                print(f"  seed={seed} w={w} {i0 + n}/{len(itens)}", flush=True)
    manifest.close()
    print(f"\npronto: {total} imagens em {args_cli.out_dir}", flush=True)


if __name__ == "__main__":
    main()
