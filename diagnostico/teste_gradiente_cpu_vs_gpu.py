"""
testar_gpu.py -- a GPU esta corrompendo o treino? Prova ou inocenta.

Metodo, sem intuicao: a CPU nao usa MIOpen. Entao rodamos o MESMO batch, com
os MESMOS pesos, na GPU e na CPU, e comparamos os gradientes.

Dois testes:

  1. ERRO RELATIVO. Para batches normais, compara gradiente GPU vs CPU.
     - ~1e-6 a 1e-5  -> diferenca normal de fp32 (ordem de reducao distinta)
     - >= 1e-2       -> a GPU esta calculando gradiente errado
     Os limiares estao declarados ANTES de ver o resultado, de proposito.

  2. O TESTE DECISIVO. Captura um batch em que a GPU produz gradiente
     nao-finito e recalcula esse batch exato na CPU. Se a CPU devolver um
     gradiente finito e sao, a GPU esta provada culpada -- mesmo dado, mesmos
     pesos, resultados incompativeis.

Uso:
    python testar_gpu.py                 # usa o checkpoint do v2
    CKPT_DIR=./model_bressay_full_12ep/models python testar_gpu.py

Precisa da GPU livre (pare o treino antes). Leva ~10-20 min: o passo na CPU
e lento, por isso rodamos poucos batches.
"""

import os
import sys
import argparse

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diffusionpen", "DiffusionPen")
sys.path.insert(0, REPO)

import torch
import torch.nn as nn
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CanineModel, CanineTokenizer

from unet import UNetModel
from feature_extractor import ImageEncoder
from utils.bressay_dataset import BRESSAY_Dataset

CKPT_DIR = os.environ.get("CKPT_DIR", "")
STYLE_PATH = "./style_models/mixed_bressay_mobilenetv2_100.pth"
STABLE_DIF = "runwayml/stable-diffusion-v1-5"
MAX_BATCHES = int(os.environ.get("MAX_BATCHES", "25"))
N_COMPARACOES = int(os.environ.get("N_COMPARACOES", "3"))


class Passa(nn.Module):
    """Repassa a chamada para o modulo interno.

    Serve para o lado CPU: reproduz exatamente o prefixo "module." que o
    DataParallel poe nas chaves do state_dict (e que o checkpoint espera),
    sem envolver CUDA. DataParallel(device_ids=[]) nao serve porque estoura
    no construtor quando ha CUDA na maquina.
    """

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *a, **kw):
        return self.module(*a, **kw)


def build_args(device):
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
    a.color = True
    a.level = "word"
    a.unet = "unet_latent"
    a.batch_size = 32
    a.max_samples = 0
    a.dataset = "bressay"
    return a


def monta(device, text_encoder):
    args = build_args(device)
    unet = UNetModel(
        image_size=args.img_size, in_channels=4, model_channels=320, out_channels=4,
        num_res_blocks=1, attention_resolutions=(1, 1), channel_mult=(1, 1),
        num_heads=4, num_classes=339, context_dim=320, vocab_size=79,
        text_encoder=text_encoder, args=args,
    )
    unet = DataParallel(unet, device_ids=[0]) if device != "cpu" else Passa(unet)
    unet = unet.to(device)
    unet.load_state_dict(torch.load(f"{CKPT_DIR}/ckpt.pt", map_location=device, weights_only=True))
    # eval() e OBRIGATORIO aqui: o CANINE fica aninhado dentro do UNet e tem
    # dropout 0.1. Em train() cada lado sorteia mascaras diferentes, os
    # forwards passam a computar coisas distintas e a comparacao vira lixo
    # (foi o que aconteceu na primeira versao deste teste: loss 0.176 na GPU
    # contra 0.196 na CPU). O backward continua funcionando em eval.
    unet.eval()
    return unet


def grads(unet, x_t, t, txt, s_id, sf, noise, mse):
    unet.zero_grad(set_to_none=True)
    pred = unet(x_t, timesteps=t, context=txt, y=s_id, style_extractor=sf)
    loss = mse(noise, pred)
    if not torch.isfinite(loss):
        return None, float("nan"), {}
    loss.backward()
    g = {n: p.grad.detach().float().cpu().clone()
         for n, p in unet.named_parameters() if p.grad is not None}
    finito = all(torch.isfinite(v).all() for v in g.values())
    unet.zero_grad(set_to_none=True)
    return finito, float(loss.detach()), g


def erro_relativo(ga, gb):
    num = den = 0.0
    pior = (0.0, "")
    for k, v in ga.items():
        if k not in gb:
            continue
        d = float((v - gb[k]).pow(2).sum())
        n = float(v.pow(2).sum())
        num += d
        den += n
        if n > 0:
            r = (d / n) ** 0.5
            if r > pior[0]:
                pior = (r, k)
    return (num / den) ** 0.5 if den else float("nan"), pior


def main():
    print(f"checkpoint: {CKPT_DIR}\n")
    tok = CanineTokenizer.from_pretrained("google/canine-c")

    te_gpu = DataParallel(CanineModel.from_pretrained("google/canine-c"), device_ids=[0]).to("cuda:0")
    te_gpu.requires_grad_(False); te_gpu.eval()
    te_cpu = Passa(CanineModel.from_pretrained("google/canine-c")).to("cpu")
    te_cpu.requires_grad_(False); te_cpu.eval()

    print("montando modelo na GPU e na CPU (mesmos pesos)...")
    unet_gpu = monta("cuda:0", te_gpu)
    unet_cpu = monta("cpu", te_cpu)

    vae = DataParallel(AutoencoderKL.from_pretrained(STABLE_DIF, subfolder="vae"), device_ids=[0]).to("cuda:0")
    vae.requires_grad_(False)

    fe = ImageEncoder(model_name="mobilenetv2_100", num_classes=0, pretrained=True, trainable=True)
    md = fe.state_dict()
    sd = torch.load(STYLE_PATH, map_location="cpu", weights_only=True)
    md.update({k: v for k, v in sd.items() if k in md and md[k].shape == v.shape})
    fe.load_state_dict(md)
    fe_gpu = DataParallel(fe, device_ids=[0]).to("cuda:0")
    fe_gpu.requires_grad_(False); fe_gpu.eval()

    sched = DDIMScheduler.from_pretrained(STABLE_DIF, subfolder="scheduler")
    mse = nn.MSELoss()

    tf = transforms.Compose([transforms.ToTensor(),
                             torchvision.transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
    ds = BRESSAY_Dataset("./bressay_split", "train", transforms=tf, args=build_args("cpu"))
    dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=8)

    # ---- Fase 0: a GPU concorda consigo mesma? ----
    # Mesmo batch, mesmos pesos, duas chamadas seguidas no MESMO processo. Se
    # divergirem, a placa e nao-deterministica e isso sozinho explica tudo --
    # nao precisa nem comparar com a CPU.
    print("Fase 0: GPU vs GPU (mesmo batch, duas vezes seguidas)")
    for i, data in enumerate(DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)):
        if i >= 3:
            break
        txt = tok(data[1], padding="max_length", truncation=True, return_tensors="pt", max_length=40)
        txt = {k: v.to("cuda:0") for k, v in txt.items()}
        sf = fe_gpu(data[3].to("cuda:0").reshape(-1, 3, 64, 256))
        lat = vae.module.encode(data[0].to("cuda:0").to(torch.float32)).latent_dist.sample() * 0.18215
        nz = torch.randn(lat.shape, device="cuda:0")
        tt = torch.randint(0, 1000, (lat.shape[0],), device="cuda:0").long()
        xt = sched.add_noise(lat, nz, tt)
        sid = data[2].to("cuda:0")

        f1, l1, g1 = grads(unet_gpu, xt, tt, txt, sid, sf, nz, mse)
        f2, l2, g2 = grads(unet_gpu, xt, tt, txt, sid, sf, nz, mse)
        rel, (pr, pk) = erro_relativo(g1, g2)
        print(f"  batch {i}: loss {l1:.8f} vs {l2:.8f} (dif {abs(l1-l2):.2e}) | "
              f"erro relativo do gradiente GPU-vs-GPU: {rel:.2e}")
        if rel > 1e-4:
            print(f"    -> NAO-DETERMINISTICA. pior tensor: {pr:.2e} em {pk}")
    print()

    normais, ruins = [], []
    torch.manual_seed(0)

    for i, data in enumerate(dl):
        if i >= MAX_BATCHES or (len(normais) >= N_COMPARACOES and ruins):
            break
        imgs = data[0]; transcr = data[1]; s_id = data[2]; style = data[3]

        # Entradas montadas UMA vez na GPU e copiadas para a CPU: assim os dois
        # lados recebem exatamente os mesmos numeros, e a unica variavel e onde
        # o forward/backward do unet roda.
        txt_g = tok(transcr, padding="max_length", truncation=True, return_tensors="pt", max_length=40)
        sf_g = fe_gpu(style.to("cuda:0").reshape(-1, 3, 64, 256))
        lat = vae.module.encode(imgs.to("cuda:0").to(torch.float32)).latent_dist.sample() * 0.18215
        noise = torch.randn(lat.shape, device="cuda:0")
        t = torch.randint(0, 1000, (lat.shape[0],), device="cuda:0").long()
        x_t = sched.add_noise(lat, noise, t)

        fin_g, loss_g, g_gpu = grads(
            unet_gpu, x_t, t, {k: v.to("cuda:0") for k, v in txt_g.items()},
            s_id.to("cuda:0"), sf_g, noise, mse)

        if fin_g is False and not ruins:
            print(f"\n>>> batch {i}: GPU produziu gradiente NAO-FINITO (loss={loss_g:.4f}, finito)")
            print("    recalculando este batch exato na CPU (demora alguns minutos)...")
            fin_c, loss_c, g_cpu = grads(
                unet_cpu, x_t.cpu(), t.cpu(), {k: v.cpu() for k, v in txt_g.items()},
                s_id.cpu(), sf_g.cpu(), noise.cpu(), mse)
            print(f"    CPU: loss={loss_c:.4f}, gradiente finito={fin_c}")
            if fin_c:
                gn = torch.sqrt(sum(v.pow(2).sum() for v in g_cpu.values()))
                print(f"    norma do gradiente na CPU: {float(gn):.2f}")
                print("\n    *** GPU PROVADA CULPADA: mesmo batch, mesmos pesos,")
                print("        CPU devolve gradiente finito e sao, GPU devolve lixo. ***")
            else:
                print("\n    Os dois lados divergem: o batch e patologico, nao a GPU.")
            ruins.append(i)
            continue

        if fin_g and len(normais) < N_COMPARACOES:
            print(f"\nbatch {i}: comparando gradiente GPU vs CPU (demora alguns minutos)...")
            fin_c, loss_c, g_cpu = grads(
                unet_cpu, x_t.cpu(), t.cpu(), {k: v.cpu() for k, v in txt_g.items()},
                s_id.cpu(), sf_g.cpu(), noise.cpu(), mse)
            rel, (pior_r, pior_k) = erro_relativo(g_cpu, g_gpu)
            print(f"  loss  GPU {loss_g:.6f} | CPU {loss_c:.6f} | dif {abs(loss_g-loss_c):.2e}")
            print(f"  erro relativo do gradiente (GPU vs CPU): {rel:.2e}")
            print(f"  pior tensor: {pior_r:.2e} em {pior_k}")
            veredito = ("normal para fp32" if rel < 1e-4 else
                        "SUSPEITO" if rel < 1e-2 else "GPU CALCULANDO ERRADO")
            print(f"  -> {veredito}")
            normais.append(rel)

    print("\n" + "=" * 60)
    if normais:
        print(f"batches normais comparados: {len(normais)} | erro relativo medio: "
              f"{sum(normais)/len(normais):.2e}")
    if ruins:
        print(f"batches em que a GPU deu gradiente nao-finito: {ruins}")
    else:
        print("nenhum gradiente nao-finito apareceu nesta amostra "
              f"de {MAX_BATCHES} batches -- rode de novo com MAX_BATCHES maior")


if __name__ == "__main__":
    main()
