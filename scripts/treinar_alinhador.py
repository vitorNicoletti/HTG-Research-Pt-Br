"""Treina o reconhecedor CTC usado para localizar letras (acentos_sinteticos.alinhamento).

Treino em iam_training.txt, validacao em iam_val.txt (splits do clone do
DiffusionPen). Imprime perda e CER/acerto de palavra da leitura livre por
epoca e grava o melhor CER em --saida (pesos + alfabeto + parametros).

    python scripts/treinar_alinhador.py --saida modelos/alinhador_iam.pt
"""

import argparse
import json
import os
import random
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np
import torch

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
from acentos_sinteticos import alinhamento as al, iam  # noqa: E402

ESTICAR = (0.8, 1.2)     # aumento de dados: largura sorteada
LARGURA_LOTE = 128       # px; a largura de cada lote e multiplo disto


def _carregar(caminho):
    try:
        im, _ = al.normalizar(iam.carregar_cinza(caminho))
    except Exception:
        return None
    return (im * 255).astype(np.uint8)


def carregar(palavras, workers):
    with Pool(workers) as p:
        ims = p.map(_carregar, [x.caminho for x in palavras], chunksize=256)
    return [(im, x.texto) for im, x in zip(ims, palavras) if im is not None]


def lotes(itens, batch, rnd):
    """Indices agrupados por largura parecida (pouco preenchimento), em ordem aleatoria."""
    ordem = sorted(range(len(itens)), key=lambda i: itens[i][0].shape[1] * rnd.uniform(0.9, 1.1))
    grupos = [ordem[k:k + batch] for k in range(0, len(ordem), batch)]
    rnd.shuffle(grupos)
    return grupos


def montar(itens, idx, indice, rnd=None):
    ims, alvos, t_ent, t_alvo = [], [], [], []
    for i in idx:
        im, texto = itens[i]
        if rnd is not None:
            nl = int(round(im.shape[1] * rnd.uniform(*ESTICAR)))
            nl = min(max(nl, 2 * al.PASSO), al.LARGURA_MAX)
            nl += (-nl) % al.PASSO
            im = cv2.resize(im, (nl, al.ALTURA), interpolation=cv2.INTER_AREA)
        ims.append(im)
        alvos += [indice[c] for c in texto]
        t_ent.append(im.shape[1] // al.PASSO)
        t_alvo.append(len(texto))
    # largura do lote arredondada para poucos tamanhos fixos: cada formato novo
    # de entrada faz o MIOpen compilar e procurar kernels de novo
    lmax = max(im.shape[1] for im in ims)
    lmax += (-lmax) % LARGURA_LOTE
    x = np.zeros((len(ims), 1, al.ALTURA, lmax), dtype=np.float32)
    for k, im in enumerate(ims):
        x[k, 0, :, :im.shape[1]] = im / 255.0
    return (torch.from_numpy(x), torch.tensor(alvos), torch.tensor(t_ent), torch.tensor(t_alvo))


def levenshtein(a, b):
    ant = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(ant[j] + 1, cur[j - 1] + 1, ant[j - 1] + (ca != cb)))
        ant = cur
    return ant[-1]


def validar(modelo, itens, indice, alfabeto, device, batch):
    modelo.eval()
    erros = chars = certas = 0
    with torch.no_grad():
        for idx in lotes(itens, batch, random.Random(0)):
            x, _, t_ent, _ = montar(itens, idx, indice)
            y = modelo(x.to(device)).float().cpu().numpy()
            for k, i in enumerate(idx):
                lido = al.decodificar(y[:t_ent[k], k], alfabeto)
                alvo = itens[i][1]
                erros += levenshtein(lido, alvo)
                chars += len(alvo)
                certas += lido == alvo
    modelo.train()
    return erros / max(1, chars), certas / max(1, len(itens))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clone", default=os.path.join(RAIZ, "DiffusionPen"))
    ap.add_argument("--saida", default=os.path.join(RAIZ, "modelos", "alinhador_iam.pt"))
    ap.add_argument("--epocas", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if a.device.startswith("cuda") and not torch.cuda.is_available():
        sys.exit("ERRO: --device cuda pedido mas a GPU nao esta disponivel (sem fallback para CPU)")
    torch.manual_seed(a.seed)
    rnd = random.Random(a.seed)

    t0 = time.time()
    treino = carregar(iam.listar(a.clone, "iam_training.txt"), a.workers)
    val = carregar(iam.listar(a.clone, "iam_val.txt"), a.workers)
    alfabeto = sorted({c for _, t in treino + val for c in t})
    indice = {c: k + 1 for k, c in enumerate(alfabeto)}
    # o CTC precisa de quadros >= letras (+ repeticoes); descarta o resto
    def cabe(it):
        t = it[1]
        return t and it[0].shape[1] // al.PASSO >= len(t) + sum(x == y for x, y in zip(t, t[1:]))
    treino, val = [x for x in treino if cabe(x)], [x for x in val if cabe(x)]
    print(f"treino {len(treino)}  val {len(val)}  alfabeto {len(alfabeto)}  "
          f"carga {time.time() - t0:.0f}s", flush=True)

    modelo = al.criar_modelo(len(alfabeto) + 1).to(a.device)
    opt = torch.optim.AdamW(modelo.parameters(), lr=a.lr, weight_decay=1e-4)
    passos = a.epocas * ((len(treino) + a.batch - 1) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=passos, pct_start=0.1)
    ctc = torch.nn.CTCLoss(blank=al.BRANCO, zero_infinity=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.saida)), exist_ok=True)
    melhor = None
    for ep in range(1, a.epocas + 1):
        t0, soma, n = time.time(), 0.0, 0
        for idx in lotes(treino, a.batch, rnd):
            x, alvos, t_ent, t_alvo = montar(treino, idx, indice, rnd)
            y = modelo(x.to(a.device))
            perda = ctc(y.float(), alvos, t_ent, t_alvo)
            opt.zero_grad()
            perda.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 5.0)
            opt.step()
            sched.step()
            soma += float(perda.detach())
            n += 1
        cer, acerto = validar(modelo, val, indice, alfabeto, a.device, a.batch)
        marca = ""
        if melhor is None or cer < melhor:
            melhor = cer
            torch.save({"estado": modelo.state_dict(), "alfabeto": alfabeto,
                        "altura": al.ALTURA, "passo": al.PASSO, "epoca": ep,
                        "cer_val": cer, "acerto_val": acerto, "args": vars(a)}, a.saida)
            marca = "  *salvo"
        print(f"epoca {ep:2d}  perda {soma / max(1, n):.3f}  CER val {cer:.4f}  "
              f"palavras certas {acerto:.3f}  {time.time() - t0:.0f}s{marca}", flush=True)
    print(json.dumps({"melhor_cer_val": melhor, "saida": a.saida}))


if __name__ == "__main__":
    main()
