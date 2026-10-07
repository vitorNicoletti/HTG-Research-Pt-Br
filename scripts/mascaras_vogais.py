"""Pre-calcula a mascara de "lugares de acento possivel" (acentos_sinteticos.zona_vogais)
de TODA amostra que o treino iam_acentuado pode ver, e grava em <base>/zona_vogais.npz.

  originais -- toda palavra do iam_train_val (chave "iam:<caminho do split>"),
               no painel 64x256 do pre-processamento do treino;
  base      -- acentuada e par recebem a MESMA mascara, calculada no PAR com o
               rotulo sem acento, sem as letras do acento pedido (chave
               "base:<arquivo>"); sem_acento da base pt, na propria imagem.
O alinhador CTC roda uma vez aqui (CPU, varios processos), nao a cada epoca. O
dataset le o .npz com --zona vogais.

    python scripts/mascaras_vogais.py --base iam_acentuado_teto25_pares
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np
from PIL import Image

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
CLONE = os.path.join(RAIZ, "DiffusionPen")
_ALIN = None


def _iniciar(caminho_alinhador):
    global _ALIN
    import cv2
    import torch
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    from acentos_sinteticos import alinhamento
    _ALIN = alinhamento.Alinhador(caminho_alinhador, "cpu")


def _processar(item):
    chave, caminho, texto, pular = item
    sys.path.insert(0, CLONE)
    from utils.iam_acentuado_dataset import preprocessar_iam
    from acentos_sinteticos import zona_vogais
    try:
        img = preprocessar_iam(Image.open(caminho).convert("RGB"), texto)
    except OSError:
        return chave, np.zeros(zona_vogais.FORMA, dtype=bool), "imagem_ilegivel"
    g = np.asarray(img.convert("L"), dtype=np.float32)
    m, motivo = zona_vogais.mascara(g, texto, _ALIN, set(pular))
    return chave, m, motivo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--alinhador", default=os.path.join(RAIZ, "modelos/alinhador_iam.pt"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max", type=int, default=0, help="limite de itens (teste rapido)")
    a = ap.parse_args()
    destino = os.path.join(a.base, "zona_vogais.npz")
    if os.path.exists(destino):
        sys.exit(f"ERRO: {destino} ja existe")

    itens = []
    raiz_iam = os.environ.get("IAM_IMAGES", os.path.join(CLONE, "iam_data", "words"))
    with open(os.path.join(CLONE, "utils", "splits_words", "iam_train_val.txt"), encoding="utf-8") as f:
        for l in f:
            p = l.rstrip("\n").split(",")
            if len(p) >= 3:
                itens.append(("iam:" + p[0], os.path.join(raiz_iam, p[0]), ",".join(p[2:]), ()))
    man = [json.loads(l) for l in open(os.path.join(a.base, "manifesto.jsonl"), encoding="utf-8")]
    pares = {}
    for x in man:
        if x["tipo"] in ("acentuada", "par"):
            pares.setdefault(x["tarefa"], {})[x["tipo"]] = x
        elif x["tipo"] == "sem_acento":
            itens.append(("base:" + x["arquivo"], os.path.join(a.base, x["arquivo"]), x["rotulo"], ()))
    copias = {}   # chave da acentuada -> chave do par (mesma mascara)
    for t, d in pares.items():
        ac, par = d["acentuada"], d["par"]
        pular = tuple(i for i, (c1, c2) in enumerate(zip(ac["rotulo"], par["rotulo"])) if c1 != c2)
        itens.append(("base:" + par["arquivo"], os.path.join(a.base, par["arquivo"]), par["rotulo"], pular))
        copias["base:" + ac["arquivo"]] = "base:" + par["arquivo"]
    if a.max:
        itens = itens[:a.max]
    print(f"{len(itens)} mascaras a calcular (+ {len(copias)} acentuadas copiadas do par)", flush=True)

    chaves, mascaras, motivos = [], [], Counter()
    t0 = time.time()
    with Pool(a.workers, initializer=_iniciar, initargs=(a.alinhador,)) as pool:
        for n, (chave, m, motivo) in enumerate(pool.imap(_processar, itens, chunksize=32), 1):
            chaves.append(chave)
            mascaras.append(np.packbits(m.reshape(-1)))
            motivos[(chave.split(":")[0], motivo)] += 1
            if n % 5000 == 0:
                print(f"  {n}/{len(itens)}  {time.time() - t0:.0f}s", flush=True)
    por_chave = dict(zip(chaves, mascaras))
    for ac, par in copias.items():
        if par in por_chave:
            chaves.append(ac)
            mascaras.append(por_chave[par])
    np.savez_compressed(destino, chaves=np.array(chaves), mascaras=np.stack(mascaras))
    resumo = {"itens": len(chaves), "motivos": {f"{k[0]}/{k[1]}": v for k, v in sorted(motivos.items())},
              "alinhador": a.alinhador, "segundos": round(time.time() - t0)}
    with open(os.path.join(a.base, "zona_vogais.json"), "w", encoding="utf-8") as f:
        json.dump(resumo, f, ensure_ascii=False, indent=2)
    print(json.dumps(resumo, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
