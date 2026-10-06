"""Confere a mascara da zona vazia (--peso_zona) ANTES de treinar com ela.

A zona sao as celulas do latente inteiramente fora do corpo da palavra, sem
tinta e dentro das colunas da palavra, na imagem SEM sinal (mascara_zona em
diffusionpen_mods/utils/iam_acentuado_dataset.py). Confere, em TODA a base:
  1. a zona nunca encosta no acento (disjunta da mascara do acento);
  2. acentuada e par tem a mesma zona; originais do IAM, zona zerada;
  3. a zona nao cobre tinta do alvo: na acentuada, so o sinal poderia cair
     nela, e o sinal ja esta na mascara do acento -- conta pixels de tinta do
     alvo dentro da zona;
  4. quanto do latente a zona cobre (por tipo e no lote) e quanto da loss ela
     passa a pesar com o peso pedido;
  5. figura: zona (azul) e acento (vermelho) sobre a imagem.
Carrega o dataset de diffusionpen_mods/ (nao do clone), para poder rodar
enquanto um treino usa o clone.

    python diagnostico/conferir_zona.py --base iam_pt_alinhado --peso_zona 2 \
        --saida diagnostico/resultados/peso_zona
"""

import argparse
import importlib.util
import json
import os
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "DiffusionPen"))     # utils.auxilary_functions
esp = importlib.util.spec_from_file_location(
    "iam_acentuado_mods", os.path.join(RAIZ, "diffusionpen_mods", "utils", "iam_acentuado_dataset.py"))
mods = importlib.util.module_from_spec(esp)
esp.loader.exec_module(mods)
# fora do clone o modulo acharia os arquivos do IAM em diffusionpen_mods/; aponta para o clone
CLONE = os.path.join(RAIZ, "DiffusionPen")
mods.SPLIT_ORIGINAIS = os.path.join(CLONE, "utils", "splits_words", "iam_train_val.txt")
mods.WRITERS_DICT = os.path.join(CLONE, "writers_dict_train.json")
os.environ.setdefault("IAM_IMAGES", os.path.join(CLONE, "iam_data", "words"))
from acentos_sinteticos import geometria  # noqa: E402


def sobrepor(img, acento, zona):
    h, w = mods.FORMA_LATENTE
    sob = img.convert("RGB").copy()
    d = ImageDraw.Draw(sob, "RGBA")
    for m, cor in ((zona, (0, 90, 255, 60)), (acento, (255, 0, 0, 80))):
        for y, x in zip(*np.nonzero(m)):
            d.rectangle([x * 256 // w, y * 64 // h, (x + 1) * 256 // w - 1, (y + 1) * 64 // h - 1], fill=cor)
    return sob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--peso_acento", type=float, default=5.0)
    ap.add_argument("--peso_zona", type=float, default=2.0)
    ap.add_argument("--iam_originais", type=float, default=1.0)
    ap.add_argument("--n_figura", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--saida", required=True)
    a = ap.parse_args()
    os.makedirs(a.saida, exist_ok=True)

    args = SimpleNamespace(peso_acento=a.peso_acento, peso_zona=a.peso_zona,
                           iam_originais=a.iam_originais, max_samples=0)
    ds = mods.IAMAcentuadoDataset(a.base, transforms=None, args=args)
    man = [json.loads(l) for l in open(os.path.join(a.base, "manifesto.jsonl"), encoding="utf-8")]
    tipo = {os.path.join(a.base, x["arquivo"]): x["tipo"] for x in man}

    cob = {"acentuada": [], "par": [], "sem_acento": []}
    tinta_na_zona = {"acentuada": 0, "par": 0, "sem_acento": 0}
    problemas, zonas, figura = [], {}, []
    for c, _, t, sint in ds.data:
        if not sint:
            continue
        tp = tipo[c]
        ac = ds.mascara(c)
        z = ds.zona(c, t, True, ac)
        cob[tp].append(float(z.mean()))
        if (z * ac).any():
            problemas.append(f"zona encosta no acento: {c}")
        # tinta do alvo dentro da zona (px), na resolucao do painel
        g = np.asarray(mods.preprocessar_iam(Image.open(c).convert("RGB"), t).convert("L"), dtype=np.float32)
        tinta = geometria.mascara_tinta(g)
        zpx = np.kron(z.numpy(), np.ones((64 // mods.FORMA_LATENTE[0], 256 // mods.FORMA_LATENTE[1]))) > 0
        tinta_na_zona[tp] += int((tinta & zpx).sum())
        if tp in ("acentuada", "par"):
            zonas[c] = z
        figura.append((c, t, g, ac.numpy() > 0, z.numpy() > 0))
    for c, ((ca, _), (cp, _)) in ds.pares.items():
        if c == ca and not torch.equal(zonas[ca], zonas[cp]):
            problemas.append(f"zona do par difere da acentuada: {ca}")
    # originais do IAM: zona zerada
    orig = next(d for d in ds.data if not d[3])
    if ds.zona(orig[0], orig[2], False, torch.zeros(mods.FORMA_LATENTE)).any():
        problemas.append("original do IAM com zona")

    n_sint = sum(len(v) for v in cob.values())
    soma = {k: sum(v) for k, v in cob.items()}
    cob_lote = sum(soma.values()) / len(ds.data)
    cob_ac_lote = 0.0
    for c, _, t, sint in ds.data:
        if sint and c in ds.pares:
            cob_ac_lote += float(ds.mascara(c).mean())
    cob_ac_lote /= len(ds.data)
    um = 1 - cob_lote - cob_ac_lote
    total = um + a.peso_acento * cob_ac_lote + a.peso_zona * cob_lote
    rel = {
        "base": a.base, "peso_acento": a.peso_acento, "peso_zona": a.peso_zona, "iam_originais": a.iam_originais,
        "amostras": {k: len(v) for k, v in cob.items()} | {"originais_iam": ds.n_originais},
        "zona_por_amostra": {k: {"media": round(float(np.mean(v)), 4), "mediana": round(float(np.median(v)), 4),
                                 "vazia": round(float(np.mean(np.array(v) == 0)), 4)} for k, v in cob.items()},
        "zona_no_lote": round(cob_lote, 4), "acento_no_lote": round(cob_ac_lote, 4),
        "parte_da_loss_do_lote": {"zona": round(a.peso_zona * cob_lote / total, 4),
                                  "acento": round(a.peso_acento * cob_ac_lote / total, 4),
                                  "resto": round(um / total, 4)},
        "aumento_da_loss_media": round(total, 4),
        "px_de_tinta_do_alvo_na_zona": tinta_na_zona,
        "problemas": problemas[:50], "n_problemas": len(problemas),
    }
    with open(os.path.join(a.saida, "zona.json"), "w", encoding="utf-8") as f:
        json.dump(rel, f, ensure_ascii=False, indent=2)
    print(json.dumps(rel, ensure_ascii=False, indent=2))

    rnd = random.Random(a.seed)
    amostra = rnd.sample([f for f in figura if tipo[f[0]] == "acentuada"], a.n_figura // 3) + \
        rnd.sample([f for f in figura if tipo[f[0]] == "sem_acento"], a.n_figura - a.n_figura // 3)
    tela = Image.new("RGB", (2 * (256 + 8), 32 + (len(amostra) + 1) // 2 * 76), "white")
    d = ImageDraw.Draw(tela)
    d.text((4, 4), "azul = zona vazia (peso_zona); vermelho = acento (peso_acento)", fill="black")
    for i, (c, t, g, ac, z) in enumerate(amostra):
        x, y = (i % 2) * (256 + 8), 24 + (i // 2) * 76
        tela.paste(sobrepor(Image.fromarray(g.astype(np.uint8)), ac, z), (x, y + 10))
        d.text((x + 2, y), f"{tipo[c]}: {t}", fill="black")
    tela.save(os.path.join(a.saida, "zona.png"))
    if problemas:
        sys.exit(f"ERRO: {len(problemas)} problemas")
    print("ok")


if __name__ == "__main__":
    main()
