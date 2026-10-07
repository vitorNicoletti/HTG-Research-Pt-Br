"""A base sintetica e menos legivel que a escrita real? (ACHADOS 13: o CER sobe
de 0,22 para 0,32-0,34 em TODO fine-tune na base portuguesa, ja com 4 epocas.)

Le com o leitor independente da avaliacao (modelos/leitor_iam_transformer.pt),
pelo MESMO caminho do avaliar_pt.py (painel 64x256 -> recorte_tinta -> ler):
  iam_real      -- palavras reais do iam_test (escritores que o leitor nao viu)
  base_*        -- imagens da base pt depois do pre-processamento do TREINO
                   (preprocessar_iam -> 64x256), isto e, o que o modelo aprende
  base_*_direto -- as mesmas imagens lidas direto do arquivo (recorte 2x), sem
                   a reducao para 64 px: separa o efeito da reamostragem
Tipos da base: sem_acento (rotulo), par (esqueleto), acentuada (lida contra o
esqueleto, como no avaliar_pt.py). Palavras de 3 a 10 letras, so a-z (minusculas).
Referencia ja medida: o DiffusionPen do IAM gerando palavras pt da validacao
deu CER 0,22 (sem acento) e 0,18 (esqueleto) -- avaliacao_pt_val.

    python diagnostico/cer_base.py --base iam_pt_alinhado --saida diagnostico/resultados/cer_base
"""

import argparse
import json
import os
import random
import re
import sys

import numpy as np
from PIL import Image

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "scripts"))
sys.path.insert(0, os.path.join(RAIZ, "DiffusionPen"))
from acentos_sinteticos import alinhamento  # noqa: E402
from acentos_sinteticos.vocabulario import esqueleto  # noqa: E402
from avaliar_pt import levenshtein, recorte_tinta  # noqa: E402
from utils.iam_acentuado_dataset import preprocessar_iam  # noqa: E402

PALAVRA = re.compile(r"^[a-z]{3,10}$")


def cinza(img):
    return np.asarray(img.convert("L"), dtype=np.float32)


def medir(leitor, itens, via_treino):
    """itens: [(caminho, rotulo_para_preproc, alvo)] -> lista de CER."""
    cers = []
    for caminho, rot, alvo in itens:
        img = Image.open(caminho).convert("RGB")
        g = cinza(preprocessar_iam(img, rot)) if via_treino else cinza(img)
        r = recorte_tinta(g)
        lido = leitor.ler(r) if r is not None else ""
        cers.append(levenshtein(lido, alvo) / max(1, len(alvo)))
    return cers


def resumo(cers):
    c = np.array(cers)
    return {"n": len(c), "cer_medio": round(float(c.mean()), 4), "cer_mediano": round(float(np.median(c)), 4),
            "exatas": round(float((c == 0).mean()), 4), "ic95_media": [
                round(float(np.percentile([c[np.random.default_rng(k).integers(0, len(c), len(c))].mean()
                                           for k in range(2000)], q)), 4) for q in (2.5, 97.5)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--leitor", default=os.path.join(RAIZ, "modelos/leitor_iam_transformer.pt"))
    ap.add_argument("--n", type=int, default=1500, help="amostras por grupo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--saida", required=True)
    a = ap.parse_args()
    os.makedirs(a.saida, exist_ok=True)
    rnd = random.Random(a.seed)
    leitor = alinhamento.Alinhador(a.leitor, "cpu")

    # escrita real: iam_test, escritores fora do treino do leitor e do DiffusionPen
    raiz_iam = os.path.join(RAIZ, "DiffusionPen", "iam_data", "words")
    reais = []
    with open(os.path.join(RAIZ, "DiffusionPen", "utils", "splits_words", "iam_test.txt"), encoding="utf-8") as f:
        for l in f:
            c = l.rstrip("\n").split(",")
            if len(c) >= 3 and PALAVRA.match(",".join(c[2:])):
                reais.append((os.path.join(raiz_iam, c[0]), ",".join(c[2:]), ",".join(c[2:])))
    reais = rnd.sample(reais, min(a.n, len(reais)))

    man = [json.loads(l) for l in open(os.path.join(a.base, "manifesto.jsonl"), encoding="utf-8")]
    grupos = {}
    for tipo in ("sem_acento", "par", "acentuada"):
        it = [(os.path.join(a.base, x["arquivo"]), x["rotulo"], esqueleto(x["rotulo"]))
              for x in man if x["tipo"] == tipo and PALAVRA.match(esqueleto(x["rotulo"]))]
        grupos[tipo] = rnd.sample(it, min(a.n, len(it)))

    rel = {"base": a.base, "leitor": a.leitor, "referencia_avaliacao_pt_val": {
        "iam_original_gerando_pt_sem_acento": 0.22, "iam_original_gerando_pt_esqueleto": 0.18}}
    rel["iam_real"] = resumo(medir(leitor, reais, True))
    print("iam_real", rel["iam_real"], flush=True)
    for tipo, it in grupos.items():
        rel[f"base_{tipo}"] = resumo(medir(leitor, it, True))
        print(f"base_{tipo}", rel[f"base_{tipo}"], flush=True)
        rel[f"base_{tipo}_direto"] = resumo(medir(leitor, it, False))
        print(f"base_{tipo}_direto", rel[f"base_{tipo}_direto"], flush=True)
    with open(os.path.join(a.saida, "cer_base.json"), "w", encoding="utf-8") as f:
        json.dump(rel, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
