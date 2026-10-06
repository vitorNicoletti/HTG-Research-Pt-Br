"""Acrescenta o par SEM acento a uma base de acentos sinteticos sobre o IAM real.

A base de scripts/gerar_base_acentos.py so tem as acentuadas ("thé"). O par e a
propria palavra original do IAM ("the"), posta na tela da acentuada com
gerador.par_na_tela e as margens que o desenho acrescentou (params.margens =
[cima, baixo, esq, dir]). Com o par, a base serve ao --peso_acento e ao
--peso_zona do train.py (mascara = diferenca acentuada x par).

Diferente da base portuguesa (scripts/gerar_base_pt.py), nada aqui e gerado
pelo DiffusionPen: acentuada e par sao escrita real do IAM. A geracao estragava
a letra (ACHADOS.md, secao 13, adendo 3).

Saida em --destino: acentuadas por hardlink, imagens/NNNNNN_par.png novas,
split.txt com as duas, manifesto.jsonl com tipo/tarefa (o formato da base pt) e
resumo.json com pares_alinhados. A origem nao e alterada.

    python scripts/criar_pares_iam.py --origem iam_acentuado_teto25 \
        --destino iam_acentuado_teto25_pares
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np
from PIL import Image

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
from acentos_sinteticos import gerador, iam  # noqa: E402

LIMIAR_PIXEL = 40   # = utils/iam_acentuado_dataset.py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origem", required=True)
    ap.add_argument("--destino", required=True)
    ap.add_argument("--clone", default=os.path.join(RAIZ, "DiffusionPen"))
    a = ap.parse_args()
    if os.path.exists(a.destino):
        sys.exit(f"ERRO: {a.destino} ja existe")
    os.makedirs(os.path.join(a.destino, "imagens"))

    man = [json.loads(l) for l in open(os.path.join(a.origem, "manifesto.jsonl"), encoding="utf-8")]
    novo, linhas, fracoes = [], [], []
    for x in man:
        nome = x["arquivo"]
        tarefa = int(os.path.splitext(os.path.basename(nome))[0])
        os.link(os.path.join(a.origem, nome), os.path.join(a.destino, nome))
        ac = np.asarray(Image.open(os.path.join(a.destino, nome)).convert("L"), dtype=np.float32)
        orig = iam.carregar_cinza(os.path.join(a.clone, x["iam"]))
        cima, baixo, esq, dir_ = x["params"]["margens"]
        if orig.shape[0] + cima + baixo != ac.shape[0] or orig.shape[1] + esq + dir_ != ac.shape[1]:
            sys.exit(f"ERRO: {nome}: original {orig.shape} + margens {x['params']['margens']} != acentuada {ac.shape}")
        par = gerador.par_na_tela(orig, ac.shape, [esq, cima])
        nome_par = nome.replace(".png", "_par.png")
        Image.fromarray(np.clip(par, 0, 255).astype(np.uint8)).save(os.path.join(a.destino, nome_par))
        fracoes.append(float((np.abs(ac - np.clip(par, 0, 255).astype(np.uint8)) > LIMIAR_PIXEL).mean()))

        base = {"escritor": x["escritor"], "tarefa": tarefa, "iam": x["iam"], "semente": x["semente"]}
        novo.append({"arquivo": nome, "rotulo": x["rotulo"], "tipo": "acentuada", **base,
                     "original": x["original"], "indice": x["indice"], "letra": x["letra"],
                     "tipo_sinal": x["tipo"], "params": x["params"]})
        novo.append({"arquivo": nome_par, "rotulo": x["original"], "tipo": "par", **base})
        linhas.append(f"{nome_par},{x['escritor']},{x['original']}")
        linhas.append(f"{nome},{x['escritor']},{x['rotulo']}")

    with open(os.path.join(a.destino, "split.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(linhas) + "\n")
    with open(os.path.join(a.destino, "manifesto.jsonl"), "w", encoding="utf-8") as f:
        f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in novo)
    for arq in ("descartes.tsv",):
        if os.path.isfile(os.path.join(a.origem, arq)):
            shutil.copyfile(os.path.join(a.origem, arq), os.path.join(a.destino, arq))
    resumo = json.load(open(os.path.join(a.origem, "resumo.json"), encoding="utf-8"))
    fr = np.array(fracoes)
    resumo.update({"origem": os.path.abspath(a.origem), "pares_alinhados": True, "pares": len(man),
                   "diferenca_acentuada_par_px": {"mediana": round(float(np.median(fr)), 5),
                                                  "p95": round(float(np.percentile(fr, 95)), 5),
                                                  "max": round(float(fr.max()), 5)}})
    json.dump(resumo, open(os.path.join(a.destino, "resumo.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"{len(man)} pares; pixels diferentes acentuada x par (tela original): "
          f"mediana {np.median(fr):.3%}, p95 {np.percentile(fr, 95):.3%}, max {fr.max():.3%}")


if __name__ == "__main__":
    main()
