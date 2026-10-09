"""Cria uma copia da base portuguesa com os pares alinhados a acentuada.

Na base gerada antes da correcao (iam_pt), o par sem acento ficou na tela
original, menor que a da acentuada quando um sinal passou da borda, e as
duas versoes chegam ao treino em escalas diferentes (ACHADOS.md, secao 12).
Este script refaz SO os _par.png, pondo cada um na tela da sua acentuada com
gerador.par_na_tela (a mesma funcao que gerar_base_pt.py passou a usar). As
demais imagens sao hardlinks; split.txt e manifesto sao copiados. A base de
origem nao e alterada -- model_iam_pt foi treinado nela.

    python scripts/alinhar_pares.py --origem iam_pt --destino iam_pt_alinhado
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
from acentos_sinteticos import gerador  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origem", required=True)
    ap.add_argument("--destino", required=True)
    a = ap.parse_args()
    if os.path.exists(a.destino):
        sys.exit(f"ERRO: {a.destino} ja existe")
    os.makedirs(os.path.join(a.destino, "imagens"))

    man = [json.loads(l) for l in open(os.path.join(a.origem, "manifesto.jsonl"), encoding="utf-8")]
    acentuada = {x["tarefa"]: x for x in man if x["tipo"] == "acentuada"}
    n_refeitos = n_ja_alinhados = 0
    for x in man:
        org = os.path.join(a.origem, x["arquivo"])
        dst = os.path.join(a.destino, x["arquivo"])
        if x["tipo"] != "par":
            os.link(org, dst)
            continue
        ac = acentuada[x["tarefa"]]
        forma = Image.open(os.path.join(a.origem, ac["arquivo"])).size[::-1]   # (H, W)
        g = np.asarray(Image.open(org).convert("L"), dtype=np.float32)
        if g.shape == tuple(forma):
            os.link(org, dst)
            n_ja_alinhados += 1
            continue
        par = gerador.par_na_tela(g, forma, x["margens_esq_cima"])
        Image.fromarray(np.clip(par, 0, 255).astype(np.uint8)).save(dst)
        n_refeitos += 1

    # conferencia: todo par agora tem o tamanho da sua acentuada
    tam = {x["tarefa"]: Image.open(os.path.join(a.destino, x["arquivo"])).size
           for x in man if x["tipo"] == "par"}
    ruins = [t for t, ac in acentuada.items()
             if Image.open(os.path.join(a.destino, ac["arquivo"])).size != tam[t]]
    if ruins:
        sys.exit(f"ERRO: {len(ruins)} pares continuam com tamanho diferente: {ruins[:10]}")

    for arq in ("split.txt", "manifesto.jsonl", "descartes.tsv"):
        shutil.copyfile(os.path.join(a.origem, arq), os.path.join(a.destino, arq))
    resumo = json.load(open(os.path.join(a.origem, "resumo.json"), encoding="utf-8"))
    resumo.update({"origem": os.path.abspath(a.origem), "pares_alinhados": True,
                   "pares_refeitos": n_refeitos, "pares_ja_alinhados": n_ja_alinhados})
    json.dump(resumo, open(os.path.join(a.destino, "resumo.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"pares refeitos: {n_refeitos}; ja alinhados: {n_ja_alinhados}; "
          f"todos os {len(acentuada)} pares com o tamanho da acentuada")


if __name__ == "__main__":
    main()
