"""Monta a base de acentos sinteticos: uma versao acentuada por palavra elegivel do IAM.

Usa gerador.gerar(), com os filtros de qualidade: confianca do alinhamento
>= MIN_LOGP e visibilidade do sinal >= MIN_VISIBILIDADE (ate TENTATIVAS
sorteios). A semente de cada palavra e seed * 1_000_003 + posicao na lista de
elegiveis, entao a base e reprodutivel e cada amostra pode ser refeita sozinha.

Saida em --saida (gitignored):
  imagens/NNNNNN.png  -- amostras (cinza)
  split.txt           -- "imagens/NNNNNN.png,escritor,rotulo", formato dos
                         utils/splits_words/*.txt do DiffusionPen
  manifesto.jsonl     -- uma linha por amostra, com todos os parametros
  descartes.tsv       -- palavras descartadas e o motivo
  resumo.json         -- contagens, filtros, letras e commit

    python scripts/gerar_base_acentos.py --alinhador modelos/alinhador_iam.pt

--teto_por_palavra limita quantas ocorrencias de cada palavra viram acentuadas:
no maximo max(1, round(teto * ocorrencias)), sorteadas com a seed. Com 1.0
(padrao) todas viram. Motivo: com todas, o treino viu 2.907 "the" sem acento e
2.858 com acento, e o modelo passou a acentuar o "the" pedido sem acento
(ACHADOS.md, secao 10). O teto mantem a palavra frequente majoritariamente sem
acento e ainda da 1 exemplo acentuado a cada palavra rara.
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter
from multiprocessing import Pool

from PIL import Image

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
from acentos_sinteticos import gerador, iam  # noqa: E402

_ALIN = None
_CFG = None


def _iniciar(caminho_alinhador, cfg):
    global _ALIN, _CFG
    # 1 thread por processo: sem isso o OpenCV e o torch abrem um pool de
    # threads em cada worker e a maquina engasga (load 300 com 16 workers)
    import cv2
    import torch
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    from acentos_sinteticos import alinhamento
    _ALIN = alinhamento.Alinhador(caminho_alinhador, "cpu")
    _CFG = cfg


def _processar(item):
    k, caminho, escritor, texto = item
    semente = _CFG["seed"] * 1_000_003 + k
    try:
        g = iam.carregar_cinza(caminho)
    except OSError:
        return k, None, "imagem_ilegivel", semente
    am, motivo = gerador.gerar(g, texto, random.Random(semente), _ALIN,
                               min_logp=_CFG["min_logp"],
                               min_visibilidade=_CFG["min_visibilidade"],
                               tentativas=_CFG["tentativas"])
    if am is None:
        return k, None, motivo, semente
    nome = f"imagens/{k:06d}.png"
    Image.fromarray(am.imagem).save(os.path.join(_CFG["saida"], nome))
    return k, (nome, am.manifesto()), "ok", semente


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clone", default=os.path.join(RAIZ, "DiffusionPen"))
    ap.add_argument("--split", default="iam_train_val.txt",
                    help="split de utils/splits_words; o iam_test fica de fora da base")
    ap.add_argument("--alinhador", required=True)
    ap.add_argument("--saida", default=os.path.join(RAIZ, "iam_acentuado"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max", type=int, default=0, help="limite de palavras (0 = todas)")
    ap.add_argument("--min_logp", type=float, default=gerador.MIN_LOGP)
    ap.add_argument("--min_visibilidade", type=float, default=gerador.MIN_VISIBILIDADE)
    ap.add_argument("--tentativas", type=int, default=gerador.TENTATIVAS)
    ap.add_argument("--teto_por_palavra", type=float, default=1.0,
                    help="fracao maxima das ocorrencias de cada palavra que viram acentuadas")
    a = ap.parse_args()
    if not 0 < a.teto_por_palavra <= 1:
        sys.exit("ERRO: --teto_por_palavra tem de estar em (0, 1]")

    if os.path.exists(os.path.join(a.saida, "manifesto.jsonl")):
        sys.exit(f"ERRO: {a.saida} ja tem uma base; use outra --saida ou apague a pasta")
    os.makedirs(os.path.join(a.saida, "imagens"), exist_ok=True)
    palavras = [p for p in iam.listar(a.clone, a.split) if gerador.elegivel(p.texto)]
    if a.max:
        palavras = palavras[:a.max]
    itens = [(k, p.caminho, p.escritor, p.texto) for k, p in enumerate(palavras)]
    n_elegiveis = len(itens)
    print(f"palavras elegiveis em {a.split}: {n_elegiveis}", flush=True)
    if a.teto_por_palavra < 1:
        # k continua sendo a posicao na lista de elegiveis: a semente e o nome
        # da imagem de cada palavra nao mudam com o teto
        por_texto = {}
        for it in itens:
            por_texto.setdefault(it[3], []).append(it)
        rnd = random.Random(a.seed)
        escolhidos = set()
        for texto in sorted(por_texto):
            ocorr = por_texto[texto]
            n = max(1, round(a.teto_por_palavra * len(ocorr)))
            escolhidos.update(it[0] for it in rnd.sample(ocorr, min(n, len(ocorr))))
        itens = [it for it in itens if it[0] in escolhidos]
        print(f"teto {a.teto_por_palavra} por palavra: {len(itens)} ocorrencias "
              f"de {len(por_texto)} palavras distintas", flush=True)
    por_k = {it[0]: it for it in itens}

    cfg = {"seed": a.seed, "saida": a.saida, "min_logp": a.min_logp,
           "min_visibilidade": a.min_visibilidade, "tentativas": a.tentativas}
    motivos, letras, tentativas = Counter(), Counter(), Counter()
    t0 = time.time()
    with open(os.path.join(a.saida, "manifesto.jsonl"), "w", encoding="utf-8") as man, \
            open(os.path.join(a.saida, "split.txt"), "w", encoding="utf-8") as spl, \
            open(os.path.join(a.saida, "descartes.tsv"), "w", encoding="utf-8") as desc, \
            Pool(a.workers, initializer=_iniciar, initargs=(a.alinhador, cfg)) as pool:
        desc.write("n\tiam\ttexto\tmotivo\n")
        for n, (k, res, motivo, semente) in enumerate(pool.imap(_processar, itens, chunksize=16), 1):
            _, caminho, escritor, texto = por_k[k]
            rel = os.path.relpath(caminho, a.clone)
            motivos[motivo] += 1
            if res is None:
                desc.write(f"{k}\t{rel}\t{texto}\t{motivo}\n")
            else:
                nome, m = res
                letras[m["letra"]] += 1
                tentativas[m["params"]["tentativa"]] += 1
                man.write(json.dumps({"arquivo": nome, "iam": rel, "escritor": escritor,
                                      "semente": semente, **m}, ensure_ascii=False) + "\n")
                spl.write(f"{nome},{escritor},{m['rotulo']}\n")
            if n % 2000 == 0:
                print(f"  {n}/{len(itens)}  {time.time() - t0:.0f}s  {dict(motivos)}", flush=True)

    commit = subprocess.run(["git", "-C", RAIZ, "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    resumo = {"split": a.split, "elegiveis": n_elegiveis,
              "teto_por_palavra": a.teto_por_palavra, "selecionadas": len(itens),
              "geradas": motivos["ok"],
              "motivos": dict(motivos), "letras": dict(letras.most_common()),
              "tentativas": dict(sorted(tentativas.items())), "filtros": cfg,
              "alinhador": a.alinhador, "commit": commit,
              "segundos": round(time.time() - t0)}
    with open(os.path.join(a.saida, "resumo.json"), "w", encoding="utf-8") as f:
        json.dump(resumo, f, ensure_ascii=False, indent=2)
    print(json.dumps(resumo, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
