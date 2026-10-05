"""Base portuguesa: palavras do TREINO do vocabulario, escritas pelo DiffusionPen
original do IAM sem acento e acentuadas pelo pipeline sintetico.

Para cada palavra acentuada do treino (ex.: "coração"):
  1. o DiffusionPen do IAM escreve o esqueleto ("coracao") no estilo de um
     escritor do iam_train_val;
  2. a imagem e recortada na tinta e ampliada 2x (resolucao para o desenho);
  3. o alinhador CTC le o esqueleto: abaixo de MIN_LOGP a geracao e
     considerada ilegivel e sai;
  4. gerador.acentuar_palavra desenha TODOS os sinais; se algum sair
     invisivel, a amostra sai.
Tres tipos de amostra no split.txt:
  acentuada  -- imagem com os sinais, rotulo "coração"
  par        -- a MESMA imagem antes dos sinais, rotulo "coracao", na mesma
                tela da acentuada (alinhada); o contraste mais direto: so o
                sinal muda
  sem_acento -- palavra do treino que nao tem acento ("casa"), gerada e
                filtrada do mesmo jeito: portugues sem marca nenhuma

Anti-vazamento: so palavras do split "treino" de vocabulario_pt/palavras.tsv
entram (conferido antes de gerar e de novo antes de gravar), e so
escritores do iam_train_val dao o estilo. Palavras amostradas
uniformemente, um numero fixo de imagens por palavra -- nunca pela
frequencia (com frequencia, "não"/"é" dominariam como "the" dominou).

    python scripts/gerar_base_pt.py --saida iam_pt
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter

import numpy as np
from PIL import Image

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
from acentos_sinteticos import alinhamento, gerador, geometria  # noqa: E402
from acentos_sinteticos.vocabulario import Vocabulario, esqueleto  # noqa: E402
from comum.diffusionpen import EscritoresIAM, GeradorDiffusionPen  # noqa: E402

AMPLIACAO = 2       # a palavra gerada tem ~40 px de altura; o desenho fica melhor em 2x
MARGEM = 3          # px em volta da tinta no recorte (antes da ampliacao)


def recortar(img):
    """Tensor (3,64,256) em 0..1 -> cinza float recortado na tinta e ampliado; None se vazio."""
    g = (img.mean(0).numpy() * 255).astype(np.float32)
    m = geometria.mascara_tinta(g)
    if m.sum() < 20:
        return None
    ys, xs = np.where(m)
    y0, y1 = max(0, ys.min() - MARGEM), min(g.shape[0], ys.max() + 1 + MARGEM)
    x0, x1 = max(0, xs.min() - MARGEM), min(g.shape[1], xs.max() + 1 + MARGEM)
    r = Image.fromarray(g[y0:y1, x0:x1].astype(np.uint8))
    r = r.resize((r.width * AMPLIACAO, r.height * AMPLIACAO), Image.BICUBIC)
    return np.asarray(r, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default=os.path.join(RAIZ, "iam_pt"))
    ap.add_argument("--ckpt", default=os.path.join(RAIZ, "DiffusionPen/diffusionpen_iam_model_path/models/ema_ckpt.pt"))
    ap.add_argument("--style", default=os.path.join(RAIZ, "DiffusionPen/style_models/iam_style_diffusionpen.pth"))
    ap.add_argument("--alinhador", default=os.path.join(RAIZ, "modelos/alinhador_iam.pt"))
    ap.add_argument("--imagens_por_palavra", type=int, default=2)
    ap.add_argument("--sem_acento", type=int, default=5000, help="palavras do treino sem acento")
    ap.add_argument("--max_palavras", type=int, default=0, help="limite de acentuadas (teste rapido)")
    ap.add_argument("--lote", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--min_logp", type=float, default=gerador.MIN_LOGP)
    a = ap.parse_args()

    if os.path.exists(os.path.join(a.saida, "split.txt")):
        sys.exit(f"ERRO: {a.saida} ja tem uma base; use outra --saida")
    os.makedirs(os.path.join(a.saida, "imagens"), exist_ok=True)

    voc = Vocabulario()
    rnd = random.Random(a.seed)
    acentuadas = sorted(p for p in voc.lista("treino", acentuada=True) if p.isalpha())
    sem = sorted(voc.lista("treino", acentuada=False))
    if a.max_palavras:
        acentuadas = rnd.sample(acentuadas, a.max_palavras)
    sem = rnd.sample(sem, min(a.sem_acento, len(sem)))
    voc.conferir(acentuadas + sem + [esqueleto(p) for p in acentuadas], "treino", "gerar_base_pt")

    escritores = EscritoresIAM("iam_train_val.txt")
    # tarefa: (id, texto que o DiffusionPen escreve, alvo acentuado ou None, escritor)
    tarefas = []
    for p in acentuadas:
        for _ in range(a.imagens_por_palavra):
            tarefas.append((len(tarefas), esqueleto(p), p, rnd.choice(escritores.escritores)))
    for p in sem:
        tarefas.append((len(tarefas), p, None, rnd.choice(escritores.escritores)))
    rnd.shuffle(tarefas)
    print(f"{len(acentuadas)} acentuadas x {a.imagens_por_palavra} + {len(sem)} sem acento "
          f"= {len(tarefas)} geracoes, {len(escritores.escritores)} escritores", flush=True)

    ger = GeradorDiffusionPen(a.ckpt, a.style, a.device)
    alin = alinhamento.Alinhador(a.alinhador, "cpu")
    motivos, tipos = Counter(), Counter()
    t0 = time.time()
    with open(os.path.join(a.saida, "split.txt"), "w", encoding="utf-8") as spl, \
            open(os.path.join(a.saida, "manifesto.jsonl"), "w", encoding="utf-8") as man, \
            open(os.path.join(a.saida, "descartes.tsv"), "w", encoding="utf-8") as desc:
        desc.write("tarefa\ttexto\talvo\tescritor\tmotivo\n")

        def grava(nome, img, escritor, rotulo, tipo, extra):
            voc.conferir([rotulo], "treino", "gravacao")
            Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(os.path.join(a.saida, nome))
            spl.write(f"{nome},{escritor},{rotulo}\n")
            man.write(json.dumps({"arquivo": nome, "escritor": escritor, "rotulo": rotulo,
                                  "tipo": tipo, **extra}, ensure_ascii=False) + "\n")
            tipos[tipo] += 1

        for b in range(0, len(tarefas), a.lote):
            lote = tarefas[b:b + a.lote]
            semente = a.seed * 1_000_003 + b
            refs = [escritores.referencias(e, random.Random(semente + k)) for k, (_, _, _, e) in enumerate(lote)]
            imgs = ger.gerar([t[1] for t in lote], refs, semente)
            for (tid, texto, alvo, esc), img in zip(lote, imgs):
                g = recortar(img)
                if g is None:
                    motivo = "vazia"
                elif alvo is None:
                    al = alin.alinhar(g, texto)
                    if al is None or al.logp_medio < a.min_logp:
                        motivo = "confianca"
                    else:
                        motivo = "ok"
                        grava(f"imagens/{tid:06d}.png", g, esc, texto, "sem_acento",
                              {"tarefa": tid, "semente_lote": semente,
                               "logp_alinhamento": round(al.logp_medio, 4)})
                else:
                    ac, motivo, info = gerador.acentuar_palavra(
                        g, texto, alvo, random.Random(a.seed * 1_000_003 + 7919 * tid), alin,
                        min_logp=a.min_logp)
                    if ac is not None:
                        extra = {"tarefa": tid, "semente_lote": semente, "alvo": alvo, **info}
                        # o par vai na MESMA tela da acentuada (alinhado): ver gerador.par_na_tela
                        par = gerador.par_na_tela(g, ac.shape, info["margens_esq_cima"])
                        grava(f"imagens/{tid:06d}_par.png", par, esc, texto, "par", extra)
                        grava(f"imagens/{tid:06d}.png", ac, esc, alvo, "acentuada", extra)
                motivos[motivo] += 1
                if motivo != "ok":
                    desc.write(f"{tid}\t{texto}\t{alvo or ''}\t{esc}\t{motivo}\n")
            feitas = min(b + a.lote, len(tarefas))
            if (b // a.lote) % 10 == 0:
                print(f"  {feitas}/{len(tarefas)}  {time.time() - t0:.0f}s  {dict(motivos)}", flush=True)

    commit = subprocess.run(["git", "-C", RAIZ, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    resumo = {"vocabulario": "vocabulario_pt/palavras.tsv", "split_vocabulario": "treino",
              "escritores": "iam_train_val.txt", "acentuadas": len(acentuadas),
              "imagens_por_palavra": a.imagens_por_palavra, "sem_acento": len(sem),
              "geracoes": len(tarefas), "motivos": dict(motivos), "amostras": dict(tipos),
              "min_logp": a.min_logp, "min_visibilidade": gerador.MIN_VISIBILIDADE,
              "ampliacao": AMPLIACAO, "seed": a.seed, "ckpt": a.ckpt, "commit": commit,
              "segundos": round(time.time() - t0)}
    with open(os.path.join(a.saida, "resumo.json"), "w", encoding="utf-8") as f:
        json.dump(resumo, f, ensure_ascii=False, indent=2)
    print(json.dumps(resumo, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
