"""
htr_comparar.py -- compara reconhecedores nos recortes REAIS do BRESSAY.

O eixo E2 depende de um leitor que consiga ler este corpus. O TrOCR nao
consegue: CER 0.826 em manuscrito humano, 1 leitura perfeita em 120. Este
script mede candidatos no mesmo conjunto, com os mesmos numeros, para a
comparacao ser direta.

Cinco medidas, as duas ultimas emprestadas de Aldarmaki e Ghannam (2023), que
isolam desempenho de diacritico do desempenho geral do reconhecedor:

  cer_ascii    CER com alvo e predicao dobrados para ASCII. Comparavel com o
               0.826 ja medido do TrOCR.
  cer_cru      CER sem dobrar, que e o que de fato importa.
  exato        fracao de leituras identicas ao alvo.
  producao     fracao das palavras acentuadas em que a leitura traz ALGUMA
               marca. Um leitor que nunca emite diacritico e inutil aqui,
               qualquer que seja o CER.
  falso_acento fracao das palavras ASCII em que a leitura inventa marca.

    python avaliacao_diacriticos/htr_comparar.py --backend easyocr \\
        --dir avaliacao_diacriticos/amostras/reais_test \\
        --csv-out avaliacao_diacriticos/resultados/htr_easyocr.csv
"""
import argparse
import collections
import csv
import json
import os
import sys
import unicodedata

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrica as M  # noqa: E402


def marcas_de(s):
    """Multiconjunto de marcas presentes no texto."""
    nomes = []
    for c in unicodedata.normalize("NFD", s):
        if c in M.MARCAS:
            nomes.append(M.MARCAS[c][0])
    return nomes


def ler_easyocr(caminhos, idiomas=("pt",)):
    import warnings
    warnings.filterwarnings("ignore")
    import easyocr
    leitor = easyocr.Reader(list(idiomas), gpu=False, verbose=False)
    saida = []
    for c in caminhos:
        try:
            saida.append(" ".join(leitor.readtext(c, detail=0,
                                                  paragraph=False)).strip())
        except Exception:
            saida.append("")
    return saida


def ler_paddleocr(caminhos, modelo="latin_PP-OCRv5_mobile_rec"):
    """PaddleOCR, so o reconhecedor. Unico candidato com alfabeto latino
    completo: 851 simbolos, cobrindo as cinco marcas do portugues.

    Usa TextRecognition e nao o pipeline PaddleOCR completo: a imagem ja e um
    recorte de palavra, entao detectar linha antes so acrescenta um modo de
    falha. O `lang="latin"` do pipeline nao existe nesta versao; o modelo e
    nomeado direto.
    """
    import os
    import warnings
    warnings.filterwarnings("ignore")
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    from paddleocr import TextRecognition
    motor = TextRecognition(model_name=modelo)
    saida = []
    for c in caminhos:
        try:
            r = list(motor.predict(c))
            saida.append((r[0].get("rec_text", "") if r else "").strip())
        except Exception:
            saida.append("")
    return saida


def ler_trocr(caminhos, dirbase):
    """Reaproveita o cache, se existir; senao roda o modelo."""
    cache = os.path.join(dirbase, "e2_leituras.json")
    if os.path.exists(cache):
        guardado = json.load(open(cache, encoding="utf-8"))
        nomes = [os.path.basename(c) for c in caminhos]
        if all(n in guardado for n in nomes):
            return [guardado[n]["lido"] for n in nomes]
    from reconhecedor import Reconhecedor
    return Reconhecedor(device="cpu").ler(caminhos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True,
                    choices=["easyocr", "trocr", "paddleocr"])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--csv-out", required=True)
    ap.add_argument("--limite", type=int, default=0)
    a = ap.parse_args()

    itens = [json.loads(l) for l in
             open(os.path.join(a.dir, "manifest.jsonl"), encoding="utf-8")]
    if a.limite:
        itens = itens[:a.limite]
    caminhos = [os.path.join(a.dir, d["arquivo"]) for d in itens]

    if a.backend == "easyocr":
        lidos = ler_easyocr(caminhos)
    elif a.backend == "paddleocr":
        lidos = ler_paddleocr(caminhos)
    else:
        lidos = ler_trocr(caminhos, a.dir)

    linhas = []
    for d, lido in zip(itens, lidos):
        alvo = d["palavra"]
        linhas.append({
            "arquivo": d["arquivo"], "alvo": alvo, "lido": lido,
            "acentuada": d["acentuada"],
            "cer_ascii": round(M.cer(M.dobra_ascii(alvo), M.dobra_ascii(lido)), 4),
            "cer_cru": round(M.cer(alvo.lower(), lido.lower()), 4),
            "exato": int(alvo.lower() == lido.lower()),
            "marcas_alvo": "|".join(marcas_de(alvo)),
            "marcas_lidas": "|".join(marcas_de(lido)),
        })

    with open(a.csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
        w.writeheader()
        w.writerows(linhas)

    acc = [l for l in linhas if l["acentuada"]]
    asc = [l for l in linhas if not l["acentuada"]]
    print(f"\nbackend: {a.backend}   conjunto: {a.dir}")
    print(f"{'grupo':14s}{'n':>5s}{'cer_ascii':>11s}{'cer_cru':>10s}{'exato':>8s}")
    for rot, v in (("acentuadas", acc), ("ASCII", asc)):
        if not v:
            continue
        print(f"{rot:14s}{len(v):5d}"
              f"{np.mean([l['cer_ascii'] for l in v]):11.3f}"
              f"{np.mean([l['cer_cru'] for l in v]):10.3f}"
              f"{100 * np.mean([l['exato'] for l in v]):7.1f}%")

    prod = np.mean([bool(l["marcas_lidas"]) for l in acc]) if acc else 0
    falso = np.mean([bool(l["marcas_lidas"]) for l in asc]) if asc else 0
    print(f"\nproducao de diacritico nas acentuadas : {100 * prod:.1f}%")
    print(f"falso acento nas ASCII                : {100 * falso:.1f}%")

    print(f"\n{'marca':14s}{'esperadas':>11s}{'produzidas':>12s}{'certas':>9s}")
    esp = collections.Counter()
    produz = collections.Counter()
    certas = collections.Counter()
    for l in acc:
        a_ = collections.Counter(l["marcas_alvo"].split("|") if l["marcas_alvo"] else [])
        b_ = collections.Counter(l["marcas_lidas"].split("|") if l["marcas_lidas"] else [])
        esp += a_
        produz += b_
        certas += (a_ & b_)
    for m in sorted(esp):
        print(f"{m:14s}{esp[m]:11d}{produz[m]:12d}{certas[m]:9d}")

    print(f"\ncsv: {a.csv_out}")


if __name__ == "__main__":
    main()
