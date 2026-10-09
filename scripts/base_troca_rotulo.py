"""Separa o TRACO do acento do ROTULO acentuado, a partir de uma base com pares.

    python scripts/base_troca_rotulo.py --origem iam_pt_sub --modo traco  --saida ing_A_traco
    python scripts/base_troca_rotulo.py --origem iam_pt_sub --modo rotulo --saida ing_B_rotulo

Numa base de acentos cada amostra acentuada muda duas coisas em relacao ao seu
par: a imagem ganha o sinal desenhado e o rotulo ganha o caractere acentuado.
Aqui so as amostras acentuadas mudam; o resto da base fica igual:

  traco   imagem COM o sinal desenhado, rotulo SEM acento ("provavel")
  rotulo  imagem SEM o sinal (a do par), rotulo COM acento ("provável")

As duas bases sao deliberadamente erradas: servem para medir qual das duas
metades estraga a letra, nao para ensinar acento. As imagens sao links para a
base de origem.
"""
import argparse
import json
import os

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origem", required=True)
    ap.add_argument("--modo", choices=("traco", "rotulo"), required=True)
    ap.add_argument("--saida", required=True)
    a = ap.parse_args()

    with open(os.path.join(a.origem, "manifesto.jsonl"), encoding="utf-8") as f:
        man = [json.loads(l) for l in f]
    tarefa = {}
    for x in man:
        if x["tipo"] in ("acentuada", "par"):
            tarefa.setdefault(x["tarefa"], {})[x["tipo"]] = x
    acentuada = {d["acentuada"]["arquivo"]: d for d in tarefa.values() if set(d) == {"acentuada", "par"}}

    os.makedirs(a.saida, exist_ok=True)
    for pasta in ("imagens", "originais"):
        link = os.path.join(a.saida, pasta)
        if not os.path.islink(link):
            os.symlink(os.path.realpath(os.path.join(a.origem, pasta)), link)

    linhas, trocadas = [], 0
    with open(os.path.join(a.origem, "split.txt"), encoding="utf-8") as f:
        for l in f:
            arq, esc, texto = l.rstrip("\n").split(",", 2)
            if arq in acentuada:
                d = acentuada[arq]
                if a.modo == "traco":
                    texto = d["par"]["rotulo"]            # imagem acentuada, rotulo do par
                else:
                    arq = d["par"]["arquivo"]             # imagem do par, rotulo acentuado
                trocadas += 1
            linhas.append(f"{arq},{esc},{texto}")
    with open(os.path.join(a.saida, "split.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(linhas) + "\n")
    with open(os.path.join(a.saida, "resumo.json"), "w", encoding="utf-8") as f:
        json.dump({"descricao": f"{a.origem} com as amostras acentuadas trocadas (modo {a.modo}): "
                                + ("imagem com o sinal e rotulo sem acento" if a.modo == "traco"
                                   else "imagem sem o sinal e rotulo com acento")
                                + ". Base de diagnostico, errada de proposito.",
                   "origem": a.origem, "modo": a.modo, "amostras": len(linhas), "trocadas": trocadas},
                  f, ensure_ascii=False, indent=1)
    print(f"{a.saida}: {len(linhas)} amostras, {trocadas} acentuadas trocadas (modo {a.modo})")


if __name__ == "__main__":
    main()
