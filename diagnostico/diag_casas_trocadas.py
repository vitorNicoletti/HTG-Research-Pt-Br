"""De qual parte do texto o modelo le o acento? Teste das casas trocadas.

O UNet consulta 40 vetores de texto (um por caractere, CANINE -> text_lin),
sem mascara: as ~32 posicoes de preenchimento depois do fim da palavra tambem
sao lidas. diag_texto_por_posicao.py mostrou que tirar o acento muda muito o
vetor da letra acentuada (tanto quanto trocar a letra) e tambem o das posicoes
de preenchimento. Aqui a pergunta e qual dos dois o modelo USA para desenhar.

Para cada palavra, com o mesmo escritor e o mesmo ruido, geram-se 6 versoes
montando a fileira de vetores com pedacos de "haverá" (A) e "havera" (E):

  1 A inteira            comportamento normal com acento
  2 E inteira            comportamento normal sem acento
  3 letras A + vazias E  o acento sobrevive sem o rastro nas casas vazias?
  4 letras E + vazias A  o rastro nas casas vazias basta para desenhar?
  5 E + so a casa da letra acentuada vinda de A   uma casa basta?
  6 A + so a casa da letra acentuada vinda de E   tirar uma casa apaga?

"letras" = [CLS], os caracteres e o [SEP]; "vazias" = tudo depois do [SEP].
Nada e treinado e nenhum arquivo do modelo muda: a saida do codificador de
texto e trocada so durante a geracao.

    python diagnostico/diag_casas_trocadas.py --ckpt model_iam_pt_peso5/models/ema_bloco_16ep.pt \\
        --saida diagnostico/resultados/casas_trocadas
"""
import argparse
import os
import random
import sys
import unicodedata

import numpy as np
import torch

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "scripts"))
os.chdir(RAIZ)

PALAVRAS = ["haverá", "provável", "módulo", "câmera", "pântano", "você", "fogão"]
VERSOES = ["1 com acento", "2 sem acento", "3 letras A + vazias E", "4 letras E + vazias A",
           "5 E + casa do acento", "6 A - casa do acento"]


def esqueleto(p):
    return "".join(c for c in unicodedata.normalize("NFD", p) if unicodedata.category(c) != "Mn")


def fonte(tam=13):
    """Fonte com os acentos do portugues. A fonte padrao do PIL nao os tem e
    desenha um quadrado no lugar -- justamente nas palavras que interessam."""
    from PIL import ImageFont
    candidatas = []
    try:
        import matplotlib
        candidatas.append(os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf"))
    except Exception:
        pass
    candidatas += ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/TTF/DejaVuSans.ttf"]
    for c in candidatas:
        if os.path.isfile(c):
            return ImageFont.truetype(c, tam)
    raise SystemExit("nenhuma fonte com acentos encontrada (DejaVuSans); as legendas sairiam com quadrados")


def pasta_paineis(saida, esq):
    return os.path.join(saida, "paineis", esq)


def montar_folha(saida, palavra, escritores, sementes):
    """Folha de uma palavra a partir dos paineis salvos: linhas = versoes."""
    from PIL import Image, ImageDraw
    esq = esqueleto(palavra)
    colunas = [(s, e) for s in sementes for e in escritores]
    # legendas com as proprias palavras, para a folha se ler sozinha
    legendas = [(f"1  texto: {palavra}", ""),
                (f"2  texto: {esq}", ""),
                (f"3  letras de {palavra}", f"   casas vazias de {esq}"),
                (f"4  letras de {esq}", f"   casas vazias de {palavra}"),
                (f"5  {esq}, com so a casa", f"   da letra acentuada de {palavra}"),
                (f"6  {palavra}, com so a casa", f"   da letra acentuada de {esq}")]
    LEG, CAB = 250, 22
    folha = Image.new("L", (LEG + len(colunas) * 258, len(VERSOES) * 66 + CAB), 255)
    d, f = ImageDraw.Draw(folha), fonte()
    d.text((4, 3), f"{palavra} / {esq}    colunas: {len(escritores)} escritores do iam_test "
                   f"x sementes {list(sementes)}; mesma coluna = mesmo escritor e mesmo ruido", fill=0, font=f)
    for v, (l1, l2) in enumerate(legendas):
        d.text((4, CAB + v * 66 + (16 if l2 else 24)), l1, fill=0, font=f)
        d.text((4, CAB + v * 66 + 32), l2, fill=0, font=f)
        for c, (s, e) in enumerate(colunas):
            folha.paste(Image.open(os.path.join(pasta_paineis(saida, esq), f"{v + 1}_{e}_{s}.png")),
                        (LEG + c * 258, CAB + v * 66))
    folha.save(os.path.join(saida, f"{esq}.png"))


class TextoFixo(torch.nn.Module):
    """No lugar do codificador de texto: devolve a fileira de vetores preparada."""

    def __init__(self):
        super().__init__()
        self.fixo = None

    def forward(self, **_):
        return type("Saida", (), {"last_hidden_state": self.fixo})()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--style", default="DiffusionPen/style_models/iam_style_diffusionpen.pth")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--escritores", type=int, default=4)
    ap.add_argument("--sementes", type=int, nargs="+", default=[42, 43])
    ap.add_argument("--palavras", nargs="+", default=PALAVRAS)
    ap.add_argument("--refazer_folhas", action="store_true",
                    help="nao gera nada: remonta as folhas a partir de <saida>/paineis e do paineis.tsv")
    a = ap.parse_args()

    if a.refazer_folhas:
        vistos = {}
        with open(os.path.join(a.saida, "paineis.tsv"), encoding="utf-8") as f:
            for l in list(f)[1:]:
                p, _, e, s = l.split("\t")[:4]
                esc, sem = vistos.setdefault(p, ([], []))
                if e not in esc:
                    esc.append(e)
                if int(s) not in sem:
                    sem.append(int(s))
        for p, (esc, sem) in vistos.items():
            montar_folha(a.saida, p, esc, sem)
            print("folha refeita:", os.path.join(a.saida, esqueleto(p) + ".png"))
        return

    from PIL import Image
    from comum.diffusionpen import EscritoresIAM, GeradorDiffusionPen
    from medir_marcas import marcas

    g = GeradorDiffusionPen(a.ckpt, a.style, "cpu")
    codificador = g.ema.module.text_encoder
    troca = TextoFixo()
    g.ema.module.text_encoder = troca

    teste = EscritoresIAM("iam_test.txt")
    # So escritores do iam_test com TODAS as imagens no disco: nesta maquina o
    # IAM esta extraido pela metade, e uma referencia ausente derrubaria a geracao.
    completos = [e for e in teste.escritores if all(os.path.isfile(c) for c, _ in teste.por_escritor[e])]
    if len(completos) < a.escritores:
        raise SystemExit(f"so {len(completos)} escritores do iam_test com todas as imagens")
    escritores = random.Random(0).sample(completos, a.escritores)
    print(f"escritores do iam_test: {escritores} (de {len(completos)} completos)", flush=True)
    n = len(escritores)

    def vetores(p):
        t = g.tokenizer([p], padding="max_length", truncation=True, return_tensors="pt",
                        max_length=g.texto_max_len)
        with torch.no_grad():
            return codificador(**t).last_hidden_state[0]

    os.makedirs(a.saida, exist_ok=True)
    linhas = ["palavra\tversao\tescritor\tsemente\tmarca_acima\tx_marcas"]
    conta = {}
    for palavra in a.palavras:
        esq = esqueleto(palavra)
        alvo = [i + 1 for i, (c, e) in enumerate(zip(palavra, esq)) if c != e]   # +1: [CLS]
        fim = len(palavra) + 2                                                    # primeira casa vazia
        A, E = vetores(palavra), vetores(esq)
        mist = [A, E, torch.cat([A[:fim], E[fim:]]), torch.cat([E[:fim], A[fim:]]), E.clone(), A.clone()]
        mist[4][alvo] = A[alvo]
        mist[5][alvo] = E[alvo]

        os.makedirs(pasta_paineis(a.saida, esq), exist_ok=True)
        for v, H in enumerate(mist):
            for s in a.sementes:
                refs = [teste.referencias(e, random.Random(s * 1000 + j)) for j, e in enumerate(escritores)]
                troca.fixo = H.unsqueeze(0).repeat(n, 1, 1)
                imgs = g.gerar([esq] * n, refs, s)          # o texto passado aqui e ignorado
                for e, img in zip(escritores, imgs):
                    cinza = (img.mean(0).numpy() * 255).astype(np.uint8)
                    acima, _, caixas, geo = marcas(cinza)
                    xs = []
                    if geo is not None:
                        x0, x1 = geo.caixa[0], geo.caixa[1]
                        xs = [round((x + w / 2 - x0) / max(1, x1 - x0), 2) for x, y, w, h, onde in caixas if onde == "acima"]
                    linhas.append(f"{palavra}\t{VERSOES[v]}\t{e}\t{s}\t{int(acima > 0)}\t{xs}")
                    conta.setdefault((palavra, v), []).append(int(acima > 0))
                    Image.fromarray(cinza).save(os.path.join(pasta_paineis(a.saida, esq), f"{v + 1}_{e}_{s}.png"))
            print(f"{palavra}  {VERSOES[v]:<24} marca acima em {sum(conta[(palavra, v)])}/{len(conta[(palavra, v)])}",
                  flush=True)

        montar_folha(a.saida, palavra, escritores, a.sementes)

    with open(os.path.join(a.saida, "paineis.tsv"), "w", encoding="utf-8") as f:
        f.write("\n".join(linhas) + "\n")
    print("\nmarca acima do corpo, fracao das imagens (todas as palavras)")
    for v, nome in enumerate(VERSOES):
        x = [m for (p, vv), ms in conta.items() if vv == v for m in ms]
        print(f"  {nome:<24} {np.mean(x):.0%}  ({sum(x)}/{len(x)})")
    print("FIM")


if __name__ == "__main__":
    main()
