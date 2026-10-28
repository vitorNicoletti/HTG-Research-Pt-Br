"""
Etapa 2 da metrica final, diagnostico: onde o detector do colega erra a letra
do acento, contra o gabarito humano letra por letra do modelo de 40 epocas.

O gabarito por letra sai das planilhas: as letras que as duas pessoas
marcaram, quando marcaram as mesmas, e a decisao do desempate quando nao.

O detector e o scripts/medir_posicao_acento.py: marcas soltas acima do corpo
(scripts/medir_marcas.py), cada uma atribuida a letra cuja coluna, pelo
alinhador, contem o centro da marca. So imagens sem i e j.

Cada letra em que gabarito e detector discordam vai para uma causa:
  vizinha     o detector pos a marca na letra ao lado de uma letra do gabarito
              (a mesma marca, na coluna errada)
  perdida     a pessoa ve marca na letra e o detector nao acha marca perto
  inventada   o detector acha marca numa letra longe de qualquer letra do
              gabarito

Grava resultados/posicao_40ep.txt, resultados/posicao_40ep.csv (uma linha
por imagem) e folhas com as colunas do alinhador desenhadas.

    python avaliacao_diacriticos/experimentos/diagnosticar_posicao.py
"""
import collections
import csv
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

AQUI = os.path.dirname(os.path.abspath(__file__))
MODULO = os.path.dirname(AQUI)
sys.path.insert(0, MODULO)

from anotacao_dupla import ler_planilha, letras_marcadas  # noqa: E402
from sinais import Detector, sem_acento, sinais_pedidos, tem_pingo  # noqa: E402
from medir_marcas import marcas  # noqa: E402

RES = os.path.join(MODULO, "resultados")


def gabarito_por_letra():
    """{imagem: conjunto de letras (1..n) com marca em cima}."""
    b = ler_planilha(os.path.join(RES, "planilha_anotacao_bruno.csv"))
    v = ler_planilha(os.path.join(RES, "planilha_anotacao_vini.csv"))
    des = ler_planilha(os.path.join(RES, "anotacao_40ep_desempate.csv"))
    saida = {}
    for img in b:
        sb, sv = letras_marcadas(b[img]["marcas_em_cima"]), letras_marcadas(v[img]["marcas_em_cima"])
        if sb == sv:
            saida[img] = sb
        else:
            saida[img] = letras_marcadas(des[img]["decisao_marcas_em_cima"])
    return saida


def marcas_por_letra(d, g, letras):
    """(conjunto de letras 1..n com marca acima, caixas com a letra, analise)."""
    analise = d.fatias(g, letras)
    if analise is None:
        return None, [], None
    geo, fatias, x0 = analise
    caixas = [c for c in marcas(g)[2] if c[4] == "acima"]
    saida, com_letra = set(), []
    for x, y, w, h, _ in caixas:
        cx = x + w / 2 - x0
        k = next((i for i, (a, b) in enumerate(fatias) if a <= cx < b),
                 0 if cx < fatias[0][0] else len(fatias) - 1)
        saida.add(k + 1)
        com_letra.append((x, y, w, h, k + 1))
    return saida, com_letra, analise


def causas(gab, det):
    """Contagem das causas de discordancia numa imagem."""
    c = collections.Counter()
    so_gab, so_det = gab - det, det - gab
    usados = set()
    for k in sorted(so_gab):
        viz = [j for j in (k - 1, k + 1) if j in so_det and j not in usados]
        if viz:
            usados.add(viz[0]); c["vizinha"] += 1
        else:
            c["perdida"] += 1
    c["inventada"] += len(so_det - usados)
    return c


def main():
    chave = {r["imagem"]: r for r in csv.DictReader(open(os.path.join(RES, "anotacao_40ep_chave.csv"), encoding="utf-8"))}
    gab = gabarito_por_letra()
    d = Detector()
    linhas, total, grupos = [], collections.Counter(), collections.defaultdict(collections.Counter)
    exemplos = collections.defaultdict(list)
    for img, item in chave.items():
        palavra = item["palavra"]
        if tem_pingo(palavra):
            continue
        g = np.asarray(Image.open(os.path.join(MODULO, item["painel"])).convert("L"), dtype=np.float32)
        letras = sem_acento(palavra)
        det, caixas, analise = marcas_por_letra(d, g, letras)
        pedidas = {k + 1 for k, s in sinais_pedidos(palavra) if s != "cedilha"}
        grupo = "com acento em cima" if pedidas else "sem acento em cima pedido"
        if det is None:
            total["sem alinhamento"] += 1
            continue
        g_set = gab[img]
        c = causas(g_set, det)
        igual = g_set == det
        total["imagens"] += 1; total["iguais"] += igual
        grupos[grupo]["imagens"] += 1; grupos[grupo]["iguais"] += igual
        for k, n in c.items():
            if not n:
                continue
            total[k] += n; grupos[grupo][k] += n
            exemplos[k].append((img, palavra, sorted(g_set), sorted(det), item["painel"], caixas, analise))
        linhas.append({"imagem": img, "palavra": palavra, "grupo": grupo, "gabarito": " ".join(map(str, sorted(g_set))),
                       "detector": " ".join(map(str, sorted(det))), "vizinha": c["vizinha"],
                       "perdida": c["perdida"], "inventada": c["inventada"]})

    with open(os.path.join(RES, "posicao_40ep.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader(); w.writerows(linhas)
    out = ["Letra do acento: detector do colega contra o gabarito humano letra por letra.",
           "Modelo de 40 epocas, imagens sem i e j.", ""]
    for nome, c in [("todas", total)] + sorted(grupos.items()):
        out.append(f"{nome}: {c['imagens']} imagens, letras iguais em {c['iguais']} ({c['iguais'] / max(1, c['imagens']):.0%}); "
                   f"letras em desacordo: vizinha {c['vizinha']}, perdida {c['perdida']}, inventada {c['inventada']}")
    out.append(f"sem alinhamento: {total['sem alinhamento']}")
    texto = "\n".join(out) + "\n"
    open(os.path.join(RES, "posicao_40ep.txt"), "w", encoding="utf-8").write(texto)
    print(texto)

    # folhas: cada causa, com as colunas do alinhador e as marcas achadas
    fonte = ImageFont.truetype(os.path.join(os.path.dirname(__import__("matplotlib").__file__),
                                            "mpl-data", "fonts", "ttf", "DejaVuSans.ttf"), 14)
    for causa, itens in exemplos.items():
        E, W, H = 2, 256, 64
        folha = Image.new("RGB", (2 * (W * E + 10), 24 + (len(itens) + 1) // 2 * (H * E + 22)), "white")
        dr = ImageDraw.Draw(folha)
        dr.text((4, 4), f"{causa}: azul = colunas do alinhador, vermelho = marca achada, numero = letra atribuida",
                fill="black", font=fonte)
        for n, (img, palavra, gs, ds, painel, caixas, (geo, fatias, x0)) in enumerate(itens):
            p = Image.open(os.path.join(MODULO, painel)).convert("RGB")
            dp = ImageDraw.Draw(p)
            for a, b in fatias:
                dp.line([(a + x0, 0), (a + x0, 63)], fill=(120, 160, 255))
            for x, y, w, h, k in caixas:
                dp.rectangle([x, y, x + w, y + h], outline=(230, 0, 0))
            x, y = (n % 2) * (W * E + 10), 24 + (n // 2) * (H * E + 22)
            dr.text((x, y), f"{img} {palavra}  pessoas {gs}  detector {ds}", fill="black", font=fonte)
            folha.paste(p.resize((W * E, H * E), Image.NEAREST), (x, y + 18))
        folha.save(os.path.join(RES, f"posicao_40ep_{causa}.png"))


if __name__ == "__main__":
    main()
