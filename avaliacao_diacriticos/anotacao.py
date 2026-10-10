"""
Anotacao humana de imagens geradas por um modelo que desenha acento.

    gerar   gera cerca de 200 imagens na CPU com o checkpoint dado
    folha   monta a planilha em branco e as folhas com as imagens numeradas
    kappa   compara a planilha preenchida com os detectores

As palavras vem do split de validacao do vocabulario, nunca do teste, e nenhuma
esta na base de treino do modelo. Os escritores sao do iam_test e so entram os
que tem todas as imagens no disco.

Cada palavra acentuada e gerada tambem sem o acento, com o mesmo escritor e o
mesmo ruido. Entram ainda palavras que nunca tiveram acento. Nas imagens sem
acento a resposta certa para "tem acento?" e nao, e e ali que aparece o
vazamento.

    python anotacao.py gerar
    python anotacao.py folha
    python anotacao.py kappa
"""
import argparse
import csv
import json
import os
import random
import sys
import unicodedata

import numpy as np
from PIL import Image, ImageDraw

MODULO = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(MODULO)
sys.path.insert(0, MODULO)
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "scripts"))

PASTA = os.path.join(MODULO, "amostras", "anotacao_peso5")
CHECKPOINT = os.path.join(RAIZ, "model_iam_pt_peso5", "models", "ema_bloco_16ep.pt")
ESTILO = os.path.join(RAIZ, "DiffusionPen", "style_models", "iam_style_diffusionpen.pth")
BASE_DE_TREINO = os.path.join(RAIZ, "iam_pt_alinhado")

NOME_DA_MARCA = {"́": "agudo", "̀": "grave", "̂": "circunflexo",
                 "̃": "til", "̧": "cedilha"}
PALAVRAS_POR_MARCA = 10
PALAVRAS_SEM_ACENTO = 20
ESCRITORES_POR_PALAVRA = 2
SEMENTE = 42


def marcas_da(palavra):
    """Nome de cada marca da palavra, na ordem."""
    return [NOME_DA_MARCA[s] for s in unicodedata.normalize("NFD", palavra)
            if s in NOME_DA_MARCA]


# ------------------------------------------------------------------- gerar

def escolher_palavras():
    """Palavras de validacao com um sinal so, tantas por marca, mais as sem
    acento. Mesmo filtro do protocolo do colega: 3 a 10 letras, sem i nem j."""
    from acentos_sinteticos.vocabulario import Vocabulario, esqueleto, grupo
    from avaliar_pt import conferir_bases

    with open(os.path.join(BASE_DE_TREINO, "split.txt"), encoding="utf-8") as f:
        grupos_do_treino = {grupo(l.rstrip("\n").split(",", 2)[2])
                            for l in f if l.count(",") >= 2}

    def serve(palavra):
        return (3 <= len(palavra) <= 10
                and not set("ij") & set(esqueleto(palavra))
                and grupo(palavra) not in grupos_do_treino)

    vocabulario = Vocabulario()
    sorteio = random.Random(0)
    acentuadas = []
    candidatas = sorted(p for p in vocabulario.lista("val", True) if serve(p))
    for marca in ["agudo", "til", "circunflexo", "cedilha", "grave"]:
        da_marca = [p for p in candidatas if marcas_da(p) == [marca]]
        acentuadas += sorteio.sample(da_marca, min(PALAVRAS_POR_MARCA, len(da_marca)))
    sem_acento = sorteio.sample(sorted(p for p in vocabulario.lista("val", False) if serve(p)),
                                PALAVRAS_SEM_ACENTO)

    itens = [(p, "acentuada", esqueleto(p)) for p in acentuadas] + \
            [(esqueleto(p), "esqueleto", esqueleto(p)) for p in acentuadas] + \
            [(p, "sem_acento", p) for p in sem_acento]
    # as duas conferencias do protocolo do colega; param com erro se falharem
    vocabulario.conferir([texto for texto, _, _ in itens], "val", "anotacao")
    conferir_bases(itens, [BASE_DE_TREINO])
    return acentuadas, sem_acento


def escritores_completos(escritores):
    """Escritores cujas imagens estao todas no disco. O IAM local esta pela metade."""
    return [e for e in escritores.escritores
            if all(os.path.exists(caminho) for caminho, _ in escritores.por_escritor[e])]


def gerar(a):
    from acentos_sinteticos.vocabulario import esqueleto
    from comum.diffusionpen import EscritoresIAM, GeradorDiffusionPen

    acentuadas, sem_acento = escolher_palavras()
    iam_teste = EscritoresIAM("iam_test.txt")
    iam_treino = EscritoresIAM("iam_train_val.txt")
    completos = [e for e in escritores_completos(iam_teste)
                 if e not in set(iam_treino.escritores)]
    print(f"{len(acentuadas)} acentuadas, {len(sem_acento)} sem acento, "
          f"{len(completos)} escritores completos do iam_test")

    os.makedirs(a.pasta, exist_ok=True)
    gerador = GeradorDiffusionPen(a.checkpoint, ESTILO, "cpu")   # geracao so na CPU
    manifesto = []
    grupos = [(p, [(p, "acentuada"), (esqueleto(p), "esqueleto")]) for p in acentuadas] + \
             [(p, [(p, "sem_acento")]) for p in sem_acento]
    for numero, (alvo, textos) in enumerate(grupos):
        escritores = random.Random(numero).sample(completos, ESCRITORES_POR_PALAVRA)
        for texto, tipo in textos:
            # mesmos escritores, mesmas referencias e mesma semente para a
            # acentuada e o esqueleto: as duas imagens diferem so no texto
            referencias = [iam_teste.referencias(e, random.Random(SEMENTE * 1000 + j))
                           for j, e in enumerate(escritores)]
            imagens = gerador.gerar([texto] * len(escritores), referencias, SEMENTE)
            for escritor, imagem in zip(escritores, imagens):
                cinza = (imagem.mean(0).numpy() * 255).astype(np.uint8)
                arquivo = f"{numero:03d}_{tipo}_{escritor}.png"
                Image.fromarray(cinza).save(os.path.join(a.pasta, arquivo))
                manifesto.append({"arquivo": arquivo, "texto": texto, "tipo": tipo,
                                  "alvo": alvo, "escritor": escritor, "semente": SEMENTE,
                                  "desvio_padrao": round(float(cinza.std()), 2)})
        print(f"{numero + 1}/{len(grupos)} {alvo}", flush=True)
        with open(os.path.join(a.pasta, "manifesto.jsonl"), "w", encoding="utf-8") as f:
            for item in manifesto:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"{len(manifesto)} imagens em {a.pasta}")


# ------------------------------------------------------------------- folha

IMAGENS_POR_FOLHA = 40
COLUNAS_DA_FOLHA = 4
AMPLIACAO = 2
PLANILHA = os.path.join(MODULO, "resultados", "anotacao_peso5.csv")
GABARITO = os.path.join(MODULO, "resultados", "anotacao_peso5_gabarito.csv")


def ler_manifesto(pasta):
    with open(os.path.join(pasta, "manifesto.jsonl"), encoding="utf-8") as f:
        return [json.loads(linha) for linha in f]


def folha(a):
    """Planilha em branco, folhas de imagens numeradas e o gabarito.

    Quem anota ve a imagem e as letras da palavra sem acento, numeradas. Nao ve
    se o texto pedido tinha acento. A pergunta "na letra certa?" vira "em qual
    letra?", e a comparacao com a letra pedida e feita depois, no kappa.
    """
    from PIL import ImageFont
    itens = ler_manifesto(a.pasta)
    random.Random(0).shuffle(itens)   # mistura os tipos para nao virem em blocos
    fonte = ImageFont.load_default(size=16)

    with open(GABARITO, "w", newline="", encoding="utf-8") as f:
        gabarito = csv.writer(f)
        gabarito.writerow(["id", "arquivo", "texto", "tipo", "alvo", "escritor"])
        for numero, item in enumerate(itens, 1):
            gabarito.writerow([numero, item["arquivo"], item["texto"], item["tipo"],
                               item["alvo"], item["escritor"]])

    with open(PLANILHA, "w", newline="", encoding="utf-8") as f:
        planilha = csv.writer(f)
        planilha.writerow(["id", "letras", "tem_acento", "em_qual_letra", "legivel", "observacao"])
        for numero, item in enumerate(itens, 1):
            letras = unicodedata.normalize("NFD", item["alvo"])
            letras = "".join(s for s in letras if unicodedata.category(s) != "Mn")
            planilha.writerow([numero, letras, "", "", "", ""])

    largura, altura = 256 * AMPLIACAO, 64 * AMPLIACAO
    celula_x, celula_y = largura + 16, altura + 48
    for inicio in range(0, len(itens), IMAGENS_POR_FOLHA):
        pagina = itens[inicio:inicio + IMAGENS_POR_FOLHA]
        linhas = (len(pagina) + COLUNAS_DA_FOLHA - 1) // COLUNAS_DA_FOLHA
        tela = Image.new("L", (COLUNAS_DA_FOLHA * celula_x, linhas * celula_y), 255)
        desenho = ImageDraw.Draw(tela)
        for k, item in enumerate(pagina):
            x = (k % COLUNAS_DA_FOLHA) * celula_x
            y = (k // COLUNAS_DA_FOLHA) * celula_y
            letras = unicodedata.normalize("NFD", item["alvo"])
            letras = [s for s in letras if unicodedata.category(s) != "Mn"]
            numeradas = "  ".join(f"{letra}{n}" for n, letra in enumerate(letras, 1))
            desenho.text((x + 4, y + 4), f"{inicio + k + 1:03d}   {numeradas}", fill=0, font=fonte)
            imagem = Image.open(os.path.join(a.pasta, item["arquivo"]))
            tela.paste(imagem.resize((largura, altura), Image.LANCZOS), (x, y + 28))
        nome = os.path.join(MODULO, "figuras", f"anotacao_peso5_folha{inicio // IMAGENS_POR_FOLHA + 1}.png")
        tela.save(nome)
        print("gravado", nome)
    print("planilha em branco", PLANILHA)


# ------------------------------------------------------------------- kappa

LEITOR = os.path.join(RAIZ, "modelos", "leitor_iam_transformer.pt")
LIMITES_DE_CER = [0.0, 0.1, 0.2, 0.34, 0.5]


def abrir(pasta, arquivo):
    return np.asarray(Image.open(os.path.join(pasta, arquivo)).convert("L"), dtype=np.float32)


def distancia_de_edicao(a, b):
    anterior = list(range(len(b) + 1))
    for i, letra_a in enumerate(a, 1):
        atual = [i]
        for k, letra_b in enumerate(b, 1):
            atual.append(min(anterior[k] + 1, atual[k - 1] + 1,
                             anterior[k - 1] + (letra_a != letra_b)))
        anterior = atual
    return anterior[-1]


def vereditos(pasta):
    """O que cada metodo diz de cada imagem: {arquivo: dict}.

    colega   scripts/medir_marcas.py, com a letra de cada marca dada por
             scripts/medir_posicao_acento.py. Vale para toda imagem.
    par      marcas_do_par.py, a marca que a acentuada tem e o esqueleto nao.
    e1       e1_alinhador.py com o limiar da marca. Fica como tentativa.
    cer      erro por caractere do leitor, contra a palavra sem acento.

    "par" e "e1" precisam do esqueleto e so existem nas imagens pedidas com
    acento. "certa" quer dizer marca do lado certo em toda letra que pede.
    """
    import e1
    import e1_alinhador
    import marcas_do_par
    from acentos_sinteticos import alinhamento
    from avaliar_pt import recorte_tinta
    from medir_posicao_acento import letras_das_marcas

    with open(os.path.join(MODULO, "resultados", "limiares_e1_alinhador.json"), encoding="utf-8") as f:
        limiares = json.load(f)["alinhador"]
    alinhador = e1_alinhador.carregar_alinhador()
    leitor = alinhamento.Alinhador(LEITOR, "cpu")
    itens = ler_manifesto(pasta)
    esqueletos = {(x["alvo"], x["escritor"]): x for x in itens if x["tipo"] == "esqueleto"}

    saida = {}
    for item in itens:
        cinza = abrir(pasta, item["arquivo"])
        letras = e1.sem_acento(item["texto"])
        pedidos = [(posicao, "acima" if acima else "abaixo")
                   for posicao, acima in e1.diacriticos(item["texto"])]

        recorte = recorte_tinta(cinza)
        lido = leitor.ler(recorte) if recorte is not None else ""
        veredito = {"cer": distancia_de_edicao(lido, letras) / len(letras)}

        achadas, _ = letras_das_marcas(cinza, letras, alinhador)
        veredito["colega_tem"] = int(len(achadas) > 0)
        veredito["colega_certa"] = int(bool(pedidos) and all(p in achadas for p in pedidos))

        if pedidos:
            cinza_par = abrir(pasta, esqueletos[(item["alvo"], item["escritor"])]["arquivo"])
            novas = marcas_do_par.marcas_novas(cinza, cinza_par, item["texto"], alinhador)
            veredito["par_tem"] = int(len(novas) > 0)
            veredito["par_certa"] = int(all(p in novas for p in pedidos))

            acentuada, par = 1.0 - cinza / 255.0, 1.0 - cinza_par / 255.0
            colunas = e1_alinhador.colunas_pelo_alinhador(par, letras, alinhador)
            tem = certa = False
            for (posicao, lado), marca in zip(pedidos, marcas_da(item["texto"])):
                # a mesma marca em todas as letras, para ver onde a medida e maior
                simbolo = "\u0301" if lado == "acima" else "\u0327"
                em_todas = e1.e1(acentuada, par, "".join(l + simbolo for l in letras), colunas)
                limiar = limiares[marca]["p95"]
                tem = tem or max(em_todas) > limiar
                certa = certa or (em_todas[posicao] > limiar
                                  and int(np.argmax(em_todas)) == posicao)
            veredito["e1_tem"], veredito["e1_certa"] = int(tem), int(certa)
        saida[item["arquivo"]] = veredito
    return saida


def cohen(a, b):
    """Kappa de Cohen entre duas listas de 0 e 1. nan se nao da para calcular."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0:
        return float("nan")
    concordam = np.mean(a == b)
    ao_acaso = np.mean(a) * np.mean(b) + (1 - np.mean(a)) * (1 - np.mean(b))
    if ao_acaso == 1:
        return float("nan")
    return (concordam - ao_acaso) / (1 - ao_acaso)


def comparar(titulo, linhas, humano, metodos):
    """Uma linha por metodo: kappa, acertos e alarmes falsos contra a pessoa."""
    h = [l[humano] for l in linhas]
    print(f"{titulo} (n={len(h)}, a pessoa diz sim em {sum(h)})")
    for nome, campo in metodos:
        m = [l[campo] for l in linhas]
        acertos = sum(1 for x, y in zip(h, m) if x and y)
        alarmes = sum(1 for x, y in zip(h, m) if not x and y)
        print(f"  {nome:8s} kappa {cohen(h, m):+.2f}   diz sim em {acertos} das {sum(h)} que a pessoa"
              f" viu e em {alarmes} das {len(h) - sum(h)} que nao viu")


def kappa(a):
    """Compara a planilha preenchida com os metodos automaticos."""
    import e1
    automaticos = vereditos(a.pasta)
    with open(GABARITO, encoding="utf-8") as f:
        gabarito = {linha["id"]: linha for linha in csv.DictReader(f)}
    with open(PLANILHA, encoding="utf-8") as f:
        respostas = [r for r in csv.DictReader(f) if r["tem_acento"].strip() in ("0", "1")]
    if not respostas:
        sys.exit("a planilha ainda nao tem respostas")

    linhas = []
    for resposta in respostas:
        g = gabarito[resposta["id"]]
        pedidas = [posicao + 1 for posicao, _ in e1.diacriticos(g["texto"])]
        apontadas = [int(n) for n in resposta["em_qual_letra"].replace(",", " ").split()]
        linhas.append({"tipo": g["tipo"], "marcas": marcas_da(g["texto"]),
                       "humano_tem": int(resposta["tem_acento"]),
                       "humano_certa": int(bool(pedidas) and sorted(apontadas) == pedidas),
                       "humano_legivel": resposta["legivel"].strip(),
                       **automaticos[g["arquivo"]]})
    com_acento = [l for l in linhas if l["tipo"] == "acentuada"]
    sem_acento = [l for l in linhas if l["tipo"] != "acentuada"]
    tres = [("colega", "colega_{}"), ("par", "par_{}"), ("e1", "e1_{}")]

    print(f"{len(linhas)} imagens anotadas\n")
    print("TEM ACENTO")
    comparar("pedidas com acento", com_acento, "humano_tem",
             [(n, c.format("tem")) for n, c in tres])
    comparar("pedidas sem acento", sem_acento, "humano_tem", [("colega", "colega_tem")])

    print("\nNA LETRA CERTA, so nas pedidas com acento")
    certa = [(n, c.format("certa")) for n, c in tres]
    comparar("todos os sinais", com_acento, "humano_certa", certa)
    for marca in ["agudo", "til", "circunflexo", "cedilha", "grave"]:
        da_marca = [l for l in com_acento if l["marcas"] == [marca]]
        if da_marca:
            comparar(marca, da_marca, "humano_certa", certa)

    print("\nLEGIVEL, contra o CER do leitor em varios limites")
    for nome, grupo in [("todas", linhas), ("pedidas com acento", com_acento),
                        ("pedidas sem acento", sem_acento)]:
        grupo = [l for l in grupo if l["humano_legivel"] in ("0", "1")]
        h = [int(l["humano_legivel"]) for l in grupo]
        print(f"{nome} (n={len(h)}, a pessoa diz legivel em {sum(h)})")
        for limite in LIMITES_DE_CER:
            m = [int(l["cer"] <= limite) for l in grupo]
            print(f"  CER ate {limite:.2f}  kappa {cohen(h, m):+.2f}   o leitor aceita {sum(m)},"
                  f" das quais {sum(1 for x, y in zip(h, m) if x and y)} a pessoa acha legiveis;"
                  f" concordam em {100 * np.mean(np.array(h) == np.array(m)):.0f}%")
        legiveis = [l["cer"] for l in grupo if l["humano_legivel"] == "1"]
        ilegiveis = [l["cer"] for l in grupo if l["humano_legivel"] == "0"]
        if legiveis and ilegiveis:
            print(f"  CER medio  {np.mean(legiveis):.2f} nas legiveis, {np.mean(ilegiveis):.2f} nas ilegiveis")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("comando", choices=["gerar", "folha", "kappa"])
    ap.add_argument("--pasta", default=PASTA)
    ap.add_argument("--checkpoint", default=CHECKPOINT)
    a = ap.parse_args()
    {"gerar": gerar, "folha": folha, "kappa": kappa}[a.comando](a)


if __name__ == "__main__":
    main()
