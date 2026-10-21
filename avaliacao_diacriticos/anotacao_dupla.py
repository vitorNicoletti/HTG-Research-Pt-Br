"""
Anotacao humana feita por duas pessoas, sobre imagens do avaliar_pt.py.

    gerar_i_j     gera na CPU palavras com i e j, que a avaliacao nao tem
    pacote        sorteia as imagens e monta a pasta que vai para quem anota
    concordancia  le as duas planilhas preenchidas e compara; com o desempate
                  preenchido, monta o gabarito final e compara com ele

Quem anota nao roda nada. Recebe as folhas com as imagens numeradas, as
imagens soltas, a planilha CSV e as instrucoes em texto. A folha mostra a
palavra sem acento, entao nada diz o que foi pedido, o modelo ou o que o
detector achou. A chave que liga cada imagem a original fica em resultados/
e nao vai no pacote.

    python anotacao_dupla.py gerar_i_j
    python anotacao_dupla.py pacote
    python anotacao_dupla.py detector
    python anotacao_dupla.py concordancia --planilhas a.csv b.csv

O detector grava as respostas dos detectores em resultados/. Com esse arquivo,
a concordancia so usa a biblioteca padrao do Python e roda em qualquer
computador, sem as imagens e sem o nix.
"""
import argparse
import csv
import json
import os
import random
import shutil
import sys
import unicodedata

MODULO = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(MODULO)

NOME = "anotacao_40ep"
AVALIACAO = os.path.join(MODULO, "amostras", "zona_vogais_40ep", "avaliacao_zona_vogais_40ep_val")
COM_I_J = os.path.join(MODULO, "amostras", NOME, "com_i_j")
PACOTE = os.path.join(MODULO, "amostras", NOME, "pacote")
CHAVE = os.path.join(MODULO, "resultados", f"{NOME}_chave.csv")

# palavras que nao entram em folha nem figura (a acentuada e o esqueleto dela)
EXCLUIDAS = {"rap\u00e9"}

# imagens por palavra acentuada da avaliacao, pelos sinais da palavra
POR_PALAVRA = {("cedilha",): 5, ("agudo",): 4, ("til",): 4,
               ("cedilha", "til"): 4, ("circunflexo",): 8}
ESQUELETOS = 30
SEM_ACENTO = 15
POR_TEXTO_COM_I_J = 2

NOME_DA_MARCA = {"\u0301": "agudo", "\u0300": "grave", "\u0302": "circunflexo",
                 "\u0303": "til", "\u0327": "cedilha"}

PERGUNTAS = ["marcas_em_cima", "cedilha", "legivel"]
COLUNAS = ["imagem", "letras"] + PERGUNTAS + ["observacao"]

IMAGENS_POR_FOLHA = 40
COLUNAS_DA_FOLHA = 4


def sem_acento(palavra):
    return "".join(s for s in unicodedata.normalize("NFD", palavra)
                   if unicodedata.category(s) != "Mn")


def sinais_pedidos(palavra):
    """[(posicao da letra, nome do sinal)], com a posicao contada de zero."""
    saida, posicao = [], -1
    for simbolo in unicodedata.normalize("NFD", palavra):
        if simbolo in NOME_DA_MARCA:
            saida.append((posicao, NOME_DA_MARCA[simbolo]))
        elif unicodedata.category(simbolo) != "Mn":
            posicao += 1
    return saida


def letras_numeradas(palavra):
    """As letras da palavra SEM acento, numeradas: 'c1 a2 c3 a4'."""
    return " ".join(f"{letra}{n}" for n, letra in enumerate(sem_acento(palavra), 1))


# ------------------------------------------------------------------ pacote

def ler_paineis(avaliacao):
    """Linhas do paineis.tsv, com o caminho de cada imagem.

    O numero no nome do arquivo e a posicao do par (palavra, tipo) na ordem em
    que aparece pela primeira vez no paineis.tsv. QUOTE_NONE porque uma das
    leituras do leitor tem aspas, e o csv padrao engole linhas sem avisar.
    """
    with open(os.path.join(avaliacao, "paineis.tsv"), encoding="utf-8") as f:
        linhas = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    ordem = []
    for linha in linhas:
        if (linha["palavra"], linha["tipo"]) not in ordem:
            ordem.append((linha["palavra"], linha["tipo"]))
    for linha in linhas:
        numero = ordem.index((linha["palavra"], linha["tipo"]))
        linha["caminho"] = os.path.join(
            avaliacao, "paineis", linha["modelo"],
            f"{numero:03d}_{linha['escritor']}_{linha['semente']}.png")
    return linhas


def sortear(linhas, quantas, sorteio):
    """quantas: {(palavra, tipo): imagens}. Escritores diferentes na mesma palavra."""
    amostra = []
    for texto in sorted(quantas):
        opcoes = [l for l in linhas if (l["palavra"], l["tipo"]) == texto]
        sorteio.shuffle(opcoes)
        escritores = set()
        for linha in opcoes:
            if linha["escritor"] not in escritores and len(escritores) < quantas[texto]:
                amostra.append(linha)
                escritores.add(linha["escritor"])
    return amostra


def repartir(textos, total, sorteio):
    """Reparte total imagens entre os textos, uma a mais para os primeiros sorteados."""
    textos = sorted(textos)
    sorteio.shuffle(textos)
    return {t: total // len(textos) + (k < total % len(textos)) for k, t in enumerate(textos)}


def desenho_da_amostra(sorteio):
    """As linhas sorteadas da avaliacao e das palavras com i e j."""
    proibidas = {sem_acento(p) for p in EXCLUIDAS}
    linhas = [l for l in ler_paineis(AVALIACAO) if sem_acento(l["palavra"]) not in proibidas]
    textos = {(l["palavra"], l["tipo"]) for l in linhas}
    quantas = {t: POR_PALAVRA[tuple(sorted({s for _, s in sinais_pedidos(t[0])}))]
               for t in textos if t[1] == "acentuada"}
    quantas.update(repartir([t for t in textos if t[1] == "esqueleto"], ESQUELETOS, sorteio))
    quantas.update(repartir([t for t in textos if t[1] == "sem_acento"], SEM_ACENTO, sorteio))
    amostra = sortear(linhas, quantas, sorteio)

    linhas = ler_paineis(COM_I_J)
    textos = {(l["palavra"], l["tipo"]) for l in linhas}
    amostra += sortear(linhas, {t: POR_TEXTO_COM_I_J for t in textos}, sorteio)
    return amostra


INSTRUCOES = """Anotação de acentos em palavras escritas por um programa

Cada imagem é uma palavra escrita à mão por um programa. Queremos saber o que
ele desenhou. São {n} imagens, em {folhas} folhas. Leva cerca de uma hora.

As folhas mostram, em cima de cada imagem, o número dela e as letras da
palavra sem acento, numeradas. A pasta imagens tem as mesmas imagens soltas,
para ampliar quando precisar.

Preencha a planilha planilha_anotacao.csv, uma linha por imagem.

marcas_em_cima
  Os números das letras que têm acento ou qualquer outra marca solta em cima,
  separados por espaço, por exemplo 2 6. Se não houver, escreva nenhuma. Se
  não der para decidir, escreva não sei. O formato da marca não importa.
  Se a marca ficar entre duas letras, use a letra mais perto do meio dela.

cedilha
  Os números das letras com cedilha embaixo, presa ou solta do c. Se não
  houver, escreva nenhuma. Se não der para decidir, escreva não sei.

legivel
  sim se dá para ler a palavra, ignorando os acentos. não se não dá.
  não sei se não der para decidir.

observacao
  Livre.

O que não conta como marca em cima
  O pingo do i e do j. Anote o i ou o j só se houver uma marca além do pingo,
  ou uma marca no lugar dele que não pareça um pingo.
  O traço do t.
  As hastes de letras altas, como l, d, f, h, t.

Responda pelo que você vê, e não pelo que a palavra deveria ter. Anote
sozinho e não comente as imagens com a outra pessoa que também está
anotando.

Ao terminar, salve a planilha em CSV com o seu nome no arquivo, por exemplo
planilha_anotacao_maria.csv. No Excel, use Salvar como e escolha o tipo
CSV UTF-8 (delimitado por vírgulas). No Google Planilhas, use Arquivo,
Fazer download, Valores separados por vírgula.
"""


def folhas(amostra_com_nomes):
    """Folhas PNG com as imagens numeradas e as letras sem acento em cima."""
    from PIL import Image, ImageDraw, ImageFont
    fonte = ImageFont.load_default(size=18)
    largura, altura = 512, 128
    celula_x, celula_y = largura + 24, altura + 40
    caminhos = []
    for inicio in range(0, len(amostra_com_nomes), IMAGENS_POR_FOLHA):
        pagina = amostra_com_nomes[inicio:inicio + IMAGENS_POR_FOLHA]
        linhas = (len(pagina) + COLUNAS_DA_FOLHA - 1) // COLUNAS_DA_FOLHA
        tela = Image.new("L", (COLUNAS_DA_FOLHA * celula_x, linhas * celula_y), 255)
        desenho = ImageDraw.Draw(tela)
        for k, (imagem, linha) in enumerate(pagina):
            x = (k % COLUNAS_DA_FOLHA) * celula_x
            y = (k // COLUNAS_DA_FOLHA) * celula_y
            desenho.text((x + 4, y + 6), f"{imagem[:-4]}    {letras_numeradas(linha['palavra'])}",
                         fill=0, font=fonte)
            painel = Image.open(linha["caminho"]).convert("L").resize((largura, altura), Image.LANCZOS)
            tela.paste(painel, (x, y + 34))
        caminho = os.path.join(PACOTE, f"folha_{inicio // IMAGENS_POR_FOLHA + 1}.png")
        tela.save(caminho)
        caminhos.append(caminho)
    return caminhos


def pacote(a):
    amostra = desenho_da_amostra(random.Random(0))
    # embaralha ate nao haver a mesma palavra em duas imagens seguidas
    sorteio = random.Random(1)
    while True:
        sorteio.shuffle(amostra)
        palavras = [sem_acento(linha["palavra"]) for linha in amostra]
        if all(a != b for a, b in zip(palavras, palavras[1:])):
            break
    com_nomes = [(f"img_{n:03d}.png", linha) for n, linha in enumerate(amostra, 1)]

    if os.path.exists(PACOTE):
        shutil.rmtree(PACOTE)
    os.makedirs(os.path.join(PACOTE, "imagens"))
    for imagem, linha in com_nomes:
        shutil.copy(linha["caminho"], os.path.join(PACOTE, "imagens", imagem))
    paginas = folhas(com_nomes)

    with open(CHAVE, "w", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        escritor.writerow(["imagem", "palavra", "tipo", "painel"])
        for imagem, linha in com_nomes:
            escritor.writerow([imagem, linha["palavra"], linha["tipo"],
                               os.path.relpath(linha["caminho"], MODULO)])
    # ; e BOM para o Excel em portugues abrir com as colunas certas
    with open(os.path.join(PACOTE, "planilha_anotacao.csv"), "w", newline="", encoding="utf-8-sig") as f:
        escritor = csv.writer(f, delimiter=";")
        escritor.writerow(COLUNAS)
        for imagem, linha in com_nomes:
            escritor.writerow([imagem, letras_numeradas(linha["palavra"]), "", "", "", ""])
    with open(os.path.join(PACOTE, "instrucoes.txt"), "w", encoding="utf-8") as f:
        f.write(INSTRUCOES.format(n=len(com_nomes), folhas=len(paginas)))
    shutil.make_archive(PACOTE, "zip", PACOTE)

    contagem = {}
    for _, linha in com_nomes:
        sinais = "+".join(sorted({s for _, s in sinais_pedidos(linha["palavra"])}))
        com_i = " com i ou j" if set("ij") & set(sem_acento(linha["palavra"])) else ""
        grupo = (sinais or linha["tipo"]) + com_i
        contagem[grupo] = contagem.get(grupo, 0) + 1
    for grupo, n in sorted(contagem.items()):
        print(f"  {grupo:28s} {n}")
    print("total", len(com_nomes), "| pacote em", PACOTE + ".zip")


# ------------------------------------------------------------ concordancia

NAO_SEI = "não sei"
DESEMPATE = os.path.join(MODULO, "resultados", f"{NOME}_desempate.csv")


def ler_planilha(caminho):
    """{imagem: respostas}. Aceita ; ou , o BOM do Excel e o cp1252 do
    "CSV" comum do Excel no Windows."""
    try:
        with open(caminho, encoding="utf-8-sig") as f:
            texto = f.read()
    except UnicodeDecodeError:
        with open(caminho, encoding="cp1252") as f:
            texto = f.read()
    separador = ";" if texto.splitlines()[0].count(";") >= texto.splitlines()[0].count(",") else ","
    linhas = csv.DictReader(texto.splitlines(), delimiter=separador)
    return {linha["imagem"].strip(): linha for linha in linhas}


def letras_marcadas(texto):
    """'2 6' -> {2, 6}; 'nenhuma' -> set(); 'nao sei' ou vazio -> None.
    Qualquer outra coisa vira erro, para nao passar resposta mal digitada."""
    texto = texto.strip().lower()
    if texto in ("", NAO_SEI, "nao sei"):
        return None
    if texto == "nenhuma":
        return set()
    numeros = texto.replace(",", " ").split()
    if not all(n.isdigit() for n in numeros):
        raise ValueError(f"resposta que nao e lista de letras: {texto!r}")
    return {int(n) for n in numeros}


def classe_em_cima(palavra, marcadas):
    """As quatro classes, a partir das letras com marca em cima.

    Numa palavra sem sinal em cima pedido (sem acento, esqueleto ou so
    cedilha), "so a mais" quer dizer qualquer marca em cima."""
    if marcadas is None:
        return None
    pedidas = {posicao + 1 for posicao, sinal in sinais_pedidos(palavra) if sinal != "cedilha"}
    certas = pedidas and pedidas <= marcadas
    a_mais = bool(marcadas - pedidas)
    if certas and not a_mais:
        return "so o certo"
    if certas:
        return "certo e a mais"
    if a_mais:
        return "so a mais"
    return "nenhum"


def cohen(a, b):
    """Kappa de Cohen entre duas listas de rotulos (duas ou mais classes)."""
    if not a:
        return float("nan")
    classes = sorted(set(a) | set(b))
    concordam = sum(x == y for x, y in zip(a, b)) / len(a)
    ao_acaso = sum((a.count(c) / len(a)) * (b.count(c) / len(b)) for c in classes)
    return float("nan") if ao_acaso == 1 else (concordam - ao_acaso) / (1 - ao_acaso)


def respostas_humanas(linha, palavra):
    """As tres respostas de uma pessoa, ja no formato comparado. None = nao sei."""
    em_cima = letras_marcadas(linha["marcas_em_cima"])
    cedilha = letras_marcadas(linha["cedilha"])
    legivel = linha["legivel"].strip().lower()
    return {"classe": classe_em_cima(palavra, em_cima),
            "cedilha": None if cedilha is None else ("sim" if cedilha else "nao"),
            "legivel": {"sim": "sim", "não": "nao", "nao": "nao"}.get(legivel)}


DETECTOR = os.path.join(MODULO, "resultados", f"{NOME}_detector.csv")


def respostas_do_detector(chave):
    """Os detectores nas mesmas imagens, no mesmo formato das pessoas.

    classe e cedilha: scripts/medir_marcas.py com a letra dada pelo alinhador
    (scripts/medir_posicao_acento.py); cedilha e marca embaixo de um c.
    cedilha_abaixo: experimentos/cedilha_abaixo.py, tinta abaixo do c numa
    imagem so, com o limiar gravado em resultados/cedilha_abaixo_40ep.json.
    marca_paineis: a coluna "marca" do paineis.tsv, a usada em
    diagnostico/grafico_epocas.py para o acento a mais nas palavras so com
    cedilha."""
    import numpy as np
    from PIL import Image
    sys.path.insert(0, RAIZ)
    sys.path.insert(0, os.path.join(RAIZ, "scripts"))
    from acentos_sinteticos import alinhamento
    from medir_posicao_acento import letras_das_marcas

    sys.path.insert(0, os.path.join(MODULO, "experimentos"))
    import cedilha_abaixo

    alinhador = alinhamento.Alinhador(os.path.join(RAIZ, "modelos", "alinhador_iam.pt"), "cpu")
    cedilha_abaixo.ALINHADOR = alinhador
    with open(os.path.join(MODULO, "resultados", "cedilha_abaixo_40ep.json"), encoding="utf-8") as f:
        limiar_cedilha = json.load(f)["negativo"]["p95"]
    marca_do_painel = {os.path.relpath(l["caminho"], MODULO): l["marca"]
                       for pasta in (AVALIACAO, COM_I_J) for l in ler_paineis(pasta)}
    saida = {}
    for item in chave:
        cinza = np.asarray(Image.open(os.path.join(MODULO, item["painel"])).convert("L"),
                           dtype=np.float32)
        letras = sem_acento(item["palavra"])
        achadas, _ = letras_das_marcas(cinza, letras, alinhador)
        em_cima = {posicao + 1 for posicao, lado in achadas if lado == "acima"}
        cedilha = any(lado == "abaixo" and letras[posicao] == "c" for posicao, lado in achadas)
        nos_c = cedilha_abaixo.medidas_nos_c(cinza, letras) or []
        abaixo = any(medida > limiar_cedilha for _, medida, _, _ in nos_c)
        saida[item["imagem"]] = {"classe": classe_em_cima(item["palavra"], em_cima),
                                 "cedilha": "sim" if cedilha else "nao",
                                 "cedilha_abaixo": "sim" if abaixo else "nao",
                                 "marca_paineis": "sim" if marca_do_painel[item["painel"]] == "1" else "nao"}
    return saida


def detector(a):
    """Grava as respostas dos detectores num CSV, para a concordancia rodar
    depois sem torch e sem as imagens, num computador mais fraco."""
    with open(CHAVE, encoding="utf-8") as f:
        chave = list(csv.DictReader(f))
    respostas = respostas_do_detector(chave)
    campos = ["imagem"] + list(next(iter(respostas.values())))
    with open(DETECTOR, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=campos)
        escritor.writeheader()
        for imagem, r in respostas.items():
            escritor.writerow({"imagem": imagem, **r})
    print("gravado", DETECTOR)


def concordancia(a):
    with open(CHAVE, encoding="utf-8") as f:
        chave = list(csv.DictReader(f))
    pessoas = [os.path.splitext(os.path.basename(c))[0].replace("planilha_anotacao_", "")
               for c in a.planilhas]
    planilhas = [ler_planilha(c) for c in a.planilhas]
    humanos = []
    for pessoa, planilha in zip(pessoas, planilhas):
        respostas = {}
        for item in chave:
            if item["imagem"] not in planilha:
                print(f"{pessoa}: falta a linha de {item['imagem']}")
                continue
            try:
                respostas[item["imagem"]] = respostas_humanas(planilha[item["imagem"]], item["palavra"])
            except ValueError as erro:
                # resposta mal digitada fica de fora e aparece aqui para corrigir
                print(f"{pessoa}: {item['imagem']} {erro}")
        humanos.append(respostas)
    if os.path.exists(DETECTOR):
        with open(DETECTOR, encoding="utf-8") as f:
            detector = {linha["imagem"]: linha for linha in csv.DictReader(f)}
    else:
        detector = respostas_do_detector(chave)

    def grupo_da(item):
        if set("ij") & set(sem_acento(item["palavra"])):
            return "com i ou j"
        sinais = {s for _, s in sinais_pedidos(item["palavra"])}
        if not sinais:
            return "sem sinal pedido"
        return "so cedilha" if sinais == {"cedilha"} else "com sinal em cima"

    def comparar(titulo, pares):
        """pares: [(rotulo1, rotulo2)], sem os nao sei."""
        validos = [(x, y) for x, y in pares if x is not None and y is not None]
        a_, b_ = [x for x, _ in validos], [y for _, y in validos]
        concordam = sum(x == y for x, y in validos)
        print(f"  {titulo:34s} kappa {cohen(a_, b_):+.2f}  concordam {concordam} de {len(validos)}"
              f"  (fora {len(pares) - len(validos)} com nao sei)")

    for pergunta in ("classe", "cedilha", "legivel"):
        print(f"\n{pergunta.upper()}")
        for grupo in ("com sinal em cima", "so cedilha", "sem sinal pedido", "com i ou j"):
            itens = [i for i in chave if grupo_da(i) == grupo]
            if pergunta == "cedilha" and grupo == "com sinal em cima":
                itens = [i for i in itens if "ç" in i["palavra"]]
            if not itens:
                continue
            print(f" {grupo} ({len(itens)} imagens)")
            comparar(f"{pessoas[0]} x {pessoas[1]}",
                     [(humanos[0].get(i["imagem"], {}).get(pergunta),
                       humanos[1].get(i["imagem"], {}).get(pergunta)) for i in itens])
            # o detector conta o pingo como marca; nas palavras com i ou j
            # so as duas pessoas sao comparadas
            if pergunta != "legivel" and grupo != "com i ou j":
                for pessoa, h in zip(pessoas, humanos):
                    comparar(f"{pessoa} x detector",
                             [(h.get(i["imagem"], {}).get(pergunta), detector[i["imagem"]][pergunta])
                              for i in itens])
                    if pergunta == "cedilha":
                        comparar(f"{pessoa} x cedilha abaixo",
                                 [(h.get(i["imagem"], {}).get(pergunta),
                                   detector[i["imagem"]]["cedilha_abaixo"]) for i in itens])

    print("\nA MAIS NAS PALAVRAS SO COM CEDILHA, contra a coluna marca do paineis.tsv")
    itens = [i for i in chave if grupo_da(i) == "so cedilha"]
    for pessoa, h in zip(pessoas, humanos):
        pares = []
        for i in itens:
            classe = h.get(i["imagem"], {}).get("classe")
            humano = None if classe is None else ("sim" if classe != "nenhum" else "nao")
            pares.append((humano, detector[i["imagem"]]["marca_paineis"]))
        comparar(f"{pessoa} x marca do paineis.tsv", pares)

    consenso = {p: {imagem: r[p] for imagem, r in humanos[0].items()
                    if r[p] is not None and r[p] == humanos[1].get(imagem, {}).get(p)}
                for p in ("classe", "cedilha", "legivel")}
    contra_a_referencia("CONTRA O CONSENSO DAS DUAS PESSOAS", consenso, chave, detector, grupo_da)

    # imagens em que as duas pessoas discordam, para o desempate
    discordancias = []
    for item in chave:
        r1 = humanos[0].get(item["imagem"], {})
        r2 = humanos[1].get(item["imagem"], {})
        diferentes = [p for p in ("classe", "cedilha", "legivel") if r1.get(p) != r2.get(p)]
        if diferentes:
            discordancias.append({
                "imagem": item["imagem"], "palavra_pedida": item["palavra"],
                "perguntas": " ".join(diferentes),
                **{f"{pessoa}_{campo}": planilha[item["imagem"]][campo]
                   for pessoa, planilha in zip(pessoas, planilhas)
                   for campo in PERGUNTAS if item["imagem"] in planilha},
                "decisao_marcas_em_cima": "", "decisao_cedilha": "", "decisao_legivel": ""})
    # o desempate preenchido nao pode ser apagado por uma nova rodada
    if os.path.exists(DESEMPATE):
        print(f"\n{len(discordancias)} imagens com discordancia; desempate lido de {DESEMPATE}")
        gabarito_final(chave, consenso, discordancias, detector, grupo_da)
    elif discordancias:
        with open(DESEMPATE, "w", newline="", encoding="utf-8-sig") as f:
            escritor = csv.DictWriter(f, fieldnames=list(discordancias[0]), delimiter=";")
            escritor.writeheader()
            escritor.writerows(discordancias)
        print(f"\n{len(discordancias)} imagens com discordancia em {DESEMPATE}")


GABARITO = os.path.join(MODULO, "resultados", f"{NOME}_gabarito.csv")


def gabarito_final(chave, consenso, discordancias, detector, grupo_da):
    """A resposta comum onde as duas pessoas concordam e a decisao do
    desempate onde discordam. Grava o gabarito e compara os detectores com ele."""
    palavra = {i["imagem"]: i["palavra"] for i in chave}
    decisoes = {}
    for imagem, linha in ler_planilha(DESEMPATE).items():
        try:
            decisoes[imagem] = respostas_humanas(
                {campo: linha[f"decisao_{campo}"] for campo in PERGUNTAS}, palavra[imagem])
        except ValueError as erro:
            print(f"desempate: {imagem} {erro}")
    gabarito = {p: dict(respostas) for p, respostas in consenso.items()}
    desempatadas = {}
    for d in discordancias:
        for p in d["perguntas"].split():
            resposta = decisoes.get(d["imagem"], {}).get(p)
            if resposta is None:
                print(f"desempate: falta a decisao de {p} em {d['imagem']}")
                continue
            gabarito[p][d["imagem"]] = resposta
            desempatadas.setdefault(d["imagem"], []).append(p)
    # onde as duas pessoas concordam vale a resposta delas, mesmo que o
    # desempate tenha respondido a pergunta; aqui so se conta quando difere
    for imagem, r in decisoes.items():
        for p, resposta in r.items():
            if imagem in consenso[p] and resposta is not None and resposta != consenso[p][imagem]:
                print(f"desempate: {imagem} {p} = {resposta}, as duas pessoas disseram {consenso[p][imagem]}")

    with open(GABARITO, "w", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        escritor.writerow(["imagem", "palavra", "grupo", "classe", "cedilha", "legivel", "desempate"])
        for i in chave:
            escritor.writerow([i["imagem"], i["palavra"], grupo_da(i)]
                              + [gabarito[p].get(i["imagem"], "") for p in ("classe", "cedilha", "legivel")]
                              + [" ".join(desempatadas.get(i["imagem"], []))])
    print(f"{sum(len(v) for v in desempatadas.values())} respostas desempatadas, gabarito em {GABARITO}")

    print("\nGABARITO FINAL POR GRUPO")
    for grupo in ("com sinal em cima", "so cedilha", "sem sinal pedido", "com i ou j"):
        itens = [i["imagem"] for i in chave if grupo_da(i) == grupo]
        classes = {}
        for imagem in itens:
            classe = gabarito["classe"].get(imagem)
            classes[classe] = classes.get(classe, 0) + 1
        com_c = [i["imagem"] for i in chave if grupo_da(i) == grupo and "ç" in i["palavra"]]
        sem_c = [imagem for imagem in itens if imagem not in com_c]
        print(f" {grupo} ({len(itens)} imagens)")
        print("  classe   " + ", ".join(f"{c} {n}" for c, n in sorted(classes.items(), key=str)))
        print(f"  cedilha  vista em {sum(gabarito['cedilha'].get(x) == 'sim' for x in com_c)} de {len(com_c)}"
              f" pedidas com ç e em {sum(gabarito['cedilha'].get(x) == 'sim' for x in sem_c)} de {len(sem_c)} sem ç")
        print(f"  legivel  {sum(gabarito['legivel'].get(x) == 'sim' for x in itens)} de {len(itens)}")

    contra_a_referencia("CONTRA O GABARITO FINAL", gabarito, chave, detector, grupo_da)


def contra_a_referencia(titulo, referencia, chave, detector, grupo_da):
    """Precisao, recall e kappa dos detectores contra uma referencia humana,
    {pergunta: {imagem: resposta}}, nas imagens que ela tem. As palavras com
    i ou j ficam de fora, porque o detector conta o pingo como marca."""
    def com_resposta(pergunta):
        return [i for i in chave if grupo_da(i) != "com i ou j"
                and referencia[pergunta].get(i["imagem"]) is not None]

    def linha(nome, itens, humano, maquina):
        h = [humano(i) for i in itens]
        m = [maquina(i) for i in itens]
        vp = sum(x and y for x, y in zip(h, m))
        fp = sum(y and not x for x, y in zip(h, m))
        fn = sum(x and not y for x, y in zip(h, m))
        precisao = f"{vp / (vp + fp):.2f}" if vp + fp else "  -"
        recall = f"{vp / (vp + fn):.2f}" if vp + fn else "  -"
        print(f"  {nome:44s} n={len(itens):3d} VP={vp:3d} FP={fp:3d} FN={fn:3d}"
              f"  precisao {precisao}  recall {recall}  kappa {cohen(h, m):+.2f}")

    def certo(classe):
        return classe in ("so o certo", "certo e a mais")

    def a_mais(classe):
        return classe in ("so a mais", "certo e a mais")

    def ref(pergunta):
        return lambda i: referencia[pergunta][i["imagem"]]

    print(f"\n{titulo}, sem as palavras com i ou j")
    cedilha = com_resposta("cedilha")
    linha("cedilha, detector do colega", cedilha,
          lambda i: ref("cedilha")(i) == "sim", lambda i: detector[i["imagem"]]["cedilha"] == "sim")
    linha("cedilha, cedilha abaixo do c", cedilha,
          lambda i: ref("cedilha")(i) == "sim", lambda i: detector[i["imagem"]]["cedilha_abaixo"] == "sim")
    classe = com_resposta("classe")
    em_cima = [i for i in classe if grupo_da(i) == "com sinal em cima"]
    linha("acento pedido presente, colega", em_cima,
          lambda i: certo(ref("classe")(i)), lambda i: certo(detector[i["imagem"]]["classe"]))
    linha("acento a mais, colega", em_cima,
          lambda i: a_mais(ref("classe")(i)), lambda i: a_mais(detector[i["imagem"]]["classe"]))
    linha("alguma marca em cima, colega", classe,
          lambda i: ref("classe")(i) != "nenhum", lambda i: detector[i["imagem"]]["classe"] != "nenhum")
    so_cedilha = [i for i in classe if grupo_da(i) == "so cedilha"]
    linha("a mais nas so cedilha, marca do paineis.tsv", so_cedilha,
          lambda i: ref("classe")(i) != "nenhum", lambda i: detector[i["imagem"]]["marca_paineis"] == "sim")
    linha("a mais nas so cedilha, so marcas em cima", so_cedilha,
          lambda i: ref("classe")(i) != "nenhum", lambda i: detector[i["imagem"]]["classe"] != "nenhum")


# --------------------------------------------------------- palavras com i e j

CHECKPOINT_40EP = os.path.join(RAIZ, "modelos", "zona_vogais_40ep.pt")
PALAVRAS_COM_ACENTO_NO_I = 4
PALAVRAS_COM_I_E_OUTRO_ACENTO = 4


def escolher_palavras_com_i_j(avaliacao):
    """Palavras de validacao com i ou j e um sinal so, fora da base de treino.
    Metade com o acento no proprio i, metade com o acento em outra letra."""
    sys.path.insert(0, RAIZ)
    sys.path.insert(0, os.path.join(RAIZ, "scripts"))
    from acentos_sinteticos.vocabulario import Vocabulario, grupo
    from avaliar_pt import conferir_bases

    with open(os.path.join(avaliacao, "resumo.json"), encoding="utf-8") as f:
        ja_avaliadas = set(json.load(f)["palavras"])
    vocabulario = Vocabulario()
    candidatas = sorted(
        p for p in vocabulario.lista("val", True)
        if 3 <= len(p) <= 10 and set("ij") & set(sem_acento(p))
        and len(sinais_pedidos(p)) == 1 and p not in ja_avaliadas
        and p not in EXCLUIDAS and grupo(p) not in EXCLUIDAS)
    no_i = [p for p in candidatas if "í" in p]
    outro = [p for p in candidatas if "í" not in p]
    sorteio = random.Random(0)
    palavras = sorteio.sample(no_i, PALAVRAS_COM_ACENTO_NO_I) + \
        sorteio.sample(outro, PALAVRAS_COM_I_E_OUTRO_ACENTO)
    itens = [(p, "acentuada", sem_acento(p)) for p in palavras] + \
            [(sem_acento(p), "esqueleto", sem_acento(p)) for p in palavras]
    vocabulario.conferir([texto for texto, _, _ in itens], "val", "palavras com i e j")
    conferir_bases(itens, [os.path.join(RAIZ, "iam_pt_alinhado")])
    return itens


def gerar_i_j(a):
    """Gera as palavras com i e j na CPU, pelo mesmo protocolo do avaliar_pt.py:
    os mesmos 20 escritores, as mesmas referencias de estilo, semente 42.
    Grava no mesmo formato de pasta (paineis/<modelo>/ e paineis.tsv)."""
    import numpy as np
    from PIL import Image
    from comum.diffusionpen import EscritoresIAM, GeradorDiffusionPen
    from medir_marcas import marcas

    itens = escolher_palavras_com_i_j(AVALIACAO)
    with open(os.path.join(AVALIACAO, "resumo.json"), encoding="utf-8") as f:
        escritores = json.load(f)["escritores"]
    iam_teste = EscritoresIAM("iam_test.txt")
    gerador = GeradorDiffusionPen(CHECKPOINT_40EP, os.path.join(
        RAIZ, "DiffusionPen", "style_models", "iam_style_diffusionpen.pth"), "cpu")
    modelo = "zona_vogais_40ep"
    pasta = os.path.join(COM_I_J, "paineis", modelo)
    os.makedirs(pasta, exist_ok=True)
    linhas = ["modelo\tpalavra\ttipo\tescritor\tsemente\tmarca"]
    semente = 42
    referencias = [iam_teste.referencias(e, random.Random(semente * 1000 + j))
                   for j, e in enumerate(escritores)]
    for numero, (texto, tipo, _) in enumerate(itens):
        imagens = gerador.gerar([texto] * len(escritores), referencias, semente)
        for escritor, imagem in zip(escritores, imagens):
            cinza = (imagem.mean(0).numpy() * 255).astype(np.float32)
            acima, abaixo, _, _ = marcas(cinza)
            Image.fromarray(cinza.astype(np.uint8)).save(
                os.path.join(pasta, f"{numero:03d}_{escritor}_{semente}.png"))
            linhas.append(f"{modelo}\t{texto}\t{tipo}\t{escritor}\t{semente}\t{int(acima + abaixo > 0)}")
        print(f"{numero + 1}/{len(itens)} {texto}", flush=True)
    with open(os.path.join(COM_I_J, "paineis.tsv"), "w", encoding="utf-8") as f:
        f.write("\n".join(linhas) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("comando", choices=["gerar_i_j", "pacote", "detector", "concordancia"])
    ap.add_argument("--planilhas", nargs=2)
    a = ap.parse_args()
    if a.comando == "gerar_i_j":
        sys.path.insert(0, RAIZ)
        sys.path.insert(0, os.path.join(RAIZ, "scripts"))
    {"gerar_i_j": gerar_i_j, "pacote": pacote, "detector": detector,
     "concordancia": concordancia}[a.comando](a)


if __name__ == "__main__":
    main()
