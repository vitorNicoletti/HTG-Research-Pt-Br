"""Monta a folha de contato da sonda de diacriticos (Fase 2, passo 4).

Le os PNGs gerados em saidas/<modelo>/sonda/ e monta um grid agrupado por
categoria, com a palavra SOLICITADA rotulada acima de cada imagem. O rotulo e
o que torna a inspecao visual possivel: sem ele nao da para dizer se o modelo
errou ou se voce leu errado.

Uso:
    python comum/folha_contato.py saidas/diffusionpen/sonda
"""

import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import palavras as P  # noqa: E402

MARGEM = 20
ESPACO_X = 12
ESPACO_Y = 8
ALT_ROTULO = 22
ALT_TITULO = 34
FUNDO = (255, 255, 255)
COR_TEXTO = (0, 0, 0)
COR_TITULO = (140, 0, 0)
COR_FALTANDO = (245, 245, 245)
COR_BORDA = (200, 200, 200)


def carregar_fonte(tamanho):
    """Precisa de uma fonte que renderize acentos - a bitmap default do PIL nao
    renderiza, e ai os rotulos sairiam errados justamente nas palavras que
    importam.

    Pergunta ao fontconfig (fc-match) em vez de chutar caminhos: quais fontes
    existem varia demais entre distros para uma lista fixa funcionar.
    """
    try:
        saida = subprocess.run(
            ["fc-match", "-f", "%{file}", "sans:lang=pt"],
            capture_output=True, text=True, timeout=10,
        )
        caminho = saida.stdout.strip()
        if caminho and Path(caminho).exists():
            return ImageFont.truetype(caminho, tamanho)
    except (FileNotFoundError, subprocess.SubprocessError):
        pass

    for caminho in [
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/gnu-free/FreeSans.otf",
    ]:
        if Path(caminho).exists():
            return ImageFont.truetype(caminho, tamanho)

    print("AVISO: nenhuma fonte TrueType encontrada; rotulos com acento podem "
          "sair incorretos.", file=sys.stderr)
    return ImageFont.load_default()


def montar(dir_sonda: Path, saida: Path):
    fonte = carregar_fonte(14)
    fonte_titulo = carregar_fonte(19)

    # descobre o tamanho das celulas a partir das imagens que existem
    encontradas = {}
    for grupo, palavra, seed in P.itens():
        caminho = dir_sonda / P.nome_arquivo(grupo, palavra, seed)
        if caminho.exists():
            encontradas[(grupo, palavra, seed)] = caminho

    if not encontradas:
        sys.exit(f"FALHA: nenhuma imagem da sonda encontrada em {dir_sonda}")

    amostras = [Image.open(c) for c in list(encontradas.values())[:20]]
    cel_l = max(im.width for im in amostras)
    cel_a = max(im.height for im in amostras)
    for im in amostras:
        im.close()

    n_seeds = len(P.SEEDS)
    largura_bloco = n_seeds * cel_l + (n_seeds - 1) * ESPACO_X
    n_colunas = max(len(v) for v in P.GRUPOS.values())

    largura = MARGEM * 2 + n_colunas * largura_bloco + (n_colunas - 1) * ESPACO_X * 3
    altura = MARGEM
    for _ in P.GRUPOS:
        altura += ALT_TITULO + ALT_ROTULO + cel_a + ESPACO_Y * 3
    altura += MARGEM

    folha = Image.new("RGB", (largura, altura), FUNDO)
    d = ImageDraw.Draw(folha)

    y = MARGEM
    faltando = 0
    for grupo, lista in P.GRUPOS.items():
        d.text((MARGEM, y), f"{grupo.upper()}  (seeds: {P.SEEDS})",
               fill=COR_TITULO, font=fonte_titulo)
        y += ALT_TITULO

        x = MARGEM
        for palavra in lista:
            d.text((x, y), palavra, fill=COR_TEXTO, font=fonte)
            yy = y + ALT_ROTULO
            for i, seed in enumerate(P.SEEDS):
                cx = x + i * (cel_l + ESPACO_X)
                chave = (grupo, palavra, seed)
                if chave in encontradas:
                    with Image.open(encontradas[chave]) as im:
                        folha.paste(im.convert("RGB"), (cx, yy))
                else:
                    # celula vazia explicita: ausencia de imagem e um resultado,
                    # nao pode virar espaco em branco ambiguo
                    d.rectangle([cx, yy, cx + cel_l, yy + cel_a],
                                fill=COR_FALTANDO, outline=COR_BORDA)
                    d.text((cx + 6, yy + cel_a // 2 - 7), "ausente",
                           fill=(150, 150, 150), font=fonte)
                    faltando += 1
            x += largura_bloco + ESPACO_X * 3
        y += ALT_ROTULO + cel_a + ESPACO_Y * 3

    saida.parent.mkdir(parents=True, exist_ok=True)
    folha.save(saida)
    total = len(list(P.itens()))
    print(f"folha de contato: {saida}")
    print(f"celulas: {total}  presentes: {len(encontradas)}  ausentes: {faltando}")
    return faltando


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    dir_sonda = Path(sys.argv[1])
    saida = Path(sys.argv[2]) if len(sys.argv) > 2 else dir_sonda.parent / "folha_contato.png"
    montar(dir_sonda, saida)
