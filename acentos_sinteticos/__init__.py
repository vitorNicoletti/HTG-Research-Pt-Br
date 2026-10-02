"""Acentos sinteticos sobre recortes de palavras do IAM.

Ideia: o IAM tem escrita em boa resolucao, mas nao tem diacriticos do
portugues. Desenha-se o acento (ou a cedilha) sobre uma palavra real do IAM e
troca-se a letra no rotulo: "can" + til sobre o "a" -> imagem nova, rotulo
"cãn". O modelo aprende a relacao "ã no texto <-> til sobre o a na imagem"
com o traco, a resolucao e o estilo do escritor preservados.

Modulos:
    iam        -- lista de palavras e leitura das imagens do IAM
    geometria  -- mascara de tinta, corpo da palavra (altura-x e base),
                  espessura do traco, fatias por letra, ajuste das fronteiras
                  aos vales de tinta e pontos de contato
    alinhamento -- reconhecedor CTC (scripts/treinar_alinhador.py) e
                  alinhamento forcado da transcricao: onde esta cada letra
    tracos     -- formas parametricas dos sinais (til, agudo, grave,
                  circunflexo, cedilha), com variacao aleatoria
    desenho    -- rasterizacao do traco com espessura e tom da tinta
    gerador    -- escolhe a letra e o sinal, posiciona, desenha, troca o rotulo

Uso: scripts/amostras_acentos.py; posicao das letras medida por
scripts/avaliar_posicao_letras.py.
"""
