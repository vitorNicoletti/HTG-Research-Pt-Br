"""Onde esta cada letra: alinhamento forcado com um reconhecedor CTC.

As fatias iguais erram quando as letras tem larguras muito diferentes (o "f"
de "for" ocupa metade da palavra). Aqui um reconhecedor pequeno, treinado no
proprio IAM (scripts/treinar_alinhador.py), le a palavra em quadros de PASSO
px e da a probabilidade de cada caractere por quadro. Como a transcricao e
conhecida, nao se decodifica: o caminho de Viterbi do CTC restrito ao texto
diz em que quadro cada letra "dispara". As fronteiras das fatias ficam no
meio entre os disparos de letras vizinhas.

O modelo e so convolucional (sem LSTM): o campo receptivo limitado mantem o
disparo de cada letra perto da propria letra, que e o que interessa aqui --
a taxa de acerto da leitura livre e secundaria.
"""

from dataclasses import dataclass

import cv2
import numpy as np

ALTURA = 64          # altura normalizada da palavra
PASSO = 4            # px da imagem normalizada por quadro da sequencia
LARGURA_MAX = 1024   # px; palavras mais largas sao comprimidas
BRANCO = 0           # indice do branco do CTC


def normalizar(g, esticar=1.0):
    """Cinza 0..255 -> (imagem ALTURA x L em 0..1 com tinta=1, px originais por px).

    L e multiplo de PASSO. `esticar` alarga ou estreita (aumento de dados).
    """
    h, w = g.shape
    nl = int(round(w * ALTURA / h * esticar))
    nl = min(max(nl, 2 * PASSO), LARGURA_MAX)
    nl += (-nl) % PASSO
    im = cv2.resize(np.asarray(g, dtype=np.float32), (nl, ALTURA),
                    interpolation=cv2.INTER_AREA)
    im = 255.0 - im
    im /= max(float(im.max()), 1.0)
    return im, w / nl


def criar_modelo(n_classes, cabeca="conv"):
    """CNN 2D (altura 64 -> 1) + cabeca sobre os quadros; n_classes inclui o branco.

    cabeca "conv" (alinhador): CNN 1D, campo receptivo local -- o disparo de
    cada letra fica perto dela. cabeca "lstm" (leitor da avaliacao): LSTM
    bidirecional, le melhor mas nao localiza; e um modelo SEPARADO do
    alinhador para a legibilidade nao ser medida pelo mesmo modelo que
    filtrou a base (avaliacao circular).
    """
    if cabeca not in ("conv", "lstm"):
        raise ValueError(f"cabeca desconhecida: {cabeca}")
    import torch.nn as nn

    def bloco(a, b):
        return [nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True)]

    class Leitor(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = nn.Sequential(
                *bloco(1, 64), nn.MaxPool2d(2),                      # 32 x L/2
                *bloco(64, 128), nn.MaxPool2d(2),                    # 16 x L/4
                *bloco(128, 256), *bloco(256, 256), nn.MaxPool2d((2, 1)),  # 8
                *bloco(256, 256), *bloco(256, 256), nn.MaxPool2d((2, 1)),  # 4
            )
            if cabeca == "conv":
                seq = []
                for _ in range(4):
                    seq += [nn.Conv1d(256 * 4 if not seq else 256, 256, 5, padding=2),
                            nn.BatchNorm1d(256), nn.ReLU(inplace=True)]
                self.seq = nn.Sequential(*seq, nn.Dropout(0.2), nn.Conv1d(256, n_classes, 1))
            else:
                self.proj = nn.Linear(256 * 4, 256)
                self.lstm = nn.LSTM(256, 128, num_layers=2, bidirectional=True,
                                    dropout=0.2, batch_first=True)
                self.saida = nn.Linear(256, n_classes)

        def forward(self, x):
            """x: (B, 1, ALTURA, L) -> log-probabilidades (T, B, C), T = L / PASSO."""
            f = self.cnn(x)
            b, c, h, t = f.shape
            f = f.reshape(b, c * h, t)
            if cabeca == "conv":
                y = self.seq(f).permute(2, 0, 1)
            else:
                z, _ = self.lstm(self.proj(f.permute(0, 2, 1)))
                y = self.saida(z).permute(1, 0, 2)
            return y.log_softmax(-1)

    return Leitor()


def viterbi_ctc(logp, alvo):
    """Caminho mais provavel do CTC que soletra exatamente `alvo`.

    logp: (T, C) log-probabilidades; alvo: lista de indices (sem branco).
    Devolve (estado por quadro, log-prob do caminho) ou None se T for curto.
    Estados: 0 = branco inicial, 2k+1 = letra k, 2k+2 = branco depois dela.
    """
    T, L = logp.shape[0], len(alvo)
    ext = np.full(2 * L + 1, BRANCO, dtype=np.int64)
    ext[1::2] = alvo
    S = len(ext)
    # pode pular o branco entre letras diferentes
    pula = np.zeros(S, dtype=bool)
    pula[3::2] = ext[3::2] != ext[1:-2:2]
    menos_inf = -np.inf
    dp = np.full(S, menos_inf)
    dp[0] = logp[0, ext[0]]
    if S > 1:
        dp[1] = logp[0, ext[1]]
    volta = np.zeros((T, S), dtype=np.int8)   # 0 fica, 1 veio de s-1, 2 de s-2
    for t in range(1, T):
        c0 = dp
        c1 = np.concatenate([[menos_inf], dp[:-1]])
        c2 = np.where(pula, np.concatenate([[menos_inf, menos_inf], dp[:-2]]), menos_inf)
        pilha = np.stack([c0, c1, c2])
        volta[t] = np.argmax(pilha, axis=0)
        dp = pilha.max(axis=0) + logp[t, ext]
    fins = [S - 1] + ([S - 2] if S > 1 else [])
    s = max(fins, key=lambda k: dp[k])
    if not np.isfinite(dp[s]):
        return None
    total = float(dp[s])
    estados = np.empty(T, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        estados[t] = s
        s -= int(volta[t, s])
    return estados, total


def decodificar(logp, alfabeto):
    """Leitura livre (gulosa) de (T, C) log-probabilidades."""
    ids = logp.argmax(-1)
    saida, ant = [], BRANCO
    for i in ids:
        if i != ant and i != BRANCO:
            saida.append(alfabeto[i - 1])
        ant = i
    return "".join(saida)


@dataclass
class Alinhamento:
    centros: list        # x (px da imagem original) do disparo de cada letra
    leitura: str         # leitura livre do modelo (para filtro de qualidade)
    logp_medio: float    # log-prob do caminho forcado por quadro


class Alinhador:
    """Carrega o reconhecedor treinado e alinha transcricoes conhecidas."""

    def __init__(self, caminho, device="cpu"):
        import torch
        ck = torch.load(caminho, map_location=device, weights_only=False)
        self.alfabeto = ck["alfabeto"]
        self.indice = {c: k + 1 for k, c in enumerate(self.alfabeto)}
        self.modelo = criar_modelo(len(self.alfabeto) + 1, ck.get("cabeca", "conv")).to(device)
        self.modelo.load_state_dict(ck["estado"])
        self.modelo.eval()
        self.device = device
        self.torch = torch

    def logp(self, g):
        """(T, C) log-probabilidades e px originais por quadro."""
        im, esc = normalizar(g)
        x = self.torch.from_numpy(im)[None, None].to(self.device)
        with self.torch.no_grad():
            y = self.modelo(x)[:, 0].float().cpu().numpy()
        return y, esc * PASSO

    def ler(self, g):
        """Leitura livre (gulosa) da imagem."""
        y, _ = self.logp(g)
        return decodificar(y, self.alfabeto)

    def alinhar(self, g, texto):
        """Alinhamento de `texto` sobre a imagem; None se impossivel."""
        if not texto or any(c not in self.indice for c in texto):
            return None
        y, px = self.logp(g)
        r = viterbi_ctc(y, [self.indice[c] for c in texto])
        if r is None:
            return None
        estados, total = r
        centros = []
        for k in range(len(texto)):
            q = np.where(estados == 2 * k + 1)[0]
            centros.append((float(q.mean()) + 0.5) * px)
        return Alinhamento(centros, decodificar(y, self.alfabeto), total / len(y))

    def fatias(self, g, geo, texto):
        """Fatias por letra (ver fatias_do_alinhamento); None se nao alinhar."""
        a = self.alinhar(g, texto)
        return None if a is None else fatias_do_alinhamento(a, geo)


def fatias_do_alinhamento(a, geo):
    """Fatias por letra no formato de geometria.fatias: [(xa, xb)].
    Fronteira = meio entre disparos vizinhos; as pontas sao a caixa da tinta.
    O disparo cai, em mediana, ~0,25 largura de letra a direita do centro;
    o meio entre disparos compensa quase todo esse vies."""
    x0, x1 = geo.caixa[0], geo.caixa[1]
    c = np.clip(a.centros, x0, x1)
    fr = [x0] + [float((c[k] + c[k + 1]) / 2) for k in range(len(c) - 1)] + [x1]
    return [(fr[k], fr[k + 1]) for k in range(len(c))]
