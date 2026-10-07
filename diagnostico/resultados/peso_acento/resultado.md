# Peso do acento na perda — 96 pares da base iam_pt

Pares cuja imagem acentuada ganhou margem (o sinal passou da borda): 61 de 96. Aqui eles foram realinhados; no treino entram em escalas diferentes.

## 1. Quanto do latente o acento ocupa (depende so dos dados)

- area do acento: 0.46% dos pixels; 2.09% das posicoes do latente 8x32
- da diferenca entre os latentes com e sem acento, 70% cai nas posicoes do acento (o resto e o VAE espalhando)

## 1b. Quanto a perda sobe se o modelo desenhar SEM acento (penalidade) contra a perda tipica do modelo com o texto certo

| t | penalidade | perda iam | perda pt_16ep | razao iam | razao pt_16ep |
|---|---|---|---|---|---|
| 25 | 1.00815 | 1.1201 | 0.3387 | 90.0% | 297.7% |
| 100 | 0.20415 | 0.6111 | 0.1682 | 33.4% | 121.4% |
| 250 | 0.04988 | 0.2598 | 0.0870 | 19.2% | 57.4% |
| 500 | 0.00922 | 0.0790 | 0.0361 | 11.7% | 25.5% |
| 750 | 0.00144 | 0.0161 | 0.0100 | 8.9% | 14.3% |
| 950 | 0.00020 | 0.0024 | 0.0017 | 8.2% | 11.7% |

## 2. O modelo usa o diacritico do texto? (imagem acentuada; texto certo x errado)

| modelo | t | perda certo | perda errado | aumento | na mascara: certo | na mascara: errado | sensibilidade na mascara | sensibilidade fora | razao dentro/fora |
|---|---|---|---|---|---|---|---|---|---|
| iam | 25 | 1.1201 | 1.1190 | -0.11% | 8.0576 | 8.7753 | 3.83e-01 | 6.85e-02 | 5.6x |
| iam | 100 | 0.6111 | 0.5937 | -2.86% | 4.8625 | 5.0165 | 1.70e-01 | 5.91e-02 | 2.9x |
| iam | 250 | 0.2598 | 0.2521 | -2.97% | 1.5072 | 1.5271 | 6.21e-02 | 2.98e-02 | 2.1x |
| iam | 500 | 0.0790 | 0.0761 | -3.62% | 0.3032 | 0.3076 | 1.45e-02 | 9.16e-03 | 1.6x |
| iam | 750 | 0.0161 | 0.0158 | -2.01% | 0.0479 | 0.0478 | 2.98e-03 | 1.75e-03 | 1.7x |
| iam | 950 | 0.0024 | 0.0024 | -2.12% | 0.0069 | 0.0069 | 3.98e-04 | 2.44e-04 | 1.6x |
| pt_16ep | 25 | 0.3387 | 0.3387 | +0.01% | 0.6556 | 0.6623 | 9.18e-03 | 4.73e-03 | 1.9x |
| pt_16ep | 100 | 0.1682 | 0.1691 | +0.50% | 0.4627 | 0.4899 | 1.74e-02 | 4.81e-03 | 3.6x |
| pt_16ep | 250 | 0.0870 | 0.0887 | +1.99% | 0.3691 | 0.4213 | 2.58e-02 | 3.94e-03 | 6.6x |
| pt_16ep | 500 | 0.0361 | 0.0368 | +1.82% | 0.2012 | 0.2220 | 8.67e-03 | 2.25e-03 | 3.8x |
| pt_16ep | 750 | 0.0100 | 0.0104 | +3.68% | 0.0413 | 0.0433 | 1.16e-03 | 7.35e-04 | 1.6x |
| pt_16ep | 950 | 0.0017 | 0.0017 | +1.90% | 0.0059 | 0.0061 | 1.71e-04 | 1.13e-04 | 1.5x |

## 3. O codificador de texto distingue com e sem acento?

| modelo | camada | cos(acentuada, esqueleto) | cos(acentuada, outra palavra) | dist. rel. ao esqueleto | dist. rel. a outra |
|---|---|---|---|---|---|
| iam | canine | 0.9352 | 0.1102 | 0.325 | 1.342 |
| iam | text_lin | 0.9396 | 0.1184 | 0.313 | 1.339 |
| pt_16ep | canine | 0.9352 | 0.1102 | 0.325 | 1.342 |
| pt_16ep | text_lin | 0.9398 | 0.1222 | 0.312 | 1.336 |
