# Figuras para o artigo

Cada figura está em PNG (200 dpi) e em PDF vetorial. Todas saem de um script só:

```bash
python scripts/gerar_original_e_melhor.py --saida arquivo/figuras_artigo   # imagens, na CPU, uns 15 min
python scripts/figuras_artigo.py                                           # todas as figuras
```

Os gráficos leem só as tabelas versionadas em `avaliacao_diacriticos/resultados/`. As duas figuras com imagens precisam das imagens que o primeiro comando gera. O modelo original e o melhor modelo foram gerados de novo nesta máquina, na CPU, com os mesmos escritores, referências e semente do protocolo (`scripts/avaliar_pt.py`). As imagens do melhor modelo saíram iguais às da avaliação, com no máximo 1 nível de cinza de diferença.

| arquivo | o que mostra | onde entra |
|---|---|---|
| `original_e_melhor` | cinco palavras de validação, uma por classe de sinal, no modelo original e no zona vogais de 40 épocas, com os mesmos quatro escritores e a mesma semente | introdução ou resultados |
| `como_a_metrica_olha` | uma imagem com a faixa do corpo, as colunas das letras do alinhador, a marca achada em cima e a região da cedilha; ao lado, a mesma imagem com os sinais apagados, que é o que o leitor do E2 recebe | método da métrica |
| `categorias_por_modelo` | E1 por modelo: correto, a mais, letra errada e ausente, nas 1.200 imagens acentuadas | resultados |
| `quatro_categorias` | E1 cruzado com E2: acerto, omissão, degradação com e sem diacrítico | resultados |
| `por_classe` | palavras corretas por classe de sinal no modelo de 40 épocas, pelo gabarito humano e pela métrica | validação da métrica |
| `evolucao_epocas` | zona vogais de 6 a 40 épocas: correto (E1), palavra degradada (E2) e marca em palavra sem acento | resultados do treino |

Os nomes dos modelos são os da documentação (`docs/15`, `docs/09`). Se o artigo usar outros, basta trocar o dicionário `ROTULOS` em `scripts/figuras_artigo.py` e rodar de novo.

Cuidados ao legendar:

- Em `quatro_categorias`, "com diacrítico" quer dizer que os sinais pedidos estão presentes, sem olhar a letra. Por isso acerto não é subconjunto de correto.
- A métrica subestima o til (`por_classe`). Ele encosta na letra ou sai partido, e a métrica perde parte deles.
- O E2 ainda não tem validação humana direta.
