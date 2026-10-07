# Vocabulário português (particionado)

## Fonte e licença

`fonte/pt_br_50k.txt` vem do projeto **FrequencyWords**, de Hermit Dave
(https://github.com/hermitdave/FrequencyWords), lista `content/2018/pt_br/pt_br_50k.txt`.
Ela foi derivada do OpenSubtitles 2018 (http://opus.nlpl.eu/OpenSubtitles2018.php).
O conteúdo é distribuído sob **CC-BY-SA 4.0**, e o código do projeto sob MIT.

`palavras.tsv` é uma obra derivada dessa lista e segue a mesma licença (CC-BY-SA 4.0),
com atribuição ao FrequencyWords e ao OpenSubtitles. O SHA-256 do arquivo de origem
está em `resumo.json`.

## Arquivos

- `palavras.tsv`, com uma linha por palavra e as colunas:
  `palavra`, `esqueleto` (sem acento), `grupo`, `split` (`treino`/`val`/`teste`),
  `frequencia`, `acentuada` (0/1) e `forcada` (o motivo, quando o split foi imposto).
- `resumo.json`: contagens, filtros e hash da fonte.

Os dois são gerados por `scripts/preparar_vocabulario_pt.py`.

## Regra anti-vazamento

A partição é por **grupo**: o esqueleto sem acento com o plural dobrado. Assim,
`nação`/`nações`/`nacao` e `está`/`esta` caem sempre no mesmo split.

- **Teste:** toda palavra que o repositório usa para avaliar (sonda, pares da métrica,
  amostras dos experimentos, medição de marcas) e as palavras acentuadas do teste do
  BRESSAY.
- **Validação:** as palavras acentuadas da validação do BRESSAY.
- **Resto:** hash estável do grupo, com 10% para o teste, 5% para a validação e o
  restante para o treino.

Este arquivo foi **congelado antes de qualquer imagem ser gerada**. Os scripts da base
portuguesa e o leitor do treino conferem cada rótulo contra ele
(`acentos_sinteticos/vocabulario.py`) e param com erro se uma palavra de
validação ou teste aparecer no treino. Não regenere o arquivo depois que houver base ou
modelo treinado a partir dele: mudar a partição invalida a separação.
