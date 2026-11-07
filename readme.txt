HTG-Research-Pt-Br
Fidelidade de diacríticos na geração de escrita manuscrita em português
brasileiro com o DiffusionPen.

Creative Experience: Transformative Project II, Ciência da Computação, PUCPR,
Turma B, Equipe 13. Vitor Nicoletti, Vinícius Y. Borges, Bruno H. O. M.
Dutra, Leonardo Saito.


1. ONDE ESTÃO OS ARTEFATOS
==========================

Código, dados pequenos, logs, resultados, figuras e documentação:
  https://github.com/vitorNicoletti/HTG-Research-Pt-Br  (branch main)

Modelos treinados, base de treino e imagens geradas (grandes demais para o
git), numa pasta do Google Drive:
  https://drive.google.com/drive/folders/17-0RxtQUJNfRVGx2uBI07-xzw2dMSb2s?usp=sharing

Não redistribuídos, por licença ou tamanho. Ver docs/11_como_rodar.md:
  - IAM Handwriting Database, palavras (words.tgz): fki.tic.heia-fr.ch, exige
    cadastro. Vai em DiffusionPen/iam_data/words/.
  - Código e pesos originais do DiffusionPen: github.com/koninik/DiffusionPen
    e huggingface.co/konnik/DiffusionPen. Vão em DiffusionPen/.
  - VAE e agendador do Stable Diffusion 1.5:
    stable-diffusion-v1-5/stable-diffusion-v1-5 no Hugging Face, baixados
    automaticamente.
  - BRESSAY (4,3 GB), usado só no fine-tune da fase 3: site oficial do
    dataset. Vai em bressay/data/words/.

Todos os números do artigo são do split de validação. O split de teste do
vocabulário nunca foi aberto.


2. PASTA DO GOOGLE DRIVE
========================

modelos/zona_vogais_40ep.pt
    Modelo final: DiffusionPen com acento separado, professor e peso 5 no
    acento e na zona das vogais, 40 épocas. É o modelo das figuras, das
    tabelas e da anotação humana do artigo. Copiar para modelos/.
modelos/zona_vogais_30ep.pt
    O mesmo treino com 30 épocas. Copiar para modelos/.
modelos/alinhador_iam.pt
    Alinhador de letras (CNN + CTC), treinado no IAM. Diz onde está cada
    letra: usado para desenhar o acento na base sintética e para dar a letra
    de cada marca na métrica. Copiar para modelos/.
modelos/leitor_iam_transformer.pt
    Leitor (HTR, CNN + Transformer + CTC), treinado no IAM, CER 0,084 no
    iam_val. Usado no CER do protocolo e no E2 da métrica. Copiar para
    modelos/.
dados/iam_pt_alinhado.tar
    Base sintética de treino: palavras portuguesas geradas pelo DiffusionPen
    original, com o acento desenhado sobre a letra certa e um par sem acento
    alinhado; inclui as máscaras da zona das vogais (zona_vogais.npz).
    Extrair na raiz (cria iam_pt_alinhado/).
imagens/avaliacoes.tar
    As 3.200 imagens do protocolo de validação de cada treino com acento
    separado, base dos resultados por modelo. Extrair em arquivo/ (cria
    arquivo/avaliacoes/). O que é cada pasta está em arquivo/avaliacoes/LEIA.txt.
imagens/amostras_anotacao.tar
    As imagens da avaliação do modelo de 40 épocas e as 212 imagens anotadas
    por Bruno e Vinícius. Extrair em avaliacao_diacriticos/ (cria
    avaliacao_diacriticos/amostras/).
imagens/figuras_artigo.tar
    Imagens geradas na CPU para a figura original_e_melhor do artigo.
    Extrair em arquivo/ (cria arquivo/figuras_artigo/).


3. CÓDIGO
=========

Ambiente
  requirements.txt     dependências Python (Python 3.12; o torch depende da
                       GPU: ROCm ou CUDA).
  flake.nix            ambientes Nix: nix develop .#rocm, .#cuda ou .#cpu.
  env/check_env.py     confere a GPU e aborta se o torch cair para a CPU.
  diagnostico/teste_conv_isolada.py, teste_gradiente_sintetico.py,
  teste_direcao_gradiente.py
                       comparam gradiente da GPU com o da CPU. Obrigatórios
                       antes de treinar numa GPU nova (docs/03).

Modelo de geração (DiffusionPen modificado)
  diffusionpen_mods/   arquivos que substituem os do clone do DiffusionPen.
    train.py           treino e geração. Opções novas, todas desligadas por
                       padrão: --acento_separado, --lr_acento_mult,
                       --professor, --peso_professor, --peso_acento,
                       --peso_zona, --zona vazia|vogais, --iam_originais,
                       --dataset iam_acentuado. Lista completa em docs/14.
    utils/acento_separado.py
                       o CANINE lê a palavra sem acento e o sinal entra por
                       um vetor somado na posição da letra.
    utils/iam_acentuado_dataset.py, utils/bressay_dataset.py
                       leitores das bases sintética e do BRESSAY.
  scripts/aplicar_mods.sh
                       copia diffusionpen_mods/ para o clone DiffusionPen/.
                       Rodar depois de clonar. --conferir só mostra a
                       diferença.
  scripts/treino_nuvem.sh
                       treino do modelo final, com cópia do estado a cada
                       época e retomada. Variáveis: NOME (pasta
                       ./model_<NOME>), EPOCAS (total), DESTINO (cópia de
                       segurança), ARGS (opções do train.py além das fixas:
                       professor, peso 5, lr 2e-5, lote 32, 30% de palavras
                       originais do IAM, base iam_pt_alinhado).
  scripts/treino_completo.sh
                       o mesmo treino seguido da avaliação pelo protocolo.
                       Variável a mais: PONTOS (épocas a avaliar).
  Comando do modelo final: épocas 1 a 16 no Google Colab (T4) e 17 a 40 na
  RX 9060 XT, retomando do estado salvo:
    NOME=zona_vogais EPOCAS=16 DESTINO=<pasta> \
      ARGS="--acento_separado --peso_zona 5 --zona vogais" \
      bash scripts/treino_nuvem.sh
    NOME=zona_vogais EPOCAS=30 PONTOS="21 30" \
      ARGS="--acento_separado --peso_zona 5 --zona vogais" \
      bash scripts/treino_completo.sh
    (e de novo com EPOCAS=40 PONTOS="40")
  Parâmetros registrados em
  diagnostico/resultados/acento_separado/config_zona_vogais*.jsonl.
  scripts/treinar.py + experimentos/*.json
                       treinos das fases 3 a 9 (BRESSAY e primeiras bases
                       sintéticas). Cada JSON guarda todos os parâmetros do
                       treino; --dry-run mostra o comando sem treinar.
  scripts/gerar_amostras.py
                       gera imagens de um checkpoint: --ckpt, --out,
                       --palavras, --style (o extrator de estilo do treino),
                       --estilo_de iam, --seed.
  scripts/gerar_original_e_melhor.py
                       gera na CPU as palavras da figura original_e_melhor,
                       nos dois modelos, com os escritores e a semente do
                       protocolo. --saida arquivo/figuras_artigo.

Base sintética e modelos auxiliares
  acentos_sinteticos/  desenho dos acentos e da cedilha, alinhamento das
                       letras e conferência do vocabulário.
  scripts/treinar_alinhador.py
                       treina o alinhador e o leitor. --cabeca conv
                       (alinhador) ou transformer (leitor), --saida, --epocas
                       (15 nos dois), --seed (1 no leitor).
  scripts/gerar_base_pt.py
                       gera a base portuguesa: --saida iam_pt
                       --sem_acento 12000.
  scripts/alinhar_pares.py
                       refaz o par sem acento no mesmo enquadramento:
                       --origem iam_pt --destino iam_pt_alinhado.
  scripts/mascaras_vogais.py
                       pré-calcula a zona das vogais: --base iam_pt_alinhado.
  scripts/gerar_base_acentos.py, scripts/criar_pares_iam.py
                       bases das fases 4, 8 e 9, sobre palavras reais do IAM.
  vocabulario_pt/      vocabulário português congelado, dividido em treino,
                       validação e teste (fonte FrequencyWords, OpenSubtitles
                       2018).
  base_vazia/          base vazia do controle só com palavras do IAM.

Sonda e BRESSAY (fases 1 a 3)
  sonda/smoke_test.py, sonda/sonda_diacriticos.py
                       geram as palavras da sonda com o modelo original.
                       --style (escritor), --dry-run.
  comum/palavras.py    lista de palavras da sonda.
  scripts/preparar_split.py, scripts/reduzir_split.py
                       montam os splits do BRESSAY.
  bressay_split/, bressay_split_25/, bressay_split_25_v2/
                       splits do BRESSAY (sem as imagens).

Avaliação
  scripts/avaliar_pt.py
                       protocolo fixo: 30 palavras acentuadas de validação,
                       os 30 esqueletos e 20 palavras sem acento, 20
                       escritores do iam_test, sementes 42 e 43 (3.200
                       imagens por modelo). Parâmetros: --split val,
                       --modelo rotulo=checkpoint, --bases (confere
                       vazamento), --leitor modelos/leitor_iam_transformer.pt,
                       --saida, --device.
  scripts/comparar_avaliacoes.py
                       tabela e bootstrap por palavra: --modelo
                       pasta:rotulo, --comparar A B.

Métrica de diacríticos (avaliacao_diacriticos/)
  sinais.py            E1: presença do acento (marca solta acima do corpo)
                       e da cedilha (tinta abaixo do c), letra de cada marca
                       e categoria da palavra (correto, a mais, letra
                       errada, ausente).
  avaliar_sinais.py    aplica o E1 a uma avaliação: --modelo
                       rotulo=pasta_das_imagens:pasta_do_paineis.tsv,
                       --saida.
  integridade.py       E2: CER da acentuada menos CER do esqueleto, lidos
                       com os sinais apagados; cruza com o E1 nas quatro
                       categorias. --modelo como acima, --sinais (saída do
                       avaliar_sinais.py), --saida.
  por_classe.py        separa o E1 pelo sinal pedido.
  anotacao_dupla.py, anotacao_40ep.md
                       anotação humana do modelo de 40 épocas: pacote,
                       concordância e gabarito.
  relatorio.py         gera resultados/metrica/ (concordância com a anotação,
                       resultado por modelo, figuras).
  testes_metrica.py    testes com resposta conhecida do detector de marcas
                       (única suíte automática do projeto).
  experimentos/        validações citadas no README da métrica.

Figuras
  scripts/figuras_artigo.py
                       gera as figuras do artigo em resultados/figuras/artigo/.
                       --imagens arquivo/figuras_artigo.
  diagnostico/grafico_epocas.py
                       evolução das medidas por época.


4. DADOS, LOGS E RESULTADOS NO REPOSITÓRIO
==========================================

Anotação humana (avaliacao_diacriticos/resultados/)
  planilha_anotacao_bruno.csv, planilha_anotacao_vini.csv
                       respostas de Bruno e de Vinícius.
  anotacao_40ep_desempate.csv
                       decisões nas imagens em que os dois discordaram.
  anotacao_40ep_gabarito.csv
                       gabarito final das 212 imagens.
  anotacao_40ep_chave.csv
                       palavra e imagem de cada img_NNN.

Logs e configurações dos treinos (diagnostico/resultados/acento_separado/)
  treino_*.log         log de cada treino, inclusive do modelo final
                       (treino_zona_vogais*.log).
  config_*.jsonl       todos os parâmetros, ambiente, GPU e commit de cada
                       lançamento.
  avaliacao_*_val/     resumo.md e paineis.tsv (uma linha por imagem) da
                       avaliação de cada treino pelo protocolo.
  LOG.md               diário de execução do projeto inteiro, com datas.

Resultados da métrica (avaliacao_diacriticos/resultados/)
  sinais/sinais_<modelo>.tsv
                       E1, uma linha por imagem, onze modelos.
  integridade/integridade_<modelo>.tsv
                       E2, uma linha por par acentuada e esqueleto.
  sinais/resumo_sinais.tsv, integridade/resumo_integridade.tsv
                       resumo por modelo.

Resultados e figuras
  resultados/README.md página de uma página com cada fase, seus números e
                       uma figura; figuras em resultados/figuras/.
  resultados/figuras/artigo/
                       figuras do artigo em PNG e PDF.
  resultados/metrica/  resumo da métrica: concordância com a anotação,
                       resultado por modelo e figuras; validacao.tsv com uma
                       linha por imagem anotada.
  docs/09_resultados_consolidados.md
                       todas as tabelas de resultado.


5. DOCUMENTAÇÃO
===============

  README.md            entrada do repositório.
  docs/README.md       resumo executivo e ordem de leitura.
  docs/11_como_rodar.md
                       passo a passo: ambiente, dados, treino, avaliação.
  docs/14, docs/15     o método do modelo final e os testes de cada peça.
  docs/16_roteiro_do_artigo.md
                       o que o artigo usa e onde está cada número.
  avaliacao_diacriticos/README.md
                       a métrica, as escolhas e a validação.
  ACHADOS.md           achados na ordem em que apareceram.


6. REFAZER OS NÚMEROS DO ARTIGO
===============================

Com os modelos e as imagens do Drive nos lugares indicados na seção 2. As
saídas vão para saidas/refeito/, para não sobrescrever os resultados
versionados; compare com avaliacao_diacriticos/resultados/.

  A=avaliacao_diacriticos/amostras/zona_vogais_40ep/avaliacao_zona_vogais_40ep_val
  python avaliacao_diacriticos/testes_metrica.py
  python avaliacao_diacriticos/avaliar_sinais.py --saida saidas/refeito/sinais \
      --modelo zona_vogais_40ep=$A/paineis/zona_vogais_40ep:$A
  python avaliacao_diacriticos/integridade.py --saida saidas/refeito/integridade \
      --sinais saidas/refeito/sinais --modelo zona_vogais_40ep=$A/paineis/zona_vogais_40ep:$A
  python avaliacao_diacriticos/relatorio.py      # resultados/metrica/
  python scripts/figuras_artigo.py               # resultados/figuras/artigo/

Para gerar de novo as 3.200 imagens de um modelo (cerca de 35 min por modelo
em GPU):

  python scripts/avaliar_pt.py --split val \
      --modelo zona_vogais_40ep=modelos/zona_vogais_40ep.pt \
      --bases iam_pt_alinhado --leitor modelos/leitor_iam_transformer.pt \
      --saida avaliacao_zona_vogais_40ep_val
