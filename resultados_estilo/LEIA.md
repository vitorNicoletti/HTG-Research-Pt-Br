# Trocar as imagens de referencia de estilo: BRESSAY vs IAM

Mesmo checkpoint (`model_bressay_longo/models/ema_bloco_19ep.pt`, deriva 3,59%
nos pesos treinaveis), mesmo extrator (`iam_style_diffusionpen.pth`, o do
treino), mesma semente (42), mesmas palavras. **A unica variavel e a origem
dos 5 recortes que alimentam o extrator de estilo.**

Gerado na CPU: a RX 6600 XT desta maquina colapsa a amostragem mesmo em lote 1
(ver diagnostico/README.md).

    python scripts/gerar_amostras.py --ckpt <ckpt> \
        --style DiffusionPen/style_models/iam_style_diffusionpen.pth \
        --out ./est_iam_19 --estilo_de iam --device cpu --styles 3 --seed 42 \
        --palavras coração português aproximadamente não

Nenhum painel colapsou dos dois lados (std entre 0,161 e 0,286), entao a
comparacao e valida imagem a imagem.

Ver `comparacao_estilo_bressay_vs_iam.png` e `ampliacoes.png`.
