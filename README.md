# Filtragem de Ruído Speckle em Imagens SAR usando Redes Neurais Convolucionais e Distâncias Estocásticas

Este repositório contém o código para meu trabalho de mestrado intitulado **"Uso de redes neurais convolucionais e distâncias estocásticas para filtrar ruído Speckle em imagens SAR"**.

## Visão Geral

O objetivo deste projeto é desenvolver um modelo de rede neural convolucional (CNN) capaz de reduzir o ruído speckle em imagens SAR. Adicionalmente, são utilizadas diversas distâncias estocásticas para quantificar a similaridade entre as imagens originais e as imagens filtradas.

### Fluxo de Funcionamento

1. **Carregamento e Preparação dos Dados**
    - As imagens são carregadas, redimensionadas para 128x128 pixels, e normalizadas.
    - Ruído speckle é adicionado para criar versões ruidosas das imagens.

2. **Divisão do Dataset**
    - As imagens são divididas em conjuntos de treinamento e validação.
    - DataLoaders são criados para facilitar o treinamento em mini-batches.

3. **Definição e Treinamento do Modelo**
    - A arquitetura da rede neural convolucional é definida.
    - A função de perda e o otimizador são inicializados.
    - O modelo é treinado usando os dados de treinamento e validado periodicamente.

4. **Avaliação e Visualização dos Resultados**
    - Após o treinamento, o modelo é avaliado em uma amostra do conjunto de validação.
    - Diversas distâncias estocásticas são calculadas para avaliar a qualidade da remoção de ruído.
    - As imagens original, ruidosa e denoised são exibidas lado a lado para uma fácil comparação visual.

## Estrutura do Projeto

```
.
├── data
│   ├── train
│   │   └── (imagens de treinamento)
│   └── val
│       └── (imagens de validação)
├── models
│   ├── model_checkpoint.pth
├── results
│   ├── output_images
│   └── logs
├── scripts
│   ├── train.py
│   └── evaluate.py
├── README.md
└── requirements.txt
```

## Instruções de Uso

### Pré-requisitos

- Python 3.7+
- PyTorch
- NumPy
- Matplotlib

Instale as dependências executando:
```bash
pip install -r requirements.txt
```

### Treinamento do Modelo

Para treinar o modelo, execute o script `train.py`:
```bash
python scripts/train.py
```

### Avaliação do Modelo

Para avaliar o modelo, execute o script `evaluate.py`:
```bash
python scripts/evaluate.py
```

## Funcionamento Detalhado

### Carregamento e Preparação dos Dados

O script carrega as imagens do dataset, redimensiona para 128x128 pixels e adiciona ruído speckle. As imagens são então normalizadas e divididas em conjuntos de treinamento e validação.

### Definição do Modelo

A rede neural convolucional é definida com várias camadas convolucionais, normalização em batch e ativações ReLU. A arquitetura foi projetada para reduzir o ruído speckle nas imagens SAR.

### Treinamento do Modelo

O modelo é treinado usando uma função de perda composta por MSE (Mean Squared Error) e uma ou mais distâncias estocásticas, permitindo que a rede aprenda a remover o ruído de maneira eficaz.

### Avaliação e Visualização

Após o treinamento, o modelo é avaliado em uma amostra do conjunto de validação. As distâncias estocásticas são calculadas para quantificar a similaridade entre a imagem original e a imagem denoised. As imagens são visualizadas lado a lado para uma comparação qualitativa.

## Todo

Pretendo usar as distâncias estocásticas como retroalimentação para aumentar a eficiência do script, integrando-as ao processo de treinamento da rede neural.

## Contato

Para mais informações, entre em contato:
- Nome: Pedro Henrique Salles Arouck de Souza
- Email: pedrohenrique.sales@hotmail.com
