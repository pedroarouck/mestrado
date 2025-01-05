# Filtragem de Ruído Speckle em Imagens SAR usando Redes Neurais Convolucionais e Distâncias Estocásticas

Este repositório contém o código para meu trabalho de mestrado intitulado **"Uso de redes neurais convolucionais e distâncias estocásticas para filtrar ruído Speckle em imagens SAR"**.

## **Visão Geral**
Este projeto implementa um pipeline completo para a remoção de ruído speckle de imagens utilizando Redes Neurais Convolucionais (CNN). O ruído speckle, comum em imagens SAR (Synthetic Aperture Radar), pode degradar a qualidade das imagens. Este projeto filtra esse ruído e avalia os resultados usando métricas quantitativas e distâncias estocásticas.

O pipeline inclui:
- **Carregamento de dados:** Imagens do dataset BSDS500.
- **Adição de ruído:** Simula ruído speckle nas imagens originais.
- **Treinamento da CNN:** Filtra as imagens ruidosas.
- **Avaliação:** Mede a qualidade das imagens filtradas com métricas como PSNR, SSIM e distâncias estocásticas.
- **Visualização:** Exibe as imagens originais, ruidosas e filtradas lado a lado.

## **Fluxo de Funcionamento**
1. **Carregamento e Preparação:**
   - As imagens do dataset BSDS500 são carregadas e redimensionadas para 128x128 pixels.
   - Um subconjunto das imagens é usado para treinamento e validação (10% do dataset).
   - Ruído speckle é adicionado para criar pares `(imagem_ruidosa, imagem_limpa)`.

2. **Treinamento da CNN:**
   - Uma CNN personalizada filtra as imagens ruidosas.
   - O treinamento é realizado em lotes pequenos usando a função de perda MSELoss e o otimizador Adam.

3. **Avaliação e Métricas:**
   - Avaliação quantitativa:
     - **PSNR**: Relação sinal-ruído de pico.
     - **SSIM**: Índice de similaridade estrutural.
   - Avaliação estocástica:
     - Divergências (Kullback-Leibler, Rényi, Jensen-Shannon).
     - Distâncias (Hellinger, Bhattacharyya, entre outras).

4. **Visualização:**
   - Resultados exibidos com as imagens originais, ruidosas e filtradas lado a lado para análise qualitativa.

## **Estrutura do Projeto**

### **Arquivos**
- **`main.py`**: Arquivo principal que coordena todo o pipeline.
- **`checkpoint.pth`**: Arquivo gerado para salvar o estado do modelo durante o treinamento.
- **Dataset:** Localizado na pasta `BSDS500-master/BSDS500/data/images/train`.

### **Principais Funções e Classes**

#### **1. Carregamento e Preparação**
- **`load_images_from_folder(folder)`**: Carrega imagens em tons de cinza de uma pasta.
- **`add_speckle_noise(image, variance=0.1)`**: Adiciona ruído speckle nas imagens.
- **Classe `ImageDataset`:**
  - Gerencia os pares `(imagem_ruidosa, imagem_limpa)` para uso no treinamento e validação.

#### **2. Rede Neural Convolucional**
- **Classe `DenoisingCNN`:**
  - CNN personalizada composta por:
    - 1 camada inicial convolucional.
    - 7 camadas intermediárias convolucionais com normalização em lote.
    - 1 camada final convolucional para reconstruir a imagem.

#### **3. Treinamento**
- **`train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)`**:
  - Realiza treinamento e validação.
  - Calcula as métricas PSNR e SSIM.
  - Salva checkpoints do modelo.

#### **4. Avaliação Estocástica**
Métricas que medem a diferença entre distribuições das imagens original e filtrada:
- **Divergências:**
  - Kullback-Leibler (KL).
  - Rényi.
  - Jensen-Shannon.
- **Distâncias:**
  - Hellinger.
  - Bhattacharyya.
  - Baseadas em médias (geométrica, harmônica e triangular).

#### **5. Visualização**
- **Exibição com `matplotlib`:** Mostra imagens originais, ruidosas e filtradas para avaliação visual.

#### **6. Checkpoints**
- **`save_checkpoint` e `load_checkpoint`:** Gerenciam o salvamento e carregamento do modelo treinado.

## **Requisitos do Sistema**
- **Linguagem:** Python 3.8+
- **Bibliotecas Necessárias:**
  - `numpy`, `torch`, `cv2`, `matplotlib`, `sklearn`, `scipy`, `tqdm`, `skimage`.
- **Hardware:** GPU (é opcional, mas recomendada para acelerar o treinamento).

## **Execução do Projeto**
1. Clone o repositório e prepare o ambiente:
   ```bash
   git clone <URL_DO_REPOSITORIO>
   cd <PASTA_DO_REPOSITORIO>
   pip install -r requirements.txt
   ```
2. Certifique-se de que o dataset BSDS500 está no caminho correto (`BSDS500-master/BSDS500/data/images/train`).
3. Execute o script principal:
   ```bash
   python main.py
   ```
4. Observe as métricas e visualize os resultados ao final da execução.

## **Métricas e Avaliação**
- **Métricas de Qualidade:**
  - **PSNR:** Mede a relação entre o sinal (imagem limpa) e o ruído.
  - **SSIM:** Avalia a similaridade estrutural entre as imagens.
- **Distâncias Estocásticas:**
  - Divergências e distâncias para analisar diferenças entre distribuições de pixels.

## **Notas Importantes**
- **Tamanho do Dataset:** Apenas uma fração do dataset é utilizada para facilitar o treinamento.
- **Normalização:** As imagens são escalonadas para o intervalo [0, 1] antes do processamento.
- **Uso de GPU:** O código automaticamente detecta e utiliza GPU se disponível.

## **Contato**

Para mais informações, entre em contato:
- Nome: Pedro Henrique Salles Arouck de Souza
- Email: pedrohenrique.sales@hotmail.com
