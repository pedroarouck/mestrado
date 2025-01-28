import os
import torch.nn.functional as F
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.stats import entropy
from scipy.spatial.distance import euclidean
import random
from tqdm import tqdm
from scipy.spatial.distance import jensenshannon
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from scipy.special import entr

def load_images_from_folder(folder):
    """
    Carrega imagens em escala de cinza de uma pasta e valida erros durante o processo.

    Args:
        folder (str): Caminho para a pasta contendo as imagens.

    Returns:
        list: Lista de arrays numpy representando as imagens carregadas.
    """
    images = []
    if not os.path.exists(folder):
        raise FileNotFoundError(f"A pasta '{folder}' não existe.")
    
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"O caminho '{folder}' não é uma pasta válida.")

    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)

        # Verificar se é um arquivo válido
        if not os.path.isfile(filepath):
            print(f"[AVISO] '{filename}' não é um arquivo válido. Ignorando...")
            continue
        
        # Verificar se é uma imagem válida
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            print(f"[AVISO] '{filename}' não é uma imagem suportada. Ignorando...")
            continue
        
        try:
            # Tentar carregar a imagem em escala de cinza
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[ERRO] Falha ao carregar '{filename}'.")
                continue
            images.append(img)
        except Exception as e:
            print(f"[ERRO] Ocorreu um erro ao carregar '{filename}': {e}")

    if not images:
        raise ValueError("Nenhuma imagem válida foi carregada da pasta especificada.")

    print(f"[INFO] Total de {len(images)} imagens carregadas com sucesso.")
    return images


def add_speckle_noise(image, L=1):
    """
    Adiciona ruído speckle à imagem usando a distribuição Gamma.

    Args:
        image (numpy.ndarray): Imagem de entrada normalizada (valores entre 0 e 1).
        L (int): Número de looks (determina a intensidade do ruído). L >= 1.

    Returns:
        numpy.ndarray: Imagem com ruído speckle adicionado.
    """
    if L < 1:
        raise ValueError("O número de looks (L) deve ser maior ou igual a 1.")
    
    # Gerar o ruído speckle usando a distribuição Gamma
    row, col = image.shape
    gamma_noise = np.random.gamma(L, 1.0 / L, (row, col))  # Distribuição Gamma
    
    # Adicionar o ruído multiplicativo
    noisy_image = image * gamma_noise
    return noisy_image

class ImageDataset(Dataset):
    def __init__(self, noisy_images, clean_images, apply_augmentations=False):
        """
        Dataset customizado para pares de imagens ruidosas e limpas.

        Args:
            noisy_images (list or numpy.ndarray): Imagens com ruído.
            clean_images (list or numpy.ndarray): Imagens limpas correspondentes.
            apply_augmentations (bool): Se True, aplica transformações nas imagens.
        """
        # Validar que ambos os conjuntos têm o mesmo tamanho
        if len(noisy_images) != len(clean_images):
            raise ValueError("Os conjuntos 'noisy_images' e 'clean_images' devem ter o mesmo tamanho.")

        self.noisy_images = noisy_images
        self.clean_images = clean_images

        # Configurar transformações (opcionais)
        self.transform = None
        if apply_augmentations:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),  # Flip horizontal
                transforms.RandomVerticalFlip(),    # Flip vertical
                transforms.RandomRotation(degrees=20),  # Rotação aleatória de até 20 graus
            ])

    def __len__(self):
        """
        Retorna o número de amostras no dataset.
        """
        return len(self.noisy_images)

    def __getitem__(self, idx):
        """
        Retorna um par de imagens (ruidosa, limpa) no formato tensor.

        Args:
            idx (int): Índice do par de imagens.

        Returns:
            tuple: (imagem_ruidosa, imagem_limpa) no formato tensor.
        """
        noisy = self.noisy_images[idx]
        clean = self.clean_images[idx]

        # Aplicar augmentations se habilitado
        if self.transform:
            noisy = self.apply_transform(noisy)
            clean = self.apply_transform(clean)

        # Converter para tensores e ajustar a ordem dos canais
        noisy = torch.tensor(noisy, dtype=torch.float32).permute(2, 0, 1)
        clean = torch.tensor(clean, dtype=torch.float32).permute(2, 0, 1)

        return noisy, clean

    def apply_transform(self, image):
        """
        Aplica as transformações a uma imagem específica.

        Args:
            image (numpy.ndarray): Imagem a ser transformada.

        Returns:
            numpy.ndarray: Imagem transformada.
        """
        # A transformação espera uma imagem no formato PIL, então convertemos
        image_pil = transforms.ToPILImage()(image)
        transformed_image = self.transform(image_pil)
        return transforms.ToTensor()(transformed_image).permute(1, 2, 0).numpy()

class DenoisingCNN(nn.Module):
    def __init__(self, in_channels=1, dropout_rate=0.5):
        """
        Modelo CNN para redução de ruído.

        Args:
            in_channels (int): Número de canais de entrada (1 para imagens em escala de cinza, 3 para RGB, etc.).
            dropout_rate (float): Taxa de dropout para regularização.
        """
        super(DenoisingCNN, self).__init__()

        # Primeira camada convolucional
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        # Camadas intermediárias com Dropout e BatchNorm
        self.conv_layers = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate)  # Dropout para regularização
            ) for _ in range(7)]  # 7 camadas intermediárias
        )

        # Camada final para reconstrução
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # Garante saídas não negativas
)

    def forward(self, x):
        """
        Define o fluxo de dados no modelo.

        Args:
            x (torch.Tensor): Tensor de entrada no formato (batch_size, in_channels, altura, largura).

        Returns:
            torch.Tensor: Tensor de saída no formato (batch_size, in_channels, altura, largura).
        """
        x = self.relu(self.conv1(x))  # Primeira convolução com ReLU
        x = self.conv_layers(x)       # Passagem pelas camadas intermediárias
        x = self.conv2(x)             # Reconstrução pela camada final
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, log_file="training_log.txt"):
    """
    Função para treinar e validar o modelo de redução de ruído.

    Args:
        model: Modelo a ser treinado.
        train_loader: DataLoader para o conjunto de treinamento.
        val_loader: DataLoader para o conjunto de validação.
        criterion: Função de perda.
        optimizer: Otimizador usado para atualizar os pesos.
        num_epochs: Número total de épocas de treinamento.
        device: Dispositivo a ser usado (CPU ou GPU).
        log_file: Nome do arquivo para salvar os logs de treinamento.

    Returns:
        None
    """
    # Abrir o arquivo de log para gravação
    with open(log_file, "w") as log:
        log.write("Epoch,Train_Loss,Val_Loss,PSNR,SSIM\n")  # Cabeçalho do log

        for epoch in range(num_epochs):
            model.train()  # Colocar o modelo em modo de treinamento
            train_loss = 0

            # Etapa de treinamento
            for noisy, clean in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False):
                noisy, clean = noisy.to(device), clean.to(device)

                # Zerar os gradientes acumulados
                optimizer.zero_grad()

                # Forward pass (obter previsões)
                outputs = model(noisy)

                # Calcular as distribuições p e q para a perda composta
                p = clean.flatten().cpu().numpy() + 1e-10  # Evitar divisão por zero
                q = outputs.detach().flatten().cpu().numpy() + 1e-10
                # Adicione antes da linha que calcula a loss no loop de treinamento
                print(f"[DEBUG - Train] p (min: {p.min()}, max: {p.max()}), q (min: {q.min()}, max: {q.max()})")
                if np.any(p < 0) or np.any(q < 0):
                    raise ValueError("As distribuições p ou q no treinamento contêm valores negativos.")

                # Calcular a perda composta
                loss = criterion(outputs, clean, p, q)

                # Backward pass (propagação do gradiente)
                loss.backward()
                optimizer.step()  # Atualizar os pesos

                # Acumular a perda total
                train_loss += loss.item() * noisy.size(0)

            # Calcular a perda média no conjunto de treinamento
            train_loss /= len(train_loader.dataset)

            # Etapa de validação
            model.eval()  # Colocar o modelo em modo de avaliação
            val_loss = 0
            psnr_values = []
            ssim_values = []

            with torch.no_grad():  # Desativar o cálculo de gradientes para economizar memória
                for noisy, clean in val_loader:
                    noisy, clean = noisy.to(device), clean.to(device)

                    # Forward pass para validação
                    outputs = model(noisy)

                    # Calcular as distribuições p e q
                    p = clean.flatten().cpu().numpy() + 1e-10
                    q = outputs.flatten().cpu().numpy() + 1e-10

                    # Calcular a perda composta
                    loss = criterion(outputs, clean, p, q)
                    val_loss += loss.item() * noisy.size(0)

                    # Calcular PSNR e SSIM
                    clean_np = clean.squeeze().cpu().numpy()
                    outputs_np = outputs.squeeze().cpu().numpy()
                    for c_img, o_img in zip(clean_np, outputs_np):
                        psnr_values.append(psnr(c_img, o_img, data_range=1.0))
                        ssim_values.append(ssim(c_img, o_img, data_range=1.0))

            # Calcular a perda média e as métricas no conjunto de validação
            val_loss /= len(val_loader.dataset)
            avg_psnr = np.mean(psnr_values)
            avg_ssim = np.mean(ssim_values)

            # Exibir os resultados da época
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                  f'PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}')

            # Gravar os resultados no arquivo de log
            log.write(f"{epoch+1},{train_loss:.4f},{val_loss:.4f},{avg_psnr:.4f},{avg_ssim:.4f}\n")

            # Salvar um checkpoint do modelo
            save_checkpoint(model, optimizer, epoch, 'checkpoint.pth')

def normalize(p, q, smoothing=1e-10):
    """
    Normaliza duas distribuições de probabilidade, p e q, garantindo que a soma de cada uma seja 1.
    Aplica validações para entradas válidas e tratamento de distribuições nulas.

    Args:
        p (numpy.ndarray): Primeira distribuição de probabilidade.
        q (numpy.ndarray): Segunda distribuição de probabilidade.
        smoothing (float): Constante para suavização, evita divisão por zero e zeros nas distribuições.

    Returns:
        tuple: As distribuições normalizadas (p, q).
    """
    # Converter para arrays unidimensionais
    p = np.asarray(p).flatten()
    q = np.asarray(q).flatten()

    # Verificar se as distribuições contêm valores negativos
    if np.any(p < 0) or np.any(q < 0):
        raise ValueError("As distribuições não podem conter valores negativos.")

    # Calcular as somas de p e q
    p_sum = np.sum(p)
    q_sum = np.sum(q)

    # Tratamento para distribuições completamente nulas
    if p_sum == 0:
        print("[AVISO] A distribuição 'p' é nula. Aplicando suavização.")
        p = np.ones_like(p) * smoothing
        p_sum = np.sum(p)

    if q_sum == 0:
        print("[AVISO] A distribuição 'q' é nula. Aplicando suavização.")
        q = np.ones_like(q) * smoothing
        q_sum = np.sum(q)

    # Normalizar as distribuições
    p /= p_sum
    q /= q_sum

    return p, q
import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import euclidean, jensenshannon

# Função auxiliar para preparar distribuições
def prepare_distributions(p, q):
    """
    Normaliza e prepara duas distribuições para cálculos de métricas.

    Args:
        p (array-like): Primeira distribuição.
        q (array-like): Segunda distribuição.

    Returns:
        tuple: Distribuições normalizadas e suavizadas.
    """
    if len(p) != len(q):
        raise ValueError("As distribuições p e q devem ter o mesmo tamanho.")
    if np.any(p < 0) or np.any(q < 0):
        raise ValueError("As distribuições não podem conter valores negativos.")
    
    # Normalizar as distribuições
    p, q = normalize(p, q)
    
    # Garantir que não haja zeros (suavização)
    p = np.clip(p, a_min=1e-10, a_max=None)
    q = np.clip(q, a_min=1e-10, a_max=None)
    
    return p, q

# Função para calcular a Divergência de Kullback-Leibler
def kullback_leibler_divergence(p, q):
    """
    Calcula a Divergência de Kullback-Leibler entre duas distribuições.

    Args:
        p (array-like): Primeira distribuição de probabilidade.
        q (array-like): Segunda distribuição de probabilidade.

    Returns:
        float: Divergência de Kullback-Leibler.
    """
    p, q = prepare_distributions(p, q)
    return np.sum(p * np.log(p / q))

# Função para calcular a Divergência de Rényi
def renyi_divergence(p, q, alpha=0.5):
    """
    Calcula a Divergência de Rényi entre duas distribuições.

    Args:
        p (array-like): Primeira distribuição.
        q (array-like): Segunda distribuição.
        alpha (float): Parâmetro de sensibilidade (>0 e !=1).

    Returns:
        float: Divergência de Rényi.
    """
    if alpha <= 0 or alpha == 1:
        raise ValueError("Alpha deve ser maior que 0 e diferente de 1.")
    p, q = prepare_distributions(p, q)
    return (1 / (alpha - 1)) * np.log(np.sum(p**alpha * q**(1 - alpha)))

# Função para calcular a Distância de Hellinger
def hellinger_distance(p, q):
    """
    Calcula a Distância de Hellinger entre duas distribuições.

    Args:
        p (array-like): Primeira distribuição.
        q (array-like): Segunda distribuição.

    Returns:
        float: Distância de Hellinger.
    """
    p, q = prepare_distributions(p, q)
    return euclidean(np.sqrt(p), np.sqrt(q)) / np.sqrt(2)

# Função para calcular a Distância de Bhattacharyya
def bhattacharyya_distance(p, q):
    """
    Calcula a Distância de Bhattacharyya entre duas distribuições.

    Args:
        p (array-like): Primeira distribuição.
        q (array-like): Segunda distribuição.

    Returns:
        float: Distância de Bhattacharyya.
    """
    p, q = prepare_distributions(p, q)
    return -np.log(np.sum(np.sqrt(p * q)))

# Função para calcular a Divergência de Jensen-Shannon
def jensen_shannon_divergence(p, q):
    """
    Calcula a Divergência de Jensen-Shannon entre duas distribuições.

    Args:
        p (array-like): Primeira distribuição.
        q (array-like): Segunda distribuição.

    Returns:
        float: Divergência de Jensen-Shannon.
    """
    p, q = prepare_distributions(p, q)
    return jensenshannon(p, q)

# Função para calcular a Distância Aritmético-Geometrica
def arithmetic_geometric_distance(p, q):
    """
    Calcula a Distância Aritmético-Geometrica entre duas distribuições.

    Args:
        p (array-like): Primeira distribuição.
        q (array-like): Segunda distribuição.

    Returns:
        float: Distância Aritmético-Geometrica.
    """
    p, q = prepare_distributions(p, q)
    arithmetic_mean = (p + q) / 2
    geometric_mean = np.sqrt(p * q)
    return np.sum(np.abs(arithmetic_mean - geometric_mean))

# Função para calcular a Distância Triangular
def triangular_distance(p, q):
    """
    Calcula a Distância Triangular entre duas distribuições.

    Args:
        p (array-like): Primeira distribuição.
        q (array-like): Segunda distribuição.

    Returns:
        float: Distância Triangular.
    """
    p, q = prepare_distributions(p, q)
    return np.sum((p - q)**2 / (p + q))

# Função para calcular a Distância Média Harmônica
def harmonic_mean_distance(p, q):
    """
    Calcula a Distância Média Harmônica entre duas distribuições.

    Args:
        p (array-like): Primeira distribuição.
        q (array-like): Segunda distribuição.

    Returns:
        float: Distância Média Harmônica.
    """
    p, q = prepare_distributions(p, q)
    return np.sum((2 * p * q) / (p + q))

def save_checkpoint(model, optimizer, epoch, filepath):
    """
    Salva o estado atual do modelo, do otimizador e das configurações de treinamento em um arquivo de checkpoint.

    Args:
        model: O modelo a ser salvo.
        optimizer: O otimizador usado no treinamento.
        epoch: O número da época atual.
        filepath: O caminho do arquivo onde o checkpoint será salvo.

    Returns:
        None
    """
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr': optimizer.param_groups[0]['lr'],  # Salvar taxa de aprendizado
    }
    torch.save(state, filepath)
    print(f"Checkpoint salvo em: {filepath}")

def load_checkpoint(filepath, model, optimizer=None):
    """
    Carrega o estado salvo do modelo e, opcionalmente, do otimizador a partir de um arquivo de checkpoint.

    Args:
        filepath: O caminho do arquivo de checkpoint.
        model: O modelo para o qual o estado será carregado.
        optimizer: (Opcional) O otimizador para o qual o estado será carregado.

    Returns:
        int: O número da época em que o treinamento foi salvo.
        dict: Configurações adicionais, como taxa de aprendizado.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"O arquivo {filepath} não foi encontrado.")
    
    # Carregar o estado salvo
    state = torch.load(filepath)

    # Restaurar o modelo
    model.load_state_dict(state['model_state_dict'])
    print(f"Modelo carregado de: {filepath}")

    # Restaurar o otimizador, se fornecido
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer_state_dict'])
        print("Estado do otimizador restaurado.")

    # Retornar a época e as configurações adicionais
    return state['epoch'], {'lr': state.get('lr', None)}



class CombinedLoss(nn.Module):
    """
    Função de perda composta com pesos aprendíveis para MSE, KL, Hellinger e Jensen-Shannon.
    """

    def __init__(self):
        super(CombinedLoss, self).__init__()
        # Pesos aprendíveis para as métricas
        self.alpha = nn.Parameter(torch.tensor(0.5, requires_grad=True))  # Peso da Divergência KL
        self.beta = nn.Parameter(torch.tensor(0.3, requires_grad=True))   # Peso da Distância de Hellinger
        self.gamma = nn.Parameter(torch.tensor(0.2, requires_grad=True))  # Peso da Divergência de Jensen-Shannon

    def forward(self, output, target, p, q):
        """
        Calcula a perda composta combinando MSE, KL, Hellinger e Jensen-Shannon.

        Args:
            output (torch.Tensor): Saída predita pelo modelo.
            target (torch.Tensor): Valores reais.
            p (numpy.ndarray): Distribuição verdadeira.
            q (numpy.ndarray): Distribuição predita.

        Returns:
            torch.Tensor: Valor da perda composta.
        """
        # MSE Loss
        mse_loss = F.mse_loss(output, target)

        # Métricas de distribuições
        kl_loss = kullback_leibler_divergence(p, q)
        hellinger_loss = hellinger_distance(p, q)
        js_loss = jensen_shannon_divergence(p, q)

        # Combinação das perdas com pesos aprendíveis
        total_loss = mse_loss + self.alpha * kl_loss + self.beta * hellinger_loss + self.gamma * js_loss

        # Logs detalhados
        print(f"MSE Loss: {mse_loss.item():.6f}, KL Loss: {kl_loss:.6f}, "
              f"Hellinger Loss: {hellinger_loss:.6f}, JS Loss: {js_loss:.6f}")
        print(f"Pesos: Alpha: {self.alpha.item():.3f}, Beta: {self.beta.item():.3f}, Gamma: {self.gamma.item():.3f}")

        return total_loss


from torch.cuda.amp import GradScaler, autocast

from torch.cuda.amp import GradScaler, autocast

def main():
    """
    Função principal para executar o pipeline completo de treinamento e validação com Mixed Precision Training.
    """
    # Caminho do dataset
    train_folder = 'BSDS500-master/BSDS500/data/images/train'

    # Carregar imagens
    print("Carregando imagens...")
    images = load_images_from_folder(train_folder)
    image_size = (128, 128)
    images_resized = [cv2.resize(img, image_size) for img in images]

    # Selecionar uma fração das imagens para treinamento
    fraction = 0.1  # Usar 10% do dataset
    num_images = int(len(images_resized) * fraction)
    images_resized = images_resized[:num_images]
    noisy_images = [add_speckle_noise(img, L=5) for img in images_resized]

    # Normalizar imagens
    images_resized = np.array(images_resized) / 255.0
    noisy_images = np.array(noisy_images) / 255.0
    images_resized = images_resized[..., np.newaxis]
    noisy_images = noisy_images[..., np.newaxis]

    # Dividir em treinamento e validação
    print("Dividindo os dados...")
    X_train, X_val, y_train, y_val = train_test_split(noisy_images, images_resized, test_size=0.2, random_state=42)

    # Criar DataLoaders
    print("Criando DataLoaders...")
    train_dataset = ImageDataset(X_train, y_train)
    val_dataset = ImageDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # Configuração do modelo, perda e otimizador
    print("Configurando o modelo e otimizador...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DenoisingCNN(in_channels=1).to(device)
    criterion = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Mixed Precision Training: Inicializar o GradScaler
    scaler = GradScaler()

    # Carregar checkpoint, se existir
    print("Carregando checkpoint, se disponível...")
    start_epoch = 0
    checkpoint_path = 'checkpoint.pth'
    if os.path.exists(checkpoint_path):
        start_epoch, config = load_checkpoint(checkpoint_path, model, optimizer)
        print(f"Treinamento retomado na época {start_epoch + 1}, com taxa de aprendizado: {config['lr']}")

    # Treinar o modelo
    num_epochs = 10
    print("Iniciando treinamento com Mixed Precision...")
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0

        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)

            # Forward e backward pass com mixed precision
            with autocast():
                outputs = model(noisy)
                p = clean.flatten().cpu().numpy() + 1e-10
                q = outputs.detach().flatten().cpu().numpy() + 1e-10
                loss = criterion(outputs, clean, p, q)

            # Backpropagation com GradScaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            train_loss += loss.item() * noisy.size(0)

        train_loss /= len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}")

        # Salvar checkpoint
        save_checkpoint(model, optimizer, epoch, checkpoint_path)

    # Avaliação e visualização final (mesma lógica anterior)
    print("Avaliação final...")
    model.eval()
    random_index = random.randint(0, len(val_dataset) - 1)
    noisy_sample, original_sample = val_dataset[random_index]
    noisy_sample = noisy_sample.unsqueeze(0).to(device)
    original_sample = original_sample.squeeze(0).cpu().numpy()

    with torch.no_grad():
        denoised_image = model(noisy_sample).squeeze(0).cpu().numpy()

    # Calcular métricas
    p = (original_sample + 1e-10).flatten()
    q = (denoised_image + 1e-10).flatten()
    kl_divergence = kullback_leibler_divergence(p, q)
    renyi = renyi_divergence(p, q)
    hellinger = hellinger_distance(p, q)
    bhattacharyya = bhattacharyya_distance(p, q)
    jensen_shannon = jensen_shannon_divergence(p, q)

    # Calcular PSNR e SSIM
    psnr_value = psnr(p, q, data_range=1.0)
    ssim_value = ssim(p, q, data_range=1.0)

    # Exibir resultados
    print(f"KL Divergence: {kl_divergence}")
    print(f"Rényi Divergence: {renyi}")
    print(f"Hellinger Distance: {hellinger}")
    print(f"Bhattacharyya Distance: {bhattacharyya}")
    print(f"Jensen-Shannon Divergence: {jensen_shannon}")
    print(f"PSNR: {psnr_value:.4f}")
    print(f"SSIM: {ssim_value:.4f}")

    # Visualizar imagens
    print("Visualizando resultados...")
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(original_sample.squeeze(), cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Noisy Image')
    plt.imshow(noisy_sample.squeeze().cpu().numpy(), cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Denoised Image')
    plt.imshow(denoised_image.squeeze(), cmap='gray')
    plt.axis('off')

    plt.show()

if __name__ == '__main__':
    main()
