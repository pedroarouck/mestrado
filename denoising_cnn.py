import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
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


# Função para carregar imagens do BSDS500
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images

# Adicionar ruído speckle
def add_speckle_noise(image, variance=0.1):
    row, col = image.shape
    gaussian = np.random.normal(0, variance ** 0.5, (row, col))
    noisy = image + image * gaussian
    return noisy

# Dataset Customizado
class ImageDataset(Dataset):
    def __init__(self, noisy_images, clean_images):
        self.noisy_images = noisy_images
        self.clean_images = clean_images

    def __len__(self):
        return len(self.noisy_images)

    def __getitem__(self, idx):
        noisy = torch.tensor(self.noisy_images[idx], dtype=torch.float32).permute(2, 0, 1)
        clean = torch.tensor(self.clean_images[idx], dtype=torch.float32).permute(2, 0, 1)
        return noisy, clean

# Modelo CNN
class DenoisingCNN(nn.Module):
    def __init__(self):
        super(DenoisingCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Reduza o número de filtros
        self.relu = nn.ReLU()
        self.conv_layers = nn.Sequential(
            *[nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU()) for _ in range(7)]  # Reduza o número de camadas
        )
        self.conv2 = nn.Conv2d(32, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv_layers(x)
        x = self.conv2(x)
        return x

# Função de Treinamento
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        # Etapa de treinamento
        for noisy, clean in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False):
            noisy, clean = noisy.to(device), clean.to(device)

            optimizer.zero_grad()
            outputs = model(noisy)
            loss = criterion(outputs, clean)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * noisy.size(0)

        train_loss /= len(train_loader.dataset)

        # Etapa de validação
        model.eval()
        val_loss = 0
        psnr_values = []
        ssim_values = []

        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                outputs = model(noisy)
                loss = criterion(outputs, clean)
                val_loss += loss.item() * noisy.size(0)

                # Cálculo de PSNR e SSIM
                clean_np = clean.squeeze().cpu().numpy()
                outputs_np = outputs.squeeze().cpu().numpy()
               
                for c_img, o_img in zip(clean_np, outputs_np):
                    psnr_values.append(psnr(c_img, o_img, data_range=1.0))  # Normalização já realizada
                    ssim_values.append(ssim(c_img, o_img, data_range=1.0))

        val_loss /= len(val_loader.dataset)
        avg_psnr = np.mean(psnr_values)
        avg_ssim = np.mean(ssim_values)
       
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
              f'PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}')

        # Salvar checkpoint
        save_checkpoint(model, optimizer, epoch, 'checkpoint.pth')

# Funções para Distâncias Estocásticas
def kullback_leibler_divergence(p, q):
    return entropy(p.flatten(), q.flatten())

def renyi_divergence(p, q, alpha=0.5):
    p = np.asarray(p).flatten()
    q = np.asarray(q).flatten()
    return (1 / (alpha - 1)) * np.log(np.sum(p**alpha * q**(1 - alpha)))

def hellinger_distance(p, q):
    return euclidean(np.sqrt(p), np.sqrt(q)) / np.sqrt(2)

def bhattacharyya_distance(p, q):
    return -np.log(np.sum(np.sqrt(p * q)))

def jensen_shannon_divergence(p, q):
    return jensenshannon(p, q)

def arithmetic_geometric_distance(p, q):
    p = np.asarray(p).flatten()
    q = np.asarray(q).flatten()
    arithmetic_mean = (p + q) / 2
    geometric_mean = np.sqrt(p * q)
    return np.sum(arithmetic_mean - geometric_mean)

def triangular_distance(p, q):
    p = np.asarray(p).flatten()
    q = np.asarray(q).flatten()
    return np.sum((p - q)**2 / (p + q))

def harmonic_mean_distance(p, q):
    p = np.asarray(p).flatten()
    q = np.asarray(q).flatten()
    return np.sum((2 * p * q) / (p + q))

# Função para salvar o checkpoint
def save_checkpoint(model, optimizer, epoch, filepath):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(state, filepath)

# Função para carregar o checkpoint
def load_checkpoint(filepath, model, optimizer):
    state = torch.load(filepath)
    model.load_state_dict(state['model_state_dict'])
    optimizer.load_state_dict(state['optimizer_state_dict'])
    return state['epoch']


# Função Principal
def main():
    # Carregar imagens
    train_folder = 'BSDS500-master/BSDS500/data/images/train'
    images = load_images_from_folder(train_folder)
    image_size = (128, 128)  # Reduza o tamanho das imagens
    images_resized = [cv2.resize(img, image_size) for img in images]

    # Selecionar uma fração das imagens para treinamento
    fraction = 0.1  # Use 10% do dataset original, por exemplo
    num_images = int(len(images_resized) * fraction)
    images_resized = images_resized[:num_images]
    noisy_images = [add_speckle_noise(img, variance=0.5) for img in images_resized]

    # Normalizar imagens
    images_resized = np.array(images_resized) / 255.0
    noisy_images = np.array(noisy_images) / 255.0
    images_resized = images_resized[..., np.newaxis]
    noisy_images = noisy_images[..., np.newaxis]

    # Dividir em treinamento e validação
    X_train, X_val, y_train, y_val = train_test_split(noisy_images, images_resized, test_size=0.2, random_state=42)

    # Criar DataLoader
    train_dataset = ImageDataset(X_train, y_train)
    val_dataset = ImageDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)  
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # Preparar o modelo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DenoisingCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Carregar checkpoint se existir
    start_epoch = 0
    checkpoint_path = 'checkpoint.pth'
    if os.path.exists(checkpoint_path):
        start_epoch = load_checkpoint(checkpoint_path, model, optimizer)

    # Treinar o modelo
    num_epochs = 10  # Reduza o número de épocas
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs - start_epoch, device)

    # Avaliação com distâncias estocásticas
    model.eval()
    random_index = random.randint(0, len(val_dataset) - 1)
    noisy_sample, original_sample = val_dataset[0]
    noisy_sample = noisy_sample.unsqueeze(0).to(device)
    original_sample = original_sample.squeeze(0).cpu().numpy()

    with torch.no_grad():
        denoised_image = model(noisy_sample).squeeze(0).cpu().numpy()

    p = (original_sample + 1e-10).flatten()
    q = (denoised_image + 1e-10).flatten()

    kl_divergence = kullback_leibler_divergence(p, q)
    renyi = renyi_divergence(p, q)
    hellinger = hellinger_distance(p, q)
    bhattacharyya = bhattacharyya_distance(p, q)
    jensen_shannon = jensen_shannon_divergence(p, q)
    
    arithmetic_geometric = arithmetic_geometric_distance(p, q)
    triangular = triangular_distance(p, q)
    harmonic_mean = harmonic_mean_distance(p, q) 

    psnr_value = psnr(p, q, data_range=1.0)
    ssim_value = ssim(p, q, data_range=1.0)

    print(f"KL Divergence: {kl_divergence}")
    print(f"Rényi Divergence: {renyi}")
    print(f"Hellinger Distance: {hellinger}")
    print(f"Bhattacharyya Distance: {bhattacharyya}")
    print(f"Jensen-Shannon Divergence: {jensen_shannon}")
    
    print(f"Arithmetic-Geometric Distance: {arithmetic_geometric}")
    print(f"Triangular Distance: {triangular}")
    print(f"Harmonic Mean Distance: {harmonic_mean}") 

    print(f"PSNR: {psnr_value:.4f}, SSIM: {ssim_value:.4f}")

    # Visualizar resultados
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
