import copy
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import torch
import torch.nn as nn
import torch.optim as optim
from plots import generate_all_plots

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Configurando semente aleatória para reprodutibilidade científica
torch.manual_seed(42)
np.random.seed(42)
if hasattr(torch, "set_num_threads"):
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

# =====================================================================
# 1. PRÉ-PROCESSAMENTO E ENGENHARIA DE FEATURES
# =====================================================================

def preprocess_data(filepath):
    # Carregar o dataset
    df = pd.read_csv(filepath)
    
    # Criar a variável alvo: Delta_GPA = Post_Semester_GPA - Pre_Semester_GPA
    # Isso mede a evolução real obtida pelo estudante ao longo do semestre
    df['Delta_GPA'] = df['Post_Semester_GPA'] - df['Pre_Semester_GPA']
    target_mean = df['Delta_GPA'].mean()
    target_std = df['Delta_GPA'].std()
    df['Delta_GPA_norm'] = (df['Delta_GPA'] - target_mean) / (target_std if target_std != 0 else 1.0)
    
    # Variáveis preditoras (Features)
    categorical_features = ['Major_Category', 'Year_of_Study', 'Primary_Use_Case', 'Prompt_Engineering_Skill']
    numeric_features = ['Weekly_GenAI_Hours', 'Traditional_Study_Hours', 'Tool_Diversity', 
                        'Perceived_AI_Dependency', 'Anxiety_Level_During_Exams', 'Pre_Semester_GPA']
    boolean_features = ['Paid_Subscription']
    
    # Tratando booleanos
    for col in boolean_features:
        df[col] = df[col].astype(int)
    
    # Definindo os transformadores de colunas (Pipeline de Pré-processamento)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough' # Mantém colunas booleanas inalteradas
    )
    
    # Separando Features (X) e Alvo (y)
    X = df[numeric_features + categorical_features + boolean_features]
    y = df[['Delta_GPA_norm']].values.astype(np.float32)
    
    # Ajustando o preprocessor aos dados
    X_processed = preprocessor.fit_transform(X).astype(np.float32)
    
    return X_processed, y, preprocessor, X.columns.tolist(), target_mean, target_std

# =====================================================================
# 2. DATASET CUSTOMIZADO DO PYTORCH
# =====================================================================

class StudentDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# =====================================================================
# 3. ARQUITETURA DA REDE NEURAL (MLP)
# =====================================================================

class SimulaRegressor(nn.Module):
    def __init__(self, input_dim):
        super(SimulaRegressor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.15),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        return self.network(x)

# =====================================================================
# 4. CICLO DE TREINAMENTO
# =====================================================================

def train_model(model, train_loader, val_loader, epochs=12, lr=0.0008, patience=5, min_delta=1e-4, device=None):
    criterion = nn.SmoothL1Loss(beta=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_state = None
    epochs_without_improvement = 0
    best_epoch = 0
    
    print("Iniciando treinamento da rede neural...")
    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * X_batch.size(0)
            
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Validação
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                running_val_loss += loss.item() * X_batch.size(0)
                
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        scheduler.step(epoch_val_loss)
        
        if epoch_val_loss < best_val_loss - min_delta:
            best_val_loss = epoch_val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Época {epoch+1:02d}/{epochs} | Loss Treino: {epoch_train_loss:.5f} | Loss Val: {epoch_val_loss:.5f}")
            
        if epochs_without_improvement >= patience:
            print(f"Parada antecipada na época {epoch+1}: validação não melhorou por {patience} épocas consecutivas.")
            break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    print(f"Melhor época baseada em validação: {best_epoch} | Melhor Loss Val: {best_val_loss:.5f}")
    return train_losses, val_losses, best_epoch, best_val_loss

# =====================================================================
# 5. SIMULADOR DE INTERVENÇÃO PEDAGÓGICA (WHAT-IF)
# =====================================================================

def simula_impacto_ia(model, preprocessor, base_student_data, target_mean, target_std, device):
    """
    Simula o impacto de uma intervenção de IA nos hábitos do aluno.
    """
    model.eval()
    
    # Cenário A: Estudante antes do uso IA (Pouca habilidade com IA, uso passivo)
    student_before = base_student_data.copy()
    student_before['Prompt_Engineering_Skill'] = 'Beginner'
    student_before['Primary_Use_Case'] = 'Direct_Answer_Generation' # Uso sem engajamento cognitivo
    student_before['Weekly_GenAI_Hours'] = 25.0                     # Alta dependência ineficaz
    student_before['Traditional_Study_Hours'] = 4.0                 # Pouco estudo tradicional
    
    # Cenário B: Estudante após interagir com a Tutoria em IA
    # (O tutor local estimula o método socrático, melhora o prompt do aluno e equilibra o estudo)
    student_after = base_student_data.copy()
    student_after['Prompt_Engineering_Skill'] = 'Advanced'         # IA treinou o aluno
    student_after['Primary_Use_Case'] = 'Summarizing_Reading'      # Foco em leitura assistida e resumos
    student_after['Weekly_GenAI_Hours'] = 10.0                     # Uso consciente e focado da IA
    student_after['Traditional_Study_Hours'] = 12.0                # Retorno equilibrado ao estudo tradicional
    
    # Converter para DataFrame para passar pelo pipeline do sklearn
    df_before = pd.DataFrame([student_before])
    df_after = pd.DataFrame([student_after])
    
    # Aplicar transformações
    X_before = torch.tensor(preprocessor.transform(df_before).astype(np.float32), dtype=torch.float32, device=device)
    X_after = torch.tensor(preprocessor.transform(df_after).astype(np.float32), dtype=torch.float32, device=device)
    
    with torch.no_grad():
        pred_before_norm = model(X_before).item()
        pred_after_norm = model(X_after).item()

    pred_before = pred_before_norm * target_std + target_mean
    pred_after = pred_after_norm * target_std + target_mean
        
    print("\n" + "="*60)
    print(" SIMULAÇÃO DE IMPACTO PEDAGÓGICO COM A UTILIZAÇÃO DA IA")
    print("="*60)
    print(f"Estudante de {base_student_data['Major_Category']} ({base_student_data['Year_of_Study']})")
    print(f"GPA Inicial: {base_student_data['Pre_Semester_GPA']:.2f}\n")
    print(f"[-] CENÁRIO SEM TUTOR (Uso passivo de IA):")
    print(f"    - Horas de IA/semana: {student_before['Weekly_GenAI_Hours']}h | Estudo Tradicional: {student_before['Traditional_Study_Hours']}h")
    print(f"    - Proficiência de Prompt: {student_before['Prompt_Engineering_Skill']} | Caso de Uso: {student_before['Primary_Use_Case']}")
    print(f"    -> Evolução Prevista de GPA (Delta): {pred_before:+.3f}")
    print(f"    -> GPA Final Estimado: {base_student_data['Pre_Semester_GPA'] + pred_before:.2f}")
    print("-" * 60)
    print(f"[+] CENÁRIO COM IA (Uso ativo/socrático):")
    print(f"    - Horas de IA/semana: {student_after['Weekly_GenAI_Hours']}h | Estudo Tradicional: {student_after['Traditional_Study_Hours']}h")
    print(f"    - Proficiência de Prompt: {student_after['Prompt_Engineering_Skill']} | Caso de Uso: {student_after['Primary_Use_Case']}")
    print(f"    -> Evolução Prevista de GPA (Delta): {pred_after:+.3f}")
    print(f"    -> GPA Final Estimado: {base_student_data['Pre_Semester_GPA'] + pred_after:.2f}")
    print("="*60)
    print(f"RESULTADO: O suporte da IA evitou a perda de retenção de conhecimento e proporcionou um ganho líquido de {pred_after - pred_before:+.3f} no GPA do estudante.")
    print("="*60)

# =====================================================================
# EXECUÇÃO DO PIPELINE
# =====================================================================
if __name__ == "__main__":
    dataset_path = "data/ai_student_impact_dataset.csv"

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Arquivo não encontrado: '{dataset_path}'. Coloque o CSV real na pasta data para usar os dados do projeto.")

    # 1. Pré-processar
    X, y, preprocessor, feature_names, target_mean, target_std = preprocess_data(dataset_path)
    print(f"Dataset carregado com sucesso: {X.shape[0]} amostras e {X.shape[1]} features.")
    
    # 2. Particionar treino e validação
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_dataset = StudentDataset(X_train, y_train)
    val_dataset = StudentDataset(X_val, y_val)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo de treinamento: {device}")
    
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=0)
    
    # 3. Inicializar a rede neural
    input_size = X_train.shape[1]
    model = SimulaRegressor(input_dim=input_size).to(device)
    
    # 4. Treinar
    train_losses, val_losses, best_epoch, best_val_loss = train_model(model, train_loader, val_loader, epochs=12, lr=0.0008, patience=5, device=device)

    # 5. Definir o estudante de exemplo para as análises
    exemplo_estudante = {
        'Major_Category': 'STEM',
        'Year_of_Study': 'Freshman',
        'Primary_Use_Case': 'Summarizing_Reading',
        'Prompt_Engineering_Skill': 'Intermediate',
        'Weekly_GenAI_Hours': 10.0,
        'Traditional_Study_Hours': 12.0,
        'Tool_Diversity': 3,
        'Pre_Semester_GPA': 2.80,
        'Paid_Subscription': 0,
        'Anxiety_Level_During_Exams': 6,
        'Perceived_AI_Dependency': 5
    }
    
    # 6. Gerar todos os gráficos a partir de plots.py
# 6. Gerar todos os gráficos a partir de plots.py (Passando parâmetros de desnormalização)
    generate_all_plots(
        model=model,
        train_losses=train_losses,
        val_losses=val_losses,
        best_epoch=best_epoch,
        val_loader=val_loader,
        preprocessor=preprocessor,
        base_student_data=exemplo_estudante,
        target_mean=target_mean,   # <-- Adicionado
        target_std=target_std,     # <-- Adicionado
        device=device,
    )
    print("Gráficos gerados com sucesso em plots/.")

    # 7. Executar simulação teórica para um aluno de STEM no primeiro ano
    simula_impacto_ia(model, preprocessor, exemplo_estudante, target_mean, target_std, device)

    torch.save(model.state_dict(), 'simuvered_model.pth')