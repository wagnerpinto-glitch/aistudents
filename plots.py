import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# Configurando estilo visual para artigos científicos (Academic Style)
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16,
    'axes.edgecolor': '#444444',
    'grid.color': '#d0d0d0',
    'grid.linewidth': 0.8,
})

# =====================================================================
# G1. PLOT DA CURVA DE PERDA (LOSS CURVE)
# =====================================================================
def plot_loss_curve(train_losses, val_losses, best_epoch):
    plt.figure(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, label='Loss de Treino (MSE)', color='#1f77b4', linewidth=2)
    plt.plot(epochs, val_losses, label='Loss de Validação (MSE)', color='#ff7f0e', linewidth=2, linestyle='--')
    
    # Linha indicando a melhor época (onde os pesos foram salvos)
    plt.axvline(x=best_epoch, color='red', linestyle=':', label=f'Melhor Época ({best_epoch})')
    
    plt.title('Curva de Convergência do SimuVered (MLP)', pad=8)
    plt.xlabel('Épocas de Treinamento')
    plt.ylabel('Erro Quadrático Médio (MSE)')
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/g1_loss_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

# =====================================================================
# G2. REAL VS. PREDITO (VALOR DE VALIDAÇÃO)
# =====================================================================
def plot_real_vs_predicted(model, val_loader):
    model.eval()
    all_reals = []
    all_preds = []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            preds = model(X_batch)
            all_reals.extend(y_batch.numpy().flatten())
            all_preds.extend(preds.numpy().flatten())
            
    plt.figure(figsize=(7, 6))
    # Gráfico de densidade de pontos (Hexbin é ótimo para 10.000+ amostras de validação)
    plt.hexbin(all_reals, all_preds, gridsize=40, cmap='Blues', mincnt=1)
    
    # Linha de Identidade (Perfeição)
    lims = [
        min(min(all_reals), min(all_preds)),
        max(max(all_reals), max(all_preds))
    ]
    plt.plot(lims, lims, color='red', linestyle='--', alpha=0.8, label='Predição Perfeita ($y=x$)')
    
    plt.title('Precisão Preditiva: Real vs. Predito', pad=8)
    plt.xlabel('Variação de GPA Real ($\\Delta$ GPA)')
    plt.ylabel('Variação de GPA Predito pela MLP')
    plt.colorbar(label='Densidade de Estudantes')
    plt.legend(loc='upper left')
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/g2_real_vs_predicted.png', dpi=300, bbox_inches='tight')
    plt.close()

# =====================================================================
# G3. ANÁLISE DE SENSIBILIDADE (HORAS DE IA vs. DELTA GPA)
# =====================================================================
def plot_sensitivity_analysis(model, preprocessor, base_student_data, device=None):
    model.eval()
    ai_hours_range = np.linspace(0, 40, 100)
    predictions_beginner = []
    predictions_advanced = []
    
    # Gerando dados variando apenas as horas de IA para dois perfis de Prompt
    for hours in ai_hours_range:
        # Perfil A: Iniciante usando IA como atalho cognitivo
        student_beg = base_student_data.copy()
        student_beg['Weekly_GenAI_Hours'] = hours
        student_beg['Prompt_Engineering_Skill'] = 'Beginner'
        student_beg['Primary_Use_Case'] = 'Direct_Answer_Generation'
        
        # Perfil B: Avançado sob tutela do VeredAI (Socrático/Leitura)
        student_adv = base_student_data.copy()
        student_adv['Weekly_GenAI_Hours'] = hours
        student_adv['Prompt_Engineering_Skill'] = 'Advanced'
        student_adv['Primary_Use_Case'] = 'Summarizing_Reading'
        
        # Converter para Dataframe e transformar
        df_beg = pd.DataFrame([student_beg])
        df_adv = pd.DataFrame([student_adv])
        X_beg = torch.tensor(preprocessor.transform(df_beg).astype(np.float32), dtype=torch.float32, device=device)
        X_adv = torch.tensor(preprocessor.transform(df_adv).astype(np.float32), dtype=torch.float32, device=device)
        
        with torch.no_grad():
            predictions_beginner.append(model(X_beg).item())
            predictions_advanced.append(model(X_adv).item())
            
    plt.figure(figsize=(9, 5))
    plt.plot(ai_hours_range, predictions_beginner, label='Uso Passivo (Sem Tutor/Beginner)', color='#d62728', linewidth=2.5)
    plt.plot(ai_hours_range, predictions_advanced, label='Uso Ativo (Com VeredAI/Advanced)', color='#2ca02c', linewidth=2.5)
    
    plt.axhline(0, color='gray', linestyle=':', alpha=0.6)
    plt.title('Análise de Sensibilidade: Impacto das Horas de IA no GPA', pad=8)
    plt.xlabel('Horas Semanais de Uso de IA Generativa')
    plt.ylabel('Variação de GPA Prevista ($\\Delta$ GPA)')
    plt.legend(loc='upper left')
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/g3_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

# =====================================================================
# G4. FRONTEIRA PEDAGÓGICA (HEATMAP)
# =====================================================================
def plot_pedagogical_heatmap(model, preprocessor, base_student_data, device=None):
    model.eval()
    
    # Definindo eixos do Heatmap
    ai_hours = np.linspace(0, 35, 30)
    trad_study = np.linspace(0, 25, 30)
    
    grid_gpa = np.zeros((len(trad_study), len(ai_hours)))
    
    for i, trad in enumerate(trad_study):
        for j, ai in enumerate(ai_hours):
            student = base_student_data.copy()
            student['Weekly_GenAI_Hours'] = ai
            student['Traditional_Study_Hours'] = trad
            
            df_student = pd.DataFrame([student])
            X_student = torch.tensor(preprocessor.transform(df_student).astype(np.float32), dtype=torch.float32, device=device)
            
            with torch.no_grad():
                grid_gpa[i, j] = model(X_student).item()
                
    plt.figure(figsize=(9, 7))
    # Inverter o eixo Y para o estudo tradicional crescer de baixo para cima
    sns.heatmap(grid_gpa, xticklabels=np.round(ai_hours, 1), yticklabels=np.round(trad_study, 1),
                cmap='RdYlGn', center=0.0, cbar_kws={'label': 'Variação Prevista de GPA (Δ GPA)'})
    
    plt.title('Fronteira Pedagógica: Interação entre Métodos de Estudo', pad=8)
    plt.xlabel('Horas de Uso de IA / Semana')
    plt.ylabel('Horas de Estudo Tradicional / Semana')
    
    # Limitar exibição de ticks para não poluir o gráfico
    plt.locator_params(axis='x', nbins=10)
    plt.locator_params(axis='y', nbins=10)
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/g4_pedagogical_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()