import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier
import os

def run_framework_experiments():
    # Garantir que a pasta de graficos exista
    os.makedirs("graficos", exist_ok=True)
    
    # 1. Carregamento dos Dados
    if not os.path.exists("data/dataset_processado_pmc.csv"):
        print("Erro: O arquivo 'data/dataset_processado_pmc.csv' não foi encontrado.")
        print("Execute o script de NLP primeiro.")
        return

    df = pl.read_csv("data/dataset_processado_pmc.csv")
    
    # Separando Features (X) e Target (y)
    X = df.select(["palavras", "tamanho_frase", "verbos_irregulares", "vocabulario_basico"]).to_numpy()
    y_one_hot = df.select(["out_iniciante", "out_intermediario", "out_avancado"]).to_numpy()
    
    # O scikit-learn lida melhor com rótulos de classe (0, 1, 2) do que com One-Hot diretamente
    y = np.argmax(y_one_hot, axis=1) 
    
    # MELHORIA: Normalização Padrão (Z-score) costuma ser melhor que Min-Max para redes com ReLU e Adam
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Separação Treino/Teste
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # =====================================================================
    # AQUI ESTÃO OS PARÂMETROS DE ARQUITETURA CONFIGURÁVEIS DA REDE:
    # =====================================================================
    # Uma tupla representa as camadas ocultas. 
    # Ex: (15,) -> Uma camada com 15 neurônios.
    # Ex: (10, 5) -> Duas camadas ocultas, a primeira com 10 e a segunda com 5 neurônios.
    topologies = [(150, 150, 150), (200, 200, 200), (250, 250, 250)]
    
    results = {}
    fig, axes = plt.subplots(1, len(topologies), figsize=(18, 5))
    
    classes = ['Iniciante', 'Intermediário', 'Avançado']

    print("==================================================================")
    print(" INICIANDO TREINAMENTO COM FRAMEWORK (SCIKIT-LEARN)")
    print(" Melhorias Aplicadas: ReLU, Adam, Mini-Batch, Cross-Entropy Loss")
    print("==================================================================\n")

    for idx, topology in enumerate(topologies):
        print(f"--- Treinando Arquitetura Oculta: {topology} ---")
        
        # =====================================================================
        # CONFIGURAÇÃO DO MODELO (ONDE VOCÊ DEFINE AS MELHORIAS)
        # =====================================================================
        mlp = MLPClassifier(
            hidden_layer_sizes=topology,    # <- QTD NEURÔNIOS E CAMADAS OCULTAS
            activation='relu',              # <- ATIVAÇÃO: 'relu' (resolve problema de desvanecimento do gradiente)
            solver='adam',                  # <- OTIMIZADOR: 'adam' (adapta a taxa de aprendizado, muito melhor que SGD puro)
            batch_size=32,                  # <- MINI-BATCH: atualiza pesos a cada 32 amostras (não mais a cada época inteira)
            learning_rate_init=0.01,        # <- Taxa de aprendizado inicial
            max_iter=1000,                  # <- Máximo de épocas de treino
            early_stopping=True,            # <- EARLY STOPPING: Para de treinar se parar de melhorar (evita decorar/overfitting)
            validation_fraction=0.1,        # <- Reserva 10% do treino para checar a melhora do Early Stopping
            n_iter_no_change=15,            # <- Paciência: se não melhorar em 15 épocas seguidas, para o treino
            random_state=42                 # <- Semente para podermos reproduzir os mesmos resultados
        )
        # Obs: Para problemas de classificação multiclasse, o Scikit-Learn automaticamente
        # aplica a função de ativação 'Softmax' na saída e a função de custo 'Cross-Entropy Loss' (Log-Loss).
        
        # 2. Treinamento
        # A magia acontece aqui. O framework gerencia todo o backpropagation otimizado por baixo dos panos (em C/Cython).
        mlp.fit(X_train, y_train)
        
        # 3. Predição e Avaliação
        y_pred = mlp.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        results[topology] = {'accuracy': acc, 'cm': cm}
        
        print(f"-> Épocas utilizadas: {mlp.n_iter_} (Note como parou antes de 1000 graças ao Early Stopping)")
        print(f"-> Função de Custo Final: {mlp.loss_:.4f}")
        print(f"-> Taxa de Acerto (Testes): {acc * 100:.2f}%\n")
        
        # Plotando a Curva de Perda (Loss)
        axes[idx].plot(mlp.loss_curve_, color='purple', linewidth=2)
        axes[idx].set_title(f'Curva de Perda - {topology}')
        axes[idx].set_xlabel('Épocas')
        axes[idx].set_ylabel('Cross-Entropy Loss')
        axes[idx].grid(True)
        
        # Salva a matriz de confusão individualmente
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=classes, yticklabels=classes)
        plt.title(f'Matriz Confusão Framework - {topology[0]} Neurônios')
        plt.xlabel('Classe Predita')
        plt.ylabel('Classe Real')
        plt.tight_layout()
        plt.savefig(f'graficos/matriz_framework_{len(topology)}x_{topology[0]}.png', dpi=300)
        plt.close()

    # Salva o gráfico das curvas de aprendizagem
    fig.tight_layout()
    fig.savefig('graficos/curva_loss_3x_150_200_250_framework.png', dpi=300)
    plt.close()

    print("[!] Gráficos de erro e matrizes de confusão gerados com sucesso no diretório.")

if __name__ == "__main__":
    run_framework_experiments()
