import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import os

# ==========================================
# 1. GERAÇÃO DO DATASET FICTÍCIO
# ==========================================
def generate_dummy_data(n_samples=300):
    """
    Gera um dataset fictício com 4 variáveis contínuas e 3 classes de proficiência.
    Classes: 0 (Iniciante A1/A2), 1 (Intermediário B1/B2), 2 (Avançado C1/C2)
    """
    np.random.seed(42)
    
    # Gerando dados fictícios por classe
    # Iniciante (textos mais curtos, vocabulário simples)
    words_a = np.random.normal(100, 20, n_samples // 3)
    sent_len_a = np.random.normal(5, 1, n_samples // 3)
    irreg_verbs_a = np.random.normal(2, 1, n_samples // 3)
    vocab_a = np.random.normal(50, 10, n_samples // 3)
    class_a = np.zeros(n_samples // 3)
    
    # Intermediário
    words_b = np.random.normal(250, 40, n_samples // 3)
    sent_len_b = np.random.normal(10, 2, n_samples // 3)
    irreg_verbs_b = np.random.normal(10, 3, n_samples // 3)
    vocab_b = np.random.normal(150, 30, n_samples // 3)
    class_b = np.ones(n_samples // 3)
    
    # Avançado (textos longos, frases complexas)
    words_c = np.random.normal(500, 80, n_samples // 3)
    sent_len_c = np.random.normal(18, 3, n_samples // 3)
    irreg_verbs_c = np.random.normal(25, 5, n_samples // 3)
    vocab_c = np.random.normal(300, 50, n_samples // 3)
    class_c = np.full(n_samples // 3, 2)
    
    # Concatenando
    df = pl.DataFrame({
        "total_words": np.concatenate([words_a, words_b, words_c]),
        "avg_sent_len": np.concatenate([sent_len_a, sent_len_b, sent_len_c]),
        "freq_irreg_verbs": np.concatenate([irreg_verbs_a, irreg_verbs_b, irreg_verbs_c]),
        "basic_vocab": np.concatenate([vocab_a, vocab_b, vocab_c]),
        "proficiency": np.concatenate([class_a, class_b, class_c])
    })
    
    # Embaralhar os dados
    df = df.sample(fraction=1.0, seed=42)
    return df

# ==========================================
# 2. PRÉ-PROCESSAMENTO
# ==========================================
def preprocess_data(df):
    """
    Normaliza as entradas e separa as saídas (que já estão em One-Hot Encoding).
    """
    X = df.select(["palavras", "tamanho_frase", "verbos_irregulares", "vocabulario_basico"]).to_numpy()
    y_one_hot = df.select(["out_iniciante", "out_intermediario", "out_avancado"]).to_numpy()
    
    # Normalização (Min-Max para deixar os dados entre 0 e 1, ideal para a sigmoide)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Separação Treino/Teste
    return train_test_split(X_scaled, y_one_hot, test_size=0.2, random_state=42)

# ==========================================
# 3. IMPLEMENTAÇÃO DO PERCEPTRON MULTICAMADAS
# ==========================================
class MLP:
    def __init__(self, input_size, hidden_size, output_size, lr=0.1, momentum=0.9, precision=1e-6):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr
        self.momentum = momentum
        self.precision = precision
        
        # Inicialização dos pesos aleatoriamente ENTRE 0 E 1 (conforme requisitos)
        # Nota: Inicializar apenas com positivos pode causar saturação inicial rápida na sigmoide.
        self.W1 = np.random.uniform(0, 1, (self.input_size, self.hidden_size))
        self.b1 = np.random.uniform(0, 1, (1, self.hidden_size))
        self.W2 = np.random.uniform(0, 1, (self.hidden_size, self.output_size))
        self.b2 = np.random.uniform(0, 1, (1, self.output_size))
        
        # Variáveis de velocidade para o Momentum
        self.v_W1 = np.zeros_like(self.W1)
        self.v_b1 = np.zeros_like(self.b1)
        self.v_W2 = np.zeros_like(self.W2)
        self.v_b2 = np.zeros_like(self.b2)

    def sigmoid(self, x):
        # np.clip previne overflow no np.exp
        x_clipped = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x_clipped))

    def sigmoid_derivative(self, x):
        # A derivada da sigmoide recebe a própria saída ativada (x = sigmoid(z))
        return x * (1 - x)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        return self.a2

    def backward(self, X, y, output):
        m = X.shape[0]
        
        # Camada de Saída
        error_output = y - output
        # d_output = gradiente local da saída
        d_output = error_output * self.sigmoid_derivative(output)
        
        # Camada Oculta
        error_hidden = np.dot(d_output, self.W2.T)
        d_hidden = error_hidden * self.sigmoid_derivative(self.a1)
        
        # Cálculo dos gradientes médios
        grad_W2 = np.dot(self.a1.T, d_output) / m
        grad_b2 = np.sum(d_output, axis=0, keepdims=True) / m
        grad_W1 = np.dot(X.T, d_hidden) / m
        grad_b1 = np.sum(d_hidden, axis=0, keepdims=True) / m
        
        # Atualização dos Pesos (Backpropagation com Momentum)
        self.v_W2 = self.momentum * self.v_W2 + self.lr * grad_W2
        self.v_b2 = self.momentum * self.v_b2 + self.lr * grad_b2
        self.v_W1 = self.momentum * self.v_W1 + self.lr * grad_W1
        self.v_b1 = self.momentum * self.v_b1 + self.lr * grad_b1
        
        self.W2 += self.v_W2
        self.b2 += self.v_b2
        self.W1 += self.v_W1
        self.b1 += self.v_b1

    def train(self, X, y, max_epochs=10000):
        mse_history = []
        prev_mse = float('inf')
        
        for epoch in range(max_epochs):
            # Forward pass
            output = self.forward(X)
            
            # Cálculo do Erro Quadrático Médio (EQM)
            mse = np.mean(np.square(y - output))
            mse_history.append(mse)
            
            # Condição de Parada (Precisão)
            if abs(prev_mse - mse) < self.precision:
                break
                
            prev_mse = mse
            
            # Backward pass
            self.backward(X, y, output)
            
        return mse_history, epoch + 1

    def predict(self, X):
        output = self.forward(X)
        # Pós-processamento: Mapear saída contínua para números inteiros (0 ou 1)
        # através da determinação da classe com maior probabilidade (argmax)
        predictions = np.zeros_like(output)
        predictions[np.arange(len(output)), output.argmax(axis=1)] = 1
        return predictions

# ==========================================
# 4. EXPERIMENTAÇÃO E VALIDAÇÃO
# ==========================================
def run_experiments():
    # 1. Carregar e preparar os dados reais
    if not os.path.exists("dataset_processado_pmc.csv"):
        print("Erro: O arquivo 'dataset_processado_pmc.csv' não foi encontrado.")
        print("Execute o script 'nlp_pipeline.py' primeiro para gerar o dataset.")
        return
        
    df = pl.read_csv("dataset_processado_pmc.csv")
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    topologies = [5, 10, 15]
    n_runs = 3
    results = {}
    
    # Preparando o plot comparativo
    fig, axes = plt.subplots(1, len(topologies), figsize=(18, 5))
    if not isinstance(axes, np.ndarray):
        axes = [axes]
        
    for idx, hidden_nodes in enumerate(topologies):
        print(f"\n--- Topologia: 4-{hidden_nodes}-3 (Camada Oculta: {hidden_nodes}) ---")
        best_acc = 0
        best_model = None
        all_histories = []
        
        for run in range(n_runs):
            print(f"Execução {run + 1}/{n_runs}...", end=" ")
            
            # Reinicializando a rede e os pesos
            mlp = MLP(input_size=4, hidden_size=hidden_nodes, output_size=3, 
                      lr=0.1, momentum=0.9, precision=1e-6)
                      
            # Treinamento
            mse_hist, epochs = mlp.train(X_train, y_train, max_epochs=3000)
            all_histories.append(mse_hist)
            
            # Validação e Taxa de Acerto (%)
            y_pred = mlp.predict(X_test)
            # Como a saída e y_test estão em One-Hot, usamos argmax para comparar as classes
            acc = accuracy_score(y_test.argmax(axis=1), y_pred.argmax(axis=1))
            
            print(f"Épocas: {epochs} | EQM: {mse_hist[-1]:.6f} | Acerto: {acc * 100:.2f}%")
            
            # Salvando o melhor modelo dessa topologia
            if acc > best_acc:
                best_acc = acc
                best_model = mlp
                
        # Plotando o histórico de EQM (3 execuções)
        for i, hist in enumerate(all_histories):
            axes[idx].plot(hist, label=f'Run {i+1}')
        axes[idx].set_title(f'Treinamento Topologia 4-{hidden_nodes}-3')
        axes[idx].set_xlabel('Épocas')
        axes[idx].set_ylabel('Erro Quadrático Médio (EQM)')
        axes[idx].legend()
        axes[idx].grid(True)
        
        # Avaliando a melhor execução desta topologia
        y_pred_best = best_model.predict(X_test)
        cm = confusion_matrix(y_test.argmax(axis=1), y_pred_best.argmax(axis=1))
        results[hidden_nodes] = {
            'accuracy': best_acc,
            'confusion_matrix': cm
        }
        
    # Salvando Gráfico de Curvas de Aprendizado
    plt.tight_layout()
    plt.savefig('grafico_eqm_epocas.png', dpi=300)
    print("\n[!] Gráfico de curvas de aprendizado salvo como 'grafico_eqm_epocas.png'")
    
    # Fechando figura para não travar a execução
    plt.close()
    
    # Exibir Resultados Finais e Matrizes de Confusão
    print("\n==========================================")
    print("   RESULTADOS FINAIS (Melhores Modelos)")
    print("==========================================")
    classes = ['Iniciante', 'Intermediário', 'Avançado']
    
    for hidden_nodes in topologies:
        acc = results[hidden_nodes]['accuracy']
        cm = results[hidden_nodes]['confusion_matrix']
        print(f"\nTopologia 4-{hidden_nodes}-3:")
        print(f"-> Taxa de Acerto Global: {acc * 100:.2f}%")
        print(f"-> Matriz de Confusão:")
        print(cm)
        
        # Gerar e salvar imagem da Matriz de Confusão
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes, yticklabels=classes)
        plt.title(f'Matriz de Confusão (Topologia 4-{hidden_nodes}-3)')
        plt.xlabel('Classe Predita')
        plt.ylabel('Classe Real')
        plt.tight_layout()
        filename = f'matriz_confusao_{hidden_nodes}_neuronios.png'
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"   [Gráfico salvo: {filename}]")

if __name__ == "__main__":
    run_experiments()
