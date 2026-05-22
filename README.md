# LinguaScore · AI English Test 🧠

Este projeto é um sistema inteligente desenvolvido de ponta a ponta para avaliação automática do nível de proficiência em língua inglesa utilizando **Processamento de Linguagem Natural (PLN)** e **Redes Neurais Artificiais (Multilayer Perceptron - MLP)**.

O sistema analisa textos livres produzidos por usuários e estima seu nível com base em métricas linguísticas sintáticas e lexicais, mapeando-os nas classes do quadro europeu de referência **CEFR**: Iniciante (A1/A2), Intermediário (B1/B2) ou Avançado (C1/C2).

---

## 📁 Estrutura do Projeto

Os arquivos foram estruturados para facilitar a manutenção e organização do projeto:

```text
TrabalhoRedeNeural/
├── data/                      # Datasets e dados processados
│   ├── cefr_leveled_texts.csv # Dataset original com textos e labels CEFR
│   └── dataset_processado_pmc.csv # Dataset processado com features e One-Hot Encoding
├── documentos/                # Roteiros de apresentação e relatórios
│   ├── Apresentacao_Trabalho_Rede_Neural.pdf # Relatório técnico formatado
│   ├── ROTEIRO_APRESENTACAO.md # Roteiro markdown de apoio à apresentação técnica
│   └── ROTEIRO_APRESENTACAO.docx # Roteiro Word para entrega acadêmica
├── graficos/                  # Curvas de aprendizado (EQM/Loss) e matrizes de confusão
│   ├── grafico_eqm_epocas.png
│   ├── matriz_confusao_5_neuronios.png
│   ├── matriz_confusao_10_neuronios.png
│   ├── matriz_confusao_15_neuronios.png
│   ├── curva_loss_3x_150_200_250_framework.png
│   ├── matriz_framework_3x_150.png
│   └── ...
├── src/                       # Código-fonte Python
│   ├── app.py                 # Aplicação Streamlit (Interface Web Premium)
│   ├── nlp_pipeline.py        # Extração de características utilizando spaCy (PLN)
│   ├── mlp_project.py         # Implementação manual de rede MLP (NumPy)
│   ├── framework_mlp_project.py # Experimentos de MLP com Scikit-Learn
│   └── gerar_pdf_apresentacao.py # Script de geração do relatório PDF
├── requirements.txt           # Dependências do Python
└── README.md                  # Este arquivo explicativo
```

---

## 🛠️ Pré-requisitos & Instalação

1. **Clonar ou acessar o diretório do projeto**:
   ```bash
   cd TrabalhoRedeNeural
   ```

2. **Criar e ativar o ambiente virtual (opcional, mas recomendado)**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # No Linux
   # ou
   .venv\Scripts\activate     # No Windows
   ```

3. **Instalar dependências**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Baixar o modelo de PLN do spaCy para inglês**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

---

## 🚀 Como Executar

Todos os scripts devem ser executados a partir do **diretório raiz** do projeto:

### 1. Executar o Pipeline de PLN (Feature Engineering)
Processa o dataset bruto e extrai as 4 características sintáticas/lexicais salvando em `data/dataset_processado_pmc.csv`:
```bash
python src/nlp_pipeline.py
```

### 2. Rodar o Treinamento Manual da MLP (NumPy)
Treina e valida a rede implementada em NumPy com as topologias `(5,)`, `(10,)` e `(15,)`:
```bash
python src/mlp_project.py
```

### 3. Rodar os Experimentos do Framework (Scikit-Learn)
Treina e compara arquiteturas maiores `(150, 150, 150)`, `(200, 200, 200)` e `(250, 250, 250)` com ReLU, otimizador Adam e Early Stopping:
```bash
python src/framework_mlp_project.py
```

### 4. Executar o Aplicativo Web Interativo (Streamlit)
Dispara o servidor local do Streamlit para rodar a aplicação no navegador:
```bash
streamlit run src/app.py
```

---

## 📊 Resumo dos Experimentos & Resultados

Durante o desenvolvimento, testamos diversas arquiteturas usando Z-Score scaling, ativação ReLU, otimizador Adam e Early Stopping no Scikit-Learn:

* **Topologia (150, 150, 150)**: **87.29% de acurácia** (Melhor modelo - Escolhido para produção no `app.py`)
* **Topologia (200, 200, 200)**: **84.95% de acurácia**
* **Topologia (250, 250, 250)**: **83.95% de acurácia**

Os gráficos de curvas de perda e matrizes de confusão para todas as execuções podem ser visualizados na pasta `graficos/`.