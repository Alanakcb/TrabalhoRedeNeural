import streamlit as st
import polars as pl
import os

# ==========================================
# CONFIGURAÇÃO GERAL DA PÁGINA
# ==========================================
st.set_page_config(
    page_title="Projeto PMC - CEFET-MG",
    page_icon="🧠",
    layout="wide"
)



# ==========================================
# BARRA LATERAL (SIDEBAR)
# ==========================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2103/2103601.png", width=80)
st.sidebar.title("Navegação PMC")
st.sidebar.markdown("---")
opcoes_menu = ['📖 O Projeto e os Dados', '⚙️ Arquitetura da Rede', '📊 Resultados do Treinamento']
escolha = st.sidebar.radio("Selecione a página:", opcoes_menu)
st.sidebar.markdown("---")
st.sidebar.info("Projeto da disciplina de Lab. de Inteligência Artificial - CEFET-MG")

# ==========================================
# PÁGINA 1: O PROJETO E OS DADOS
# ==========================================
if escolha == '📖 O Projeto e os Dados':
    st.header("📖 O Projeto e os Dados")
    
    st.write(
        "Este dashboard apresenta os resultados de um modelo classificador baseado em "
        "**Redes Neurais Artificiais (Perceptron Multicamadas)** desenvolvido totalmente do zero (em NumPy). "
        "O objetivo principal é classificar de forma automática o nível de proficiência em inglês (Iniciante, "
        "Intermediário e Avançado) necessário para a leitura de textos específicos."
    )
    
    st.write(
        "Antes do treinamento, passamos por uma etapa de **Feature Engineering** com NLP (Processamento "
        "de Linguagem Natural) para converter os textos originais em métricas numéricas descritivas."
    )
    
    st.subheader("Visualização dos Dados de Treinamento")
    
    arquivo_dataset = "dataset_processado_pmc.csv"
    
    # Valida se o arquivo já foi gerado
    if os.path.exists(arquivo_dataset):
        try:
            # Carrega o CSV usando a biblioteca Polars
            df = pl.read_csv(arquivo_dataset)
            st.success(f"Arquivo `{arquivo_dataset}` carregado com sucesso!")
            st.write("Abaixo exibimos uma amostra do dataset final com as **4 features extraídas** e as saídas convertidas usando **One-Hot Encoding**:")
            # Renderiza o Dataframe interativamente na tela
            st.dataframe(df.head(20), use_container_width=True)
        except Exception as e:
            st.error(f"Erro ao carregar os dados: {e}")
    else:
        st.warning(
            f"O arquivo `{arquivo_dataset}` não foi encontrado no diretório. "
            "Execute o script de Feature Engineering (`nlp_pipeline.py`) antes de rodar a interface."
        )

# ==========================================
# PÁGINA 2: ARQUITETURA DA REDE
# ==========================================
elif escolha == '⚙️ Arquitetura da Rede':
    st.header("⚙️ Arquitetura da Rede Neural (PMC)")
    
    st.write(
        "Nosso modelo de Perceptron Multicamadas foi estruturado respeitando rigorosamente "
        "as diretrizes e topologias definidas para a disciplina de Laboratório de Inteligência Artificial."
    )
    
    st.markdown("""
    * **Camada de Entrada:** 4 Neurônios.
        - *palavras* (Contagem de palavras totais)
        - *tamanho_frase* (Média de palavras por sentença)
        - *verbos_irregulares* (Ocorrência de principais verbos irregulares da língua inglesa)
        - *vocabulario_basico* (Contagem de vocabulário básico de transição e stopwords)
    * **Camada Oculta:** 1 Camada
        - O número de neurônios foi testado e variado empiricamente em $N_{1} = 5$, $10$ e $15$ neurônios.
    * **Camada de Saída:** 3 Neurônios
        - Padrão **One-Hot Encoding**: Iniciante `[1,0,0]`, Intermediário `[0,1,0]`, Avançado `[0,0,1]`.
    * **Função de Ativação:** Logística (Sigmoide) utilizada universalmente em todas as camadas.
    * **Algoritmo de Treinamento:** *Backpropagation* Clássico com aplicação de Fator de Momentum.
    * **Parâmetros Base:**
        - Taxa de Aprendizado ($\eta$): $0.1$
        - Fator de Momentum ($\alpha$): $0.9$
        - Condição de Precisão / Parada ($\epsilon$): $10^{-6}$
        - Inicialização dos Pesos Sinápticos: Completamente aleatórios entre $0$ e $1$.
    """)

# ==========================================
# PÁGINA 3: RESULTADOS DO TREINAMENTO
# ==========================================
elif escolha == '📊 Resultados do Treinamento':
    st.header("📊 Resultados do Treinamento")
    st.write(
        "Nesta seção podemos comparar o desempenho e o aprendizado das 3 diferentes topologias de rede que testamos."
    )
    st.info("ℹ️ **Separação de Dados:** Utilizamos **80% dos dados para treinamento** e **20% para validação/teste**.")
    
    # Criando duas colunas estruturais
    col_esquerda, col_direita = st.columns([1, 1])
    
    # Coluna Esquerda: Exibir gráfico das Curvas de Erro
    with col_esquerda:
        st.subheader("Curvas de Aprendizado (EQM)")
        arquivo_eqm = "grafico_eqm_epocas.png"
        
        if os.path.exists(arquivo_eqm):
            # use_container_width é a atualização do antigo use_column_width no Streamlit
            st.image(arquivo_eqm, caption="Redução do Erro Quadrático Médio (EQM) no tempo para as Topologias", use_container_width=True)
        else:
            st.warning("O gráfico de Erro Quadrático Médio não foi encontrado. Você executou o treinamento (`mlp_project.py`)?")
            
    # Coluna Direita: Sistema de Abas para as Matrizes de Confusão
    with col_direita:
        st.subheader("Matrizes de Confusão e Taxas de Acerto")
        st.write("Verificação de Acertos no conjunto de testes:")
        
        # Abas utilizando st.tabs()
        aba_5, aba_10, aba_15 = st.tabs(["Topologia 4-5-3", "Topologia 4-10-3", "Topologia 4-15-3"])
        
        with aba_5:
            st.markdown("🎯 **Taxa de Acerto Global: ~79.60%**")
            arquivo_cm_5 = "matriz_confusao_5_neuronios.png"
            if os.path.exists(arquivo_cm_5):
                st.image(arquivo_cm_5, caption="Matriz de Confusão - 5 Neurônios Ocultos", use_container_width=True)
            else:
                st.info("Imagem da Matriz para 5 neurônios indisponível.")
                
        with aba_10:
            st.markdown("🎯 **Taxa de Acerto Global: ~77.93%**")
            arquivo_cm_10 = "matriz_confusao_10_neuronios.png"
            if os.path.exists(arquivo_cm_10):
                st.image(arquivo_cm_10, caption="Matriz de Confusão - 10 Neurônios Ocultos", use_container_width=True)
            else:
                st.info("Imagem da Matriz para 10 neurônios indisponível.")
                
        with aba_15:
            st.markdown("🎯 **Taxa de Acerto Global: ~77.59%**")
            arquivo_cm_15 = "matriz_confusao_15_neuronios.png"
            if os.path.exists(arquivo_cm_15):
                st.image(arquivo_cm_15, caption="Matriz de Confusão - 15 Neurônios Ocultos", use_container_width=True)
            else:
                st.info("Imagem da Matriz para 15 neurônios indisponível.")
