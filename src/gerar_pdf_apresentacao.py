from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        # Arial bold 15
        self.set_font('helvetica', 'B', 15)
        # Move to the right
        # Title
        self.cell(0, 10, 'Relatório Técnico: Sistema de Classificação de Proficiência em Inglês', 0, 1, 'C')
        self.set_font('helvetica', 'I', 11)
        self.cell(0, 10, 'Disciplina: Inteligência Artificial - 8º Período', 0, 1, 'C')
        # Line break
        self.ln(10)

    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font('helvetica', 'I', 8)
        # Page number
        self.cell(0, 10, f'Página {self.page_no()}/{{nb}}', 0, 0, 'C')

    def section_title(self, title):
        self.set_font('helvetica', 'B', 14)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 8, title, 0, 1, 'L', fill=True)
        self.ln(4)

    def section_body(self, text):
        self.set_font('helvetica', '', 11)
        self.multi_cell(0, 6, text)
        self.ln(5)

    def list_item(self, text):
        self.set_font('helvetica', '', 11)
        self.multi_cell(0, 6, chr(149) + " " + text)
        self.ln(2)

def generate_pdf():
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # 1. INTRODUÇÃO E OBJETIVOS
    pdf.section_title("1. Introdução e Objetivo do Projeto")
    text = (
        "Este documento descreve as decisões de arquitetura e implementação do projeto de Inteligência Artificial, "
        "o qual tem como objetivo desenvolver um sistema completo para avaliar o nível de proficiência em língua inglesa "
        "de um usuário. \n\n"
        "O sistema transcende abordagens baseadas em regras simples, empregando uma arquitetura em duas etapas principais: "
        "Extração de Características através de Processamento de Linguagem Natural (PLN) e Classificação usando "
        "Redes Neurais Artificiais (Multilayer Perceptron - MLP). O resultado é servido em uma interface web rica, "
        "onde o modelo treinado realiza inferências em tempo real."
    )
    pdf.section_body(text)

    # 2. PIPELINE DE PROCESSAMENTO DE LINGUAGEM NATURAL
    pdf.section_title("2. Processamento de Linguagem Natural (PLN)")
    text = (
        "Para a primeira etapa, foi desenvolvido o script 'nlp_pipeline.py'. Uma rede neural puramente conectada "
        "necessita de entradas numéricas (features) estruturadas. Para isso, utilizamos a biblioteca spaCy "
        "(modelo en_core_web_sm) para extrair heurísticas e métricas linguísticas do texto fornecido.\n\n"
        "A partir de um dataset base de textos nivelados pelo CEFR (cefr_leveled_texts.csv), o pipeline extrai 4 métricas vitais:"
    )
    pdf.section_body(text)
    
    pdf.list_item("Quantidade de Palavras: Mede a fluência geral e volume de produção textual.")
    pdf.list_item("Tamanho Médio da Frase (Average Sentence Length): Avalia a capacidade do usuário em construir orações complexas.")
    pdf.list_item("Uso de Verbos Irregulares: Indica o domínio de regras gramaticais e estruturas lexicais não-triviais.")
    pdf.list_item("Vocabulário Básico (Stop Words): Ajuda a estimar a densidade lexical do texto.")
    
    text = (
        "O dataset original contava com categorias CEFR (A1, A2, B1, B2, C1, C2). Para simplificar e otimizar o aprendizado da rede, "
        "optamos por agrupá-los em três níveis gerais usando One-Hot Encoding: Iniciante (A1/A2), Intermediário (B1/B2) e Avançado (C1/C2)."
    )
    pdf.section_body(text)

    # 3. MODELO DE REDE NEURAL (MLP)
    pdf.section_title("3. Arquitetura da Rede Neural (MLP)")
    text = (
        "Com as características extraídas numéricas, o modelo de predição é alimentado por uma rede Perceptron Multicamadas (MLP). "
        "As principais decisões para esta arquitetura envolveram:"
    )
    pdf.section_body(text)
    
    pdf.list_item("Topologia e Camadas Ocultas: Foram experimentadas topologias de 5, 10 e 15 neurônios na camada oculta. "
                  "Através das Matrizes de Confusão e curvas de Erro Quadrático Médio (EQM) geradas nas análises de validação, a melhor capacidade de generalização foi obtida sem incorrer em overfitting.")
    pdf.list_item("Padronização de Dados: Utilizamos StandardScaler. Por lidarmos com grandezas diferentes (contagens brutas vs médias), "
                  "a normalização z-score garante que o Gradiente Descendente convirja eficientemente.")
    pdf.list_item("Função de Ativação e Otimizador: O modelo baseia-se na função ReLU, evitando o problema de desvanecimento do gradiente, "
                  "e utiliza o otimizador Adam para adaptação dinâmica do passo de aprendizagem.")
    pdf.list_item("Prevenção ao Overfitting: A técnica de early_stopping é ativada para parar o treinamento caso a perda (loss) de validação não diminua consecutivamente.")

    # 4. INTERFACE E INFERÊNCIA EM TEMPO REAL
    pdf.section_title("4. Aplicação e Inferência em Tempo Real (App)")
    text = (
        "A última fase do sistema une o modelo treinado com uma interface gráfica desenvolvida via Streamlit ('app.py'). "
        "Ao invés de processamento em batch, o fluxo se torna interativo:\n"
    )
    pdf.section_body(text)
    
    pdf.list_item("Entrada Dinâmica: O usuário responde 3 perguntas abertas aleatórias de um banco de questões focadas no contexto real.")
    pdf.list_item("Feedback e Confiança: As respostas são concatenadas e passadas pelas mesmas funções de extração (extract_features). "
                  "O modelo padroniza os valores com o StandardScaler treinado e não só emite a classe final (predict), mas também fornece as probabilidades "
                  "para cada classe (predict_proba), apresentando a métrica de 'Model Confidence' ao usuário.")
    pdf.list_item("Design Premium: Adotamos regras avançadas de injeção CSS e renderização gráfica condicional responsiva para garantir UX imersiva.")

    # 5. CONCLUSÃO
    pdf.section_title("5. Conclusão")
    text = (
        "O sistema comprova empiricamente de forma prática conceitos teóricos da Inteligência Artificial: desde a engenharia de atributos (Feature Engineering) "
        "e processamento de linguagem natural, até o treinamento, otimização hiperparamétrica e "
        "aplicação prática de um modelo connectionista (MLP) servido em uma aplicação de ponta a ponta.\n\n"
        "O fluxo demonstra maturidade no desenvolvimento de IA para solucionar um problema semântico abstrato (avaliação de inglês) transformando-o "
        "em uma classificação puramente matemática baseada em padrões quantitativos textuais."
    )
    pdf.section_body(text)

    import os
    os.makedirs('documentos', exist_ok=True)
    pdf.output('documentos/Apresentacao_Trabalho_Rede_Neural.pdf')
    print("PDF gerado com sucesso!")

if __name__ == '__main__':
    generate_pdf()
