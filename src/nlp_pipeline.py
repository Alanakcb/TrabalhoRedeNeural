import polars as pl
import spacy
import sys
import os

# ==========================================
# INSTRUÇÕES PARA INSTALAÇÃO DO MODELO SPACY
# ==========================================
# Antes de rodar este script, certifique-se de que o modelo em inglês está baixado.
# Execute o seguinte comando no seu terminal:
# python -m spacy download en_core_web_sm
# ==========================================

def load_spacy_model():
    """Tenta carregar o modelo do spaCy, sugerindo o download caso falhe."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        print("ERRO CRÍTICO: Modelo do spaCy 'en_core_web_sm' não foi encontrado!")
        print("-> Por favor, abra o terminal e execute o seguinte comando para baixá-lo:")
        print("python -m spacy download en_core_web_sm")
        sys.exit(1)

# Inicializa o modelo de PLN
nlp = load_spacy_model()

# ==========================================
# CONJUNTOS DE REGRAS (HEURÍSTICAS)
# ==========================================
# Set contendo os lemmas dos principais verbos irregulares da língua inglesa
IRREGULAR_VERBS = {
    "be", "beat", "become", "begin", "bend", "bet", "bid", "bite", "bleed", "blow",
    "break", "bring", "build", "burn", "burst", "buy", "catch", "choose", "come", "cost",
    "cut", "dig", "dive", "do", "draw", "dream", "drive", "drink", "eat", "fall",
    "feel", "fight", "find", "fly", "forget", "forgive", "freeze", "get", "give", "go",
    "grow", "hang", "have", "hear", "hide", "hit", "hold", "hurt", "keep", "know",
    "lay", "lead", "leave", "lend", "let", "lie", "lose", "make", "mean", "meet",
    "pay", "put", "read", "ride", "ring", "rise", "run", "say", "see", "sell",
    "send", "show", "shut", "sing", "sit", "sleep", "speak", "spend", "stand", "swim",
    "take", "teach", "tear", "tell", "think", "throw", "understand", "wake", "wear", "win", "write"
}

def extract_features(text):
    """
    Processa um único texto em inglês e extrai as 4 features solicitadas.
    Lida com valores nulos ou vazios retornando zeros.
    """
    if not text or not isinstance(text, str) or text.strip() == "":
        return {"palavras": 0, "tamanho_frase": 0.0, "verbos_irregulares": 0, "vocabulario_basico": 0}
        
    doc = nlp(text)
    
    # 1. palavras: Contagem total (ignorando pontuação e espaços)
    valid_words = [token for token in doc if not token.is_punct and not token.is_space]
    num_words = len(valid_words)
    
    # 2. tamanho_frase: Média de palavras por frase
    sentences = list(doc.sents)
    num_sentences = len(sentences)
    avg_sentence_len = num_words / num_sentences if num_sentences > 0 else 0.0
    
    # 3. verbos_irregulares: Contagem baseada no lemma
    irregular_count = sum(1 for token in valid_words if token.pos_ == "VERB" and token.lemma_.lower() in IRREGULAR_VERBS)
    
    # 4. vocabulario_basico: Usando a heurística de stop words (frequentes na língua)
    basic_vocab_count = sum(1 for token in valid_words if token.is_stop)
    
    return {
        "palavras": num_words,
        "tamanho_frase": avg_sentence_len,
        "verbos_irregulares": irregular_count,
        "vocabulario_basico": basic_vocab_count
    }

def build_pipeline():
    # Garantir que a pasta de dados exista
    os.makedirs("data", exist_ok=True)
    
    # Caminhos de entrada e saída
    input_file = "data/cefr_leveled_texts.csv"
    output_file = "data/dataset_processado_pmc.csv"
    
    print(f"[{'NLP Pipeline'}] Iniciando pipeline de extração de características...")
    
    # Validação de existência do arquivo
    if not os.path.exists(input_file):
        print(f"AVISO: O arquivo '{input_file}' não foi encontrado no diretório atual.")
        print("Criando um arquivo CSV fictício de exemplo para demonstração do pipeline...")
        # Cria um dataframe mock caso o arquivo não exista para que o script não quebre
        df_mock = pl.DataFrame({
            "text": [
                "Hello, I am a student. I like to eat apples.", 
                "The quick brown fox jumps over the lazy dog. It was a beautiful day, so we went swimming.",
                "Although the implementation was complex, the software engineers successfully deployed the new architecture. Consequently, system latency decreased.",
                None, ""
            ],
            "label": ["A1", "B1", "C2", "A2", "C1"]
        })
        df_mock.write_csv(input_file)
        
    # Lendo o CSV
    print(f"Lendo dados de {input_file}...")
    df = pl.read_csv(input_file)
    
    # Tratando nulos na label
    df = df.drop_nulls(subset=["label"])
    
    print("Processando textos e extraindo features (isso pode demorar dependendo do tamanho do dataset)...")
    
    # Extraindo as features iterando sobre a coluna de texto
    # (Poderíamos usar apply ou map_elements, mas como envolve dicionário múltiplo, lists e structs, a abordagem abaixo é segura)
    features_list = [extract_features(t) for t in df["text"].to_list()]
    
    # Convertendo as listas de extração para o formato de DataFrame do Polars
    df_features = pl.DataFrame(features_list)
    
    # Adicionando as colunas no DataFrame principal
    df = pl.concat([df, df_features], how="horizontal")
    
    # ==========================================
    # MAPEAMENTO ONE-HOT ENCODING DAS SAÍDAS
    # ==========================================
    print("Mapeando a classificação CEFR para One-Hot Encoding...")
    df = df.with_columns([
        # Se for A1 ou A2 -> 1, caso contrário 0
        pl.col("label").is_in(["A1", "A2"]).cast(pl.Int32).alias("out_iniciante"),
        # Se for B1 ou B2 -> 1, caso contrário 0
        pl.col("label").is_in(["B1", "B2"]).cast(pl.Int32).alias("out_intermediario"),
        # Se for C1 ou C2 -> 1, caso contrário 0
        pl.col("label").is_in(["C1", "C2"]).cast(pl.Int32).alias("out_avancado")
    ])
    
    # Selecionando APENAS as colunas numéricas exigidas (4 entradas e 3 saídas)
    df_final = df.select([
        "palavras", 
        "tamanho_frase", 
        "verbos_irregulares", 
        "vocabulario_basico",
        "out_iniciante",
        "out_intermediario",
        "out_avancado"
    ])
    
    # Salvando em disco
    df_final.write_csv(output_file)
    print(f"[{'Sucesso'}] Pipeline finalizado! Arquivo exportado: '{output_file}'")
    print(df_final.head())

if __name__ == "__main__":
    build_pipeline()
