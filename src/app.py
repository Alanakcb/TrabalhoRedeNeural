import streamlit as st
import polars as pl
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import spacy
import os
import random

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="LinguaScore · AI English Test",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────
st.html("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
background: #0a0a12 !important;
font-family: 'Inter', sans-serif !important;
color: #e8e8f0 !important;
overflow-x: hidden !important;
}

[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stSidebarNav"],
[data-testid="stDecoration"],
footer { display: none !important; }

[data-testid="stAppViewContainer"] > .main,
.main .block-container {
padding: 0 !important;
max-width: 100% !important;
}

.stTextArea textarea {
font-family: 'Inter', sans-serif !important;
font-size: 1rem !important;
background: #12121e !important;
color: #e8e8f0 !important;
border: 2px solid #2a2a40 !important;
border-radius: 14px !important;
padding: 16px !important;
resize: none !important;
caret-color: #6c63ff !important;
transition: border-color 0.25s ease !important;
}
.stTextArea textarea:focus {
border-color: #6c63ff !important;
box-shadow: 0 0 0 4px rgba(108,99,255,0.15) !important;
outline: none !important;
}
.stTextArea label,
[data-testid="InputInstructions"] { display: none !important; }
.stTextArea > div > div { background: transparent !important; border: none !important; }

.stButton > button {
font-family: 'Inter', sans-serif !important;
font-weight: 600 !important;
cursor: pointer !important;
transition: all 0.22s ease !important;
border-radius: 100px !important;
border: 1px solid rgba(255,255,255,0.1) !important;
background: rgba(255,255,255,0.05) !important;
color: #c0c0e0 !important;
padding: 12px 28px !important;
}
.stButton > button:hover {
background: rgba(255,255,255,0.09) !important;
color: #fff !important;
}

[data-testid="stNotification"] { border-radius: 12px !important; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: #3d3d5c; border-radius: 4px; }

@keyframes fadeUp {
from { opacity: 0; transform: translateY(24px); }
to   { opacity: 1; transform: translateY(0); }
}
@keyframes float {
0%,100% { transform: translateY(0); }
50%      { transform: translateY(-10px); }
}
@keyframes popIn {
0%   { opacity: 0; transform: scale(0.82); }
72%  { transform: scale(1.05); }
100% { opacity: 1; transform: scale(1); }
}

.screen {
display: flex;
flex-direction: column;
align-items: center;
justify-content: center;
width: 100%;
}

.landing-tag {
display: inline-block;
font-size: .72rem; font-weight: 700;
letter-spacing: .2em; text-transform: uppercase;
color: #6c63ff;
background: rgba(108,99,255,.1);
border: 1px solid rgba(108,99,255,.3);
border-radius: 100px;
padding: 6px 18px;
margin-bottom: 28px;
animation: fadeUp .55s ease both;
}
.landing-orb {
width: 160px; height: 160px;
border-radius: 50%;
background: linear-gradient(135deg, #6c63ff, #10b981);
display: flex; align-items: center; justify-content: center;
font-size: 66px;
box-shadow: 0 0 80px rgba(108,99,255,.5), 0 0 160px rgba(108,99,255,.18);
animation: float 4s ease-in-out infinite;
margin-bottom: 36px;
}
.landing-title {
font-size: clamp(2.2rem, 5.5vw, 3.4rem);
font-weight: 800; line-height: 1.15; text-align: center;
margin-bottom: 18px;
background: linear-gradient(135deg, #ffffff 30%, #a8a4ff 100%);
-webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
animation: fadeUp .6s .08s ease both;
}
.landing-sub {
font-size: 1.05rem; color: #9090b0; text-align: center;
max-width: 460px; line-height: 1.75;
margin-bottom: 44px; font-weight: 400;
animation: fadeUp .6s .16s ease both;
}
.landing-stats {
display: flex; gap: 16px; flex-wrap: wrap;
justify-content: center;
margin-bottom: 44px;
animation: fadeUp .6s .24s ease both;
}
.l-stat {
background: rgba(255,255,255,.04);
border: 1px solid rgba(255,255,255,.08);
border-radius: 18px; padding: 20px 28px; text-align: center;
min-width: 110px;
}
.l-stat .icon { font-size: 1.5rem; margin-bottom: 8px; }
.l-stat .val  { font-size: 1.4rem; font-weight: 700; color: #fff; }
.l-stat .lbl  { font-size: .73rem; color: #60609a; margin-top: 4px; }

.quiz-card {
background: rgba(255,255,255,.03);
border: 1px solid rgba(255,255,255,.07);
border-radius: 28px;
padding: clamp(32px, 5vw, 52px) clamp(28px, 5vw, 52px);
max-width: 680px; width: 100%;
box-shadow: 0 32px 80px rgba(0,0,0,.4);
animation: fadeUp .4s cubic-bezier(.22,1,.36,1) both;
}
.quiz-top {
display: flex; align-items: center;
justify-content: space-between;
margin-bottom: 36px;
}
.quiz-counter {
font-size: .76rem; font-weight: 700;
letter-spacing: .15em; text-transform: uppercase;
color: #6c63ff; flex-shrink: 0;
}
.quiz-prog-bg {
flex: 1; margin-left: 18px;
height: 4px; background: rgba(255,255,255,.06);
border-radius: 100px; overflow: hidden;
}
.quiz-prog-fill {
height: 100%;
background: linear-gradient(90deg, #6c63ff, #10b981);
border-radius: 100px;
transition: width .5s ease;
}
.quiz-q {
font-size: clamp(1.2rem, 3vw, 1.5rem);
font-weight: 700; color: #fff; line-height: 1.45;
margin-bottom: 10px;
}
.quiz-hint {
font-size: .875rem; color: #60609a;
font-style: italic; margin-bottom: 28px;
}
.quiz-footer-tip {
font-size: .78rem; color: #3a3a5a; margin-top: 18px;
}

.res-hero { text-align: center; max-width: 600px; margin: 0 auto 48px; animation: fadeUp .6s ease both; }
.res-badge {
display: inline-flex; align-items: center; gap: 8px;
border-radius: 100px; padding: 7px 18px;
font-size: .72rem; font-weight: 700;
letter-spacing: .15em; text-transform: uppercase;
margin-bottom: 26px;
}
.res-orb {
width: 120px; height: 120px; border-radius: 50%;
display: flex; align-items: center; justify-content: center;
font-size: 50px; margin: 0 auto 24px;
animation: popIn .7s cubic-bezier(.22,1,.36,1) both;
}
.res-level {
font-size: clamp(2rem,5vw,3rem);
font-weight: 800; margin-bottom: 4px;
-webkit-background-clip: text;
-webkit-text-fill-color: transparent;
background-clip: text;
}
.res-cefr  { font-size: .95rem; color: #6060a0; margin-bottom: 18px; }
.res-desc  { font-size: 1rem; color: #a0a0c0; line-height: 1.72; }

.res-card {
background: rgba(255,255,255,.03);
border: 1px solid rgba(255,255,255,.07);
border-radius: 22px; padding: 26px 28px;
animation: fadeUp .65s .12s ease both;
height: 100%;
}
.res-card-title {
font-size: .7rem; font-weight: 700;
letter-spacing: .15em; text-transform: uppercase;
color: #40408a; margin-bottom: 22px;
}
.stat-lbl { font-size: .8rem; color: #60609a; margin-bottom: 5px; }
.stat-val { font-size: .88rem; color: #b0b0d0; font-weight: 600; margin-bottom: 14px; }
.bar-bg   { height: 6px; background: rgba(255,255,255,.06); border-radius: 100px; overflow: hidden; margin-bottom: 4px; }
.bar-fill { height: 100%; border-radius: 100px; }

.prob-row  { display:flex; align-items:center; gap:12px; margin-bottom:10px; }
.prob-lbl  { width:95px; font-size:.78rem; color:#70709a; flex-shrink:0; }
.prob-bg   { flex:1; height:7px; background:rgba(255,255,255,.06); border-radius:100px; overflow:hidden; }
.prob-fill { height:100%; border-radius:100px; }
.prob-val  { width:36px; font-size:.78rem; text-align:right; font-weight:600; }

.tip-card {
background: rgba(255,255,255,.03);
border: 1px solid rgba(255,255,255,.07);
border-radius: 16px; padding: 20px 24px; margin-bottom: 10px;
}
.tip-title { font-weight:600; font-size:.93rem; color:#fff; margin-bottom:6px; }
.tip-body  { font-size:.86rem; color:#80809a; line-height:1.62; }
</style>
""")


# ─────────────────────────────────────────
# CACHED RESOURCES
# ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def carregar_spacy():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        os.system(".venv/bin/python -m spacy download en_core_web_sm")
        return spacy.load("en_core_web_sm")

@st.cache_resource(show_spinner=False)
def treinar_modelo():
    arquivo = "data/dataset_processado_pmc.csv"
    if not os.path.exists(arquivo):
        return None, None
    df = pl.read_csv(arquivo)
    X  = df.select(["palavras","tamanho_frase","verbos_irregulares","vocabulario_basico"]).to_numpy()
    y  = np.argmax(df.select(["out_iniciante","out_intermediario","out_avancado"]).to_numpy(), axis=1)
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    mlp = MLPClassifier(hidden_layer_sizes=(150, 150, 150), activation="relu", solver="adam",
                        batch_size=32, max_iter=1000, early_stopping=True, random_state=42)
    mlp.fit(Xs, y)
    return mlp, sc

nlp            = carregar_spacy()
mlp_model, sc  = treinar_modelo()


# ─────────────────────────────────────────
# NLP FEATURE EXTRACTION
# ─────────────────────────────────────────
IRREGULAR_VERBS = {
    "be","beat","become","begin","bend","bet","bid","bite","bleed","blow","break","bring",
    "build","burn","burst","buy","catch","choose","come","cost","cut","dig","dive","do",
    "draw","dream","drive","drink","eat","fall","feel","fight","find","fly","forget",
    "forgive","freeze","get","give","go","grow","hang","have","hear","hide","hit","hold",
    "hurt","keep","know","lay","lead","leave","lend","let","lie","lose","make","mean",
    "meet","pay","put","read","ride","ring","rise","run","say","see","sell","send","show",
    "shut","sing","sit","sleep","speak","spend","stand","swim","take","teach","tear","tell",
    "think","throw","understand","wake","wear","win","write"
}

def extract_features(text):
    if not text.strip():
        return [0, 0.0, 0, 0]
    doc   = nlp(text)
    words = [t for t in doc if not t.is_punct and not t.is_space]
    sents = list(doc.sents)
    avg   = len(words) / len(sents) if sents else 0.0
    irreg = sum(1 for t in words if t.pos_ == "VERB" and t.lemma_.lower() in IRREGULAR_VERBS)
    stops = sum(1 for t in words if t.is_stop)
    return [len(words), avg, irreg, stops]


# ─────────────────────────────────────────
# QUESTION BANK
# ─────────────────────────────────────────
ALL_QUESTIONS = [
    {"q": "Tell me about yourself — your background, passions, and what drives you every day.",
     "hint": "Think about hobbies, career, or values."},
    {"q": "Describe a difficult challenge you've faced recently. How did you handle it?",
     "hint": "Be specific — explain the situation, your actions, and what you learned."},
    {"q": "If you could live in any era of history, which would you choose and why?",
     "hint": "Consider technology, culture, events, or lifestyle."},
    {"q": "What do you believe is the most important invention of the last 100 years?",
     "hint": "Think beyond the obvious — explain the impact on society."},
    {"q": "Describe your ideal day from morning to night.",
     "hint": "Include activities, places, people, and why it would be perfect."},
    {"q": "If you could have a conversation with any person, living or dead, who would it be and why?",
     "hint": "Explain what you'd discuss and what you hope to learn."},
    {"q": "How has technology changed the way people connect with each other?",
     "hint": "Think about social media, remote work, relationships, etc."},
]

LEVEL_DATA = {
    0: {
        "label":"Beginner",   "cefr":"A1 / A2",
        "color":"#f59e0b",
        "gradient":"linear-gradient(135deg,#f59e0b,#ef4444)",
        "icon":"🌱",
        "badge_bg":"rgba(245,158,11,0.14)",
        "badge_border":"rgba(245,158,11,0.35)",
        "glow":"rgba(245,158,11,0.45)",
        "description":"You're building your foundation! Your sentences are clear and straightforward — a great starting point for any language journey.",
        "tips":[
            ("📚 Vocabulary expansion","Aim to learn 5–10 new words daily using apps like Anki or Duolingo."),
            ("🗣️ Sentence structure","Practice forming longer sentences by connecting two ideas with 'because', 'however', or 'although'."),
            ("✍️ Daily writing habit","Keep a short English diary (3–5 sentences). Consistency beats volume at this stage."),
            ("🎧 Active listening","Watch English content with subtitles. Pause and repeat phrases out loud."),
        ]
    },
    1: {
        "label":"Intermediate","cefr":"B1 / B2",
        "color":"#6c63ff",
        "gradient":"linear-gradient(135deg,#6c63ff,#3b82f6)",
        "icon":"⚡",
        "badge_bg":"rgba(108,99,255,0.14)",
        "badge_border":"rgba(108,99,255,0.35)",
        "glow":"rgba(108,99,255,0.45)",
        "description":"Solid command of English! You communicate your ideas clearly and use a good range of vocabulary and structures.",
        "tips":[
            ("🔗 Complex connectors","Expand your use of discourse markers: 'consequently', 'in contrast', 'nevertheless', 'given that'."),
            ("📖 Academic reading","Read articles from The Guardian or BBC. Summarize each paragraph in your own words."),
            ("🎭 Idioms & collocations","Learn 3 idioms per week in context — not in isolation. Use them in sentences immediately."),
            ("🎤 Fluency over accuracy","Find a language exchange partner or join online debate clubs like iTalki."),
        ]
    },
    2: {
        "label":"Advanced",   "cefr":"C1 / C2",
        "color":"#10b981",
        "gradient":"linear-gradient(135deg,#10b981,#0ea5e9)",
        "icon":"🏆",
        "badge_bg":"rgba(16,185,129,0.14)",
        "badge_border":"rgba(16,185,129,0.35)",
        "glow":"rgba(16,185,129,0.45)",
        "description":"Exceptional proficiency! You write with complexity, variety, and nuance. Your vocabulary and sentence control are highly sophisticated.",
        "tips":[
            ("✒️ Stylistic refinement","Vary between formal, academic, and colloquial registers intentionally."),
            ("📝 Academic writing","Practice argumentative essays: clear thesis, developed body paragraphs, rebuttals."),
            ("🌍 Cultural fluency","Dive into literature and films without subtitles. Notice cultural references and humor."),
            ("🔬 Linguistic analysis","Study phonology and pragmatics to deepen your meta-understanding of language."),
        ]
    },
}


# ─────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────
DEFAULTS = {
    "screen":    "landing",
    "questions": [],
    "q_index":   0,
    "answers":   [],
    "result":    None,
    "features":  None,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────
# SCREEN 1 — LANDING
# ─────────────────────────────────────────
def render_landing():
    st.html("""
    <style>
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(ellipse 80% 55% at 50% 0%, rgba(108,99,255,.18) 0%, transparent 68%),
                    radial-gradient(ellipse 55% 45% at 85% 100%, rgba(16,185,129,.10) 0%, transparent 60%),
                    #0a0a12 !important;
    }
    .main .block-container > div[data-testid="stVerticalBlock"] {
        justify-content: center;
        min-height: 100vh;
        padding: clamp(32px, 6vh, 72px) 24px;
    }
    @media (min-width: 768px) {
        .main .block-container,
        .main .block-container > div[data-testid="stVerticalBlock"] {
            height: 100vh !important;
            max-height: 100vh !important;
            overflow: hidden !important;
        }
    }
    </style>
    """)

    html_content = """
<div class="screen">
<div class="landing-tag">🧠 Powered by Neural Network</div>
<div class="landing-orb">🎯</div>
<h1 class="landing-title">Discover Your<br>English Level</h1>
<p class="landing-sub">
Answer 3 open-ended questions in English.<br>
Our AI analyzes your grammar, vocabulary &amp; sentence complexity.
</p>
<div class="landing-stats">
<div class="l-stat"><div class="icon">❓</div><div class="val">3</div><div class="lbl">Questions</div></div>
<div class="l-stat"><div class="icon">⚡</div><div class="val">&lt;5s</div><div class="lbl">AI Analysis</div></div>
<div class="l-stat"><div class="icon">🎯</div><div class="val">88%</div><div class="lbl">Accuracy</div></div>
</div>
</div>
"""
    st.html(html_content)

    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        if st.button("Start the Test →", key="btn_start", use_container_width=True):
            st.session_state.questions = random.sample(ALL_QUESTIONS, 3)
            st.session_state.screen    = "quiz"
            st.rerun()

    btn_style = """
<style>
div[data-testid="column"]:nth-child(2) .stButton > button {
background: linear-gradient(135deg,#6c63ff,#4f46e5) !important;
color: #fff !important;
font-size: 1.05rem !important;
padding: 14px 32px !important;
box-shadow: 0 8px 32px rgba(108,99,255,.42) !important;
border: none !important;
}
div[data-testid="column"]:nth-child(2) .stButton > button:hover {
transform: translateY(-2px) !important;
box-shadow: 0 14px 40px rgba(108,99,255,.58) !important;
}
</style>
"""
    st.html(btn_style)


# ─────────────────────────────────────────
# SCREEN 2 — QUIZ
# ─────────────────────────────────────────
def render_quiz():
    st.html("""
    <style>
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(ellipse 65% 45% at 25% 20%, rgba(108,99,255,.12) 0%, transparent 60%), #0a0a12 !important;
    }
    .main .block-container > div[data-testid="stVerticalBlock"] {
        justify-content: center;
        min-height: 100vh;
        padding: clamp(32px, 6vh, 72px) 24px;
    }
    @media (min-width: 768px) {
        .main .block-container,
        .main .block-container > div[data-testid="stVerticalBlock"] {
            height: 100vh !important;
            max-height: 100vh !important;
            overflow: hidden !important;
        }
    }
    </style>
    """)

    idx   = st.session_state.q_index
    total = len(st.session_state.questions)
    q     = st.session_state.questions[idx]
    pct   = int(idx / total * 100)

    html_top = f"""
<div class="screen" style="align-items:stretch;">
<div style="display:flex;justify-content:center;width:100%;">
<div class="quiz-card">
<div class="quiz-top">
<span class="quiz-counter">Question {idx+1} of {total}</span>
<div class="quiz-prog-bg">
<div class="quiz-prog-fill" style="width:{pct}%;"></div>
</div>
</div>
<div class="quiz-q">{q["q"]}</div>
<div class="quiz-hint">💡 {q["hint"]}</div>
"""
    st.html(html_top)

    saved = st.session_state.answers[idx] if idx < len(st.session_state.answers) else ""
    answer = st.text_area(
        label="answer",
        value=saved,
        placeholder="Write your answer here — the more detail, the better the result...",
        height=150,
        key=f"ta_{idx}",
        label_visibility="collapsed"
    )

    html_bottom = """
<p class="quiz-footer-tip">💬 Longer, richer answers produce more accurate results.</p>
</div></div></div>
"""
    st.html(html_bottom)

    b1, b2 = st.columns(2)
    with b1:
        if idx > 0:
            if st.button("← Back", key="btn_back", use_container_width=True):
                st.session_state.q_index -= 1
                st.rerun()
    with b2:
        label = "Next →" if idx < total - 1 else "Analyze My English 🧠"
        if st.button(label, key="btn_next", use_container_width=True):
            if not answer.strip():
                st.warning("Please write something before continuing.")
            else:
                if idx < len(st.session_state.answers):
                    st.session_state.answers[idx] = answer
                else:
                    st.session_state.answers.append(answer)

                if idx < total - 1:
                    st.session_state.q_index += 1
                    st.rerun()
                else:
                    with st.spinner("Analyzing your English..."):
                        full_text = " ".join(st.session_state.answers)
                        feats     = extract_features(full_text)
                        feats_sc  = sc.transform([feats])
                        pred      = mlp_model.predict(feats_sc)[0]
                        proba     = mlp_model.predict_proba(feats_sc)[0]
                    st.session_state.features = feats
                    st.session_state.result   = {"label": int(pred), "proba": proba.tolist()}
                    st.session_state.screen   = "results"
                    st.rerun()

    btn_style = """
<style>
div[data-testid="column"]:nth-child(2) .stButton > button {
background: linear-gradient(135deg,#6c63ff,#4f46e5) !important;
color:#fff !important; border:none !important;
box-shadow: 0 6px 24px rgba(108,99,255,.35) !important;
}
div[data-testid="column"]:nth-child(2) .stButton > button:hover {
box-shadow: 0 10px 30px rgba(108,99,255,.52) !important;
transform: translateY(-1px) !important;
}
</style>
"""
    st.html(btn_style)


# ─────────────────────────────────────────
# SCREEN 3 — RESULTS
# ─────────────────────────────────────────
def render_results():
    res   = st.session_state.result
    feats = st.session_state.features
    pred  = res["label"]
    proba = res["proba"]
    d     = LEVEL_DATA[pred]

    words, avg_sl, irreg, _ = feats
    conf = int(proba[pred] * 100)

    bg_color = d["color"]
    badge_bg = d["badge_bg"].replace("0.14","0.22")
    
    st.html(f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background: radial-gradient(ellipse 75% 50% at 50% 0%, {badge_bg} 0%, transparent 60%), #0a0a12 !important;
    }}
    .main .block-container > div[data-testid="stVerticalBlock"] {{
        justify-content: center;
        min-height: 100vh;
        padding: clamp(32px, 6vh, 72px) 24px;
    }}
    </style>
    """)

    html_hero = f"""
<div class="screen">
<div class="res-hero">
<div class="res-badge" style="background:{d["badge_bg"]};border:1px solid {d["badge_border"]};color:{d["color"]};">
{d["icon"]} Your Result
</div>
<div class="res-orb" style="background:{d["gradient"]};box-shadow:0 0 60px {d["glow"]};">
{d["icon"]}
</div>
<div class="res-level" style="background:{d["gradient"]};">{d["label"]}</div>
<div class="res-cefr">CEFR Level · {d["cefr"]}</div>
<p class="res-desc">{d["description"]}</p>
</div>
</div>
"""
    st.html(html_hero)

    col_left, col_right = st.columns(2, gap="medium")

    wp  = min(int(words / 3),   100)
    slp = min(int(avg_sl * 5),  100)
    irp = min(int(irreg * 8),   100)

    with col_left:
        html_left = f"""
<div class="res-card">
<div class="res-card-title">📊 Text Metrics</div>
<div class="stat-lbl">Total Words</div>
<div class="bar-bg"><div class="bar-fill" style="width:{wp}%;background:{d["gradient"]};"></div></div>
<div class="stat-val">{int(words)} words</div>
<div class="stat-lbl">Avg Words / Sentence</div>
<div class="bar-bg"><div class="bar-fill" style="width:{slp}%;background:{d["gradient"]};"></div></div>
<div class="stat-val">{avg_sl:.1f} words</div>
<div class="stat-lbl">Irregular Verbs Used</div>
<div class="bar-bg"><div class="bar-fill" style="width:{irp}%;background:{d["gradient"]};"></div></div>
<div class="stat-val">{int(irreg)} verbs</div>
</div>
"""
        st.html(html_left)

    with col_right:
        labels = ["Beginner", "Intermediate", "Advanced"]
        rows_html = ""
        for i in range(3):
            p       = int(proba[i] * 100)
            bar_bg  = d["gradient"] if i == pred else "rgba(255,255,255,0.1)"
            v_color = d["color"]    if i == pred else "#50509a"
            weight  = "700"         if i == pred else "400"
            rows_html += f"""
<div class="prob-row">
<span class="prob-lbl">{labels[i]}</span>
<div class="prob-bg"><div class="prob-fill" style="width:{p}%;background:{bar_bg};"></div></div>
<span class="prob-val" style="color:{v_color};font-weight:{weight};">{p}%</span>
</div>
"""
        
        conf_color = d["color"]
        html_right = f"""
<div class="res-card">
<div class="res-card-title">🤖 Model Confidence</div>
<p style="font-size:.83rem;color:#60609a;margin-bottom:18px;">Neural network confidence across all 3 levels:</p>
{rows_html}
<div style="margin-top:16px;padding-top:14px;border-top:1px solid rgba(255,255,255,.06);font-size:.85rem;color:{conf_color};font-weight:600;">
✓ {conf}% confident in this classification
</div>
</div>
"""
        st.html(html_right)

    html_tips_start = """
<div style='height:36px'></div>
<div style='font-size:1.2rem;font-weight:700;color:#fff;margin-bottom:14px;display:flex;justify-content:center;'>
🗺️ How to improve from here
</div>
<div style='display:flex;flex-direction:column;align-items:center;'>
"""
    st.html(html_tips_start)

    for title, body in d["tips"]:
        html_tip = f"""
<div class="tip-card" style="width:100%;max-width:800px;">
<div class="tip-title">{title}</div>
<div class="tip-body">{body}</div>
</div>
"""
        st.html(html_tip)

    st.html("</div><div style='height:44px'></div>")

    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        if st.button("🔄  Take the test again", key="btn_restart", use_container_width=True):
            for k, v in DEFAULTS.items():
                st.session_state[k] = v
            st.rerun()


# ─────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────
screen = st.session_state.screen
if screen == "landing":
    render_landing()
elif screen == "quiz":
    if mlp_model is None:
        st.error("❌ Dataset not found. Run `nlp_pipeline.py` first.")
    else:
        render_quiz()
elif screen == "results":
    render_results()
