"""
MediPredict — Application de sensibilisation au risque de diabète de type 2
Développé pour SantéCo (mutuelle fictive)

⚠️ Cet outil est un outil de sensibilisation. Il ne constitue pas un avis médical.
Aucune donnée utilisateur n'est stockée (traitement en mémoire uniquement).
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

# ── Import des modules du projet ──
from src.predict import charger_modele, valider_entrees, predire, PLAGES_VALIDES, TOOLTIPS
from src.explain import (
    calculer_shap, generer_explication_naturelle,
    generer_recommandations, generer_waterfall_plot, NOMS_FRANCAIS
)
from src.visualize import (
    plot_distribution_classes, plot_distributions_features,
    plot_correlation_matrix, plot_confusion_matrix,
    plot_roc_curve, plot_comparaison_profil
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Configuration de la page ──
st.set_page_config(
    page_title="MediPredict — Risque de diabète",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS personnalisé pour l'accessibilité ──
st.markdown("""
<style>
    /* Taille de police minimum 16px */
    html, body, [class*="css"] {
        font-size: 16px !important;
    }
    /* Contraste amélioré */
    .stMarkdown p, .stMarkdown li {
        color: #1a1a1a !important;
        font-size: 16px !important;
        line-height: 1.6 !important;
    }
    h1 { color: #1a3c5e !important; }
    h2 { color: #1a3c5e !important; }
    h3 { color: #2c5f8a !important; }

    /* Jauge de risque */
    .risk-gauge {
        padding: 24px;
        border-radius: 12px;
        text-align: center;
        margin: 16px 0;
        border: 3px solid;
    }
    .risk-gauge h2 {
        font-size: 28px !important;
        margin-bottom: 8px;
    }
    .risk-low {
        background-color: #d5f5e3;
        border-color: #27ae60;
        color: #1a5e2d !important;
    }
    .risk-low h2, .risk-low p { color: #1a5e2d !important; }
    .risk-medium {
        background-color: #fdebd0;
        border-color: #f39c12;
        color: #7d5a00 !important;
    }
    .risk-medium h2, .risk-medium p { color: #7d5a00 !important; }
    .risk-high {
        background-color: #fadbd8;
        border-color: #e74c3c;
        color: #7b1a1a !important;
    }
    .risk-high h2, .risk-high p { color: #7b1a1a !important; }

    /* Mention légale */
    .legal-notice {
        background-color: #eaf2f8;
        border-left: 5px solid #2980b9;
        padding: 16px 20px;
        border-radius: 0 8px 8px 0;
        margin: 16px 0;
        color: #1a3c5e !important;
        font-size: 16px !important;
    }

    /* Consentement */
    .consent-box {
        background-color: #fef9e7;
        border: 2px solid #f1c40f;
        padding: 20px;
        border-radius: 10px;
        margin: 16px 0;
    }

    /* Footer */
    .footer-text {
        text-align: center;
        color: #666 !important;
        font-size: 14px !important;
        padding: 20px 0;
        border-top: 1px solid #ddd;
        margin-top: 40px;
    }
</style>
""", unsafe_allow_html=True)

# ── Chargement des ressources (mis en cache) ──
@st.cache_resource
def load_model():
    return charger_modele()

@st.cache_data
def load_data():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "diabetes.csv")
    return pd.read_csv(data_path)

model, scaler, feature_names = load_model()
df = load_data()


# ══════════════════════════════════════════════════════════════
# SIDEBAR — Navigation
# ══════════════════════════════════════════════════════════════
st.sidebar.image("logo.png", width=80)
st.sidebar.title("MediPredict")
st.sidebar.markdown("*Sensibilisation au risque de diabète*")
st.sidebar.markdown("---")

pages = {
    "🏠 Accueil": "accueil",
    "📋 Mon profil de risque": "profil",
    "🔍 Comprendre ma prédiction": "comprendre",
    "📊 Explorer les données": "explorer",
}
choix_page = st.sidebar.radio("Navigation", list(pages.keys()), label_visibility="collapsed")
page_active = pages[choix_page]

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div class="legal-notice" style="font-size:13px !important;">'
    "⚕️ <strong>Cet outil ne constitue pas un avis médical.</strong> "
    "En cas de doute, consultez un professionnel de santé."
    "</div>",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════
# PAGE 1 — ACCUEIL
# ══════════════════════════════════════════════════════════════
if page_active == "accueil":
    st.title("🏥 MediPredict")
    st.subheader("Outil de sensibilisation au risque de diabète de type 2")

    st.markdown("---")

    st.markdown("""
    Bienvenue sur **MediPredict**, un outil développé par **SantéCo** (mutuelle fictive)
    pour vous aider à mieux comprendre votre profil de risque face au diabète de type 2.

    En quelques clics, vous pouvez :
    - 📋 **Renseigner vos indicateurs de santé** de manière anonyme
    - 📊 **Obtenir une estimation de votre niveau de risque** (faible, modéré, élevé)
    - 🔍 **Comprendre les facteurs** qui influencent cette estimation
    - 💡 **Recevoir des recommandations** génériques de prévention
    """)

    # Mention légale obligatoire
    st.markdown(
        '<div class="legal-notice">'
        "⚠️ <strong>Mention légale importante</strong><br><br>"
        "Cet outil est un <strong>outil de sensibilisation</strong>. "
        "Il <strong>ne constitue pas un avis médical</strong> et ne remplace en aucun cas "
        "la consultation d'un professionnel de santé. Les résultats fournis sont des "
        "estimations basées sur un modèle statistique et ne doivent pas être interprétés "
        "comme un diagnostic. <strong>En cas de doute, consultez un médecin.</strong>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("#### 🔒 Politique de confidentialité")
    st.markdown("""
    Vos données sont traitées **exclusivement en mémoire** pendant votre session.
    **Aucune information personnelle n'est stockée, enregistrée ou transmise** à un tiers.
    Les données que vous saisissez sont supprimées dès la fermeture de votre navigateur.
    Ce traitement repose sur votre **consentement explicite** (Art. 6.1.a du RGPD).
    """)

    # Bouton de consentement
    st.markdown("#### ✅ Consentement")
    st.markdown(
        '<div class="consent-box">'
        "En cochant cette case, vous confirmez avoir pris connaissance de la mention légale "
        "ci-dessus et consentez au traitement temporaire de vos données de santé anonymes "
        "à des fins de sensibilisation uniquement."
        "</div>",
        unsafe_allow_html=True,
    )

    consentement = st.checkbox(
        "J'ai lu et j'accepte les conditions d'utilisation",
        key="consentement",
    )

    if consentement:
        st.session_state["consentement_donne"] = True
        st.success("✅ Consentement enregistré. Vous pouvez maintenant accéder aux fonctionnalités via le menu à gauche.")
    else:
        st.session_state["consentement_donne"] = False
        st.info("👆 Veuillez cocher la case ci-dessus pour accéder aux fonctionnalités.")


# ══════════════════════════════════════════════════════════════
# PAGE 2 — MON PROFIL DE RISQUE
# ══════════════════════════════════════════════════════════════
elif page_active == "profil":
    st.title("📋 Mon profil de risque")

    if not st.session_state.get("consentement_donne", False):
        st.warning("⚠️ Vous devez d'abord accepter les conditions d'utilisation sur la page **Accueil** avant d'accéder à cette fonctionnalité.")
        st.stop()

    st.markdown(
        '<div class="legal-notice">'
        "⚕️ Rappel : cet outil est un outil de sensibilisation. "
        "Il ne constitue pas un avis médical."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("### Renseignez vos indicateurs de santé")
    st.markdown("*Toutes les données restent en mémoire et ne sont jamais stockées.*")

    # ── Formulaire de saisie ──
    col1, col2 = st.columns(2)

    with col1:
        # Gestion spéciale de Pregnancies
        pregnancies_applicable = st.radio(
            "Le nombre de grossesses vous concerne-t-il ?",
            options=["Oui", "Non applicable"],
            help=TOOLTIPS["Pregnancies"],
            horizontal=True,
        )
        if pregnancies_applicable == "Oui":
            pregnancies = st.number_input(
                "Nombre de grossesses",
                min_value=0, max_value=20, value=0, step=1,
                help=TOOLTIPS["Pregnancies"],
            )
        else:
            pregnancies = int(df["Pregnancies"].median())
            st.info(f"Valeur par défaut utilisée : {pregnancies} (médiane du dataset)")

        glucose = st.number_input(
            "Taux de glucose (mg/dL)",
            min_value=40, max_value=250, value=110, step=1,
            help=TOOLTIPS["Glucose"],
        )

        blood_pressure = st.number_input(
            "Pression artérielle diastolique (mm Hg)",
            min_value=20, max_value=140, value=70, step=1,
            help=TOOLTIPS["BloodPressure"],
        )

        skin_thickness = st.number_input(
            "Épaisseur du pli cutané au triceps (mm)",
            min_value=5, max_value=100, value=28, step=1,
            help=TOOLTIPS["SkinThickness"],
        )

    with col2:
        insulin = st.number_input(
            "Taux d'insuline sérique (µU/mL)",
            min_value=10.0, max_value=900.0, value=100.0, step=1.0,
            help=TOOLTIPS["Insulin"],
        )

        bmi = st.number_input(
            "Indice de masse corporelle (IMC)",
            min_value=10.0, max_value=70.0, value=25.0, step=0.1,
            help=TOOLTIPS["BMI"],
        )

        dpf = st.number_input(
            "Antécédents familiaux de diabète (score 0-2.5)",
            min_value=0.05, max_value=2.50, value=0.35, step=0.01,
            help=TOOLTIPS["DiabetesPedigreeFunction"],
        )

        age = st.number_input(
            "Âge (années)",
            min_value=18, max_value=100, value=30, step=1,
            help=TOOLTIPS["Age"],
        )

    # Assembler les données
    donnees = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": float(blood_pressure),
        "SkinThickness": skin_thickness,
        "Insulin": float(insulin),
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
    }

    st.markdown("---")

    # ── Bouton d'analyse ──
    if st.button("🔍 Analyser mon profil", type="primary", use_container_width=True):
        # Validation côté serveur
        erreurs = valider_entrees(donnees)
        if erreurs:
            for err in erreurs:
                st.error(f"❌ {err}")
        else:
            # Prédiction
            proba, niveau, couleur, emoji = predire(donnees, model, scaler, feature_names)

            # Stocker en session pour la page "Comprendre"
            st.session_state["derniere_prediction"] = {
                "donnees": donnees,
                "proba": proba,
                "niveau": niveau,
                "couleur": couleur,
                "emoji": emoji,
            }

            # ── Affichage de la jauge de risque ──
            if proba >= 0.6:
                css_class = "risk-high"
            elif proba >= 0.4:
                css_class = "risk-medium"
            else:
                css_class = "risk-low"

            st.markdown(
                f'<div class="risk-gauge {css_class}">'
                f"<h2>{emoji} {niveau}</h2>"
                f"<p>Votre estimation de risque de diabète de type 2 est : <strong>{niveau.lower()}</strong>.</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Avertissement contextuel
            if proba >= 0.6:
                st.markdown(
                    '<div class="legal-notice">'
                    "⚠️ <strong>Ce résultat ne constitue pas un diagnostic médical.</strong> "
                    "Il vous est recommandé de consulter un professionnel de santé "
                    "pour un bilan personnalisé."
                    "</div>",
                    unsafe_allow_html=True,
                )
            elif proba >= 0.4:
                st.info("ℹ️ Ce résultat suggère un risque modéré. Un suivi régulier auprès de votre médecin est conseillé.")
            else:
                st.success("✅ Votre profil ne présente pas de risque élevé. Continuez à maintenir un mode de vie sain.")

            st.markdown("---")
            st.markdown("👉 Consultez la page **🔍 Comprendre ma prédiction** pour voir les facteurs qui influencent ce résultat.")


# ══════════════════════════════════════════════════════════════
# PAGE 3 — COMPRENDRE MA PRÉDICTION
# ══════════════════════════════════════════════════════════════
elif page_active == "comprendre":
    st.title("🔍 Comprendre ma prédiction")

    if not st.session_state.get("consentement_donne", False):
        st.warning("⚠️ Vous devez d'abord accepter les conditions d'utilisation sur la page **Accueil**.")
        st.stop()

    pred = st.session_state.get("derniere_prediction", None)

    if pred is None:
        st.info("📋 Vous n'avez pas encore effectué d'analyse. Rendez-vous sur la page **📋 Mon profil de risque** pour commencer.")
        st.stop()

    donnees = pred["donnees"]
    proba = pred["proba"]
    niveau = pred["niveau"]
    emoji = pred["emoji"]

    # Rappel du résultat
    if proba >= 0.6:
        css_class = "risk-high"
    elif proba >= 0.4:
        css_class = "risk-medium"
    else:
        css_class = "risk-low"

    st.markdown(
        f'<div class="risk-gauge {css_class}">'
        f"<h2>{emoji} {niveau}</h2>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── SHAP ──
    st.markdown("### 📊 Quelles variables ont influencé ce résultat ?")
    st.markdown(
        "Le graphique ci-dessous montre la **contribution de chaque variable** à votre résultat. "
        "Les barres rouges **augmentent** le risque, les barres bleues le **diminuent**."
    )

    # Calcul SHAP
    df_input = pd.DataFrame([donnees], columns=feature_names)
    X_scaled = scaler.transform(df_input)

    with st.spinner("Calcul de l'explication en cours..."):
        shap_vals, base_val, explainer = calculer_shap(model, X_scaled, feature_names)

    # Waterfall plot
    fig_waterfall = generer_waterfall_plot(shap_vals, base_val, X_scaled[0], feature_names)
    st.pyplot(fig_waterfall)

    # ── Explication en langage naturel ──
    st.markdown("### 💬 Explication en langage naturel")
    explication = generer_explication_naturelle(shap_vals, feature_names, donnees, proba)
    st.markdown(
        f'<div class="legal-notice">{explication}</div>',
        unsafe_allow_html=True,
    )

    # ── Comparaison avec le dataset ──
    st.markdown("### 📈 Votre profil comparé à la population du dataset")
    st.markdown(
        "Les histogrammes ci-dessous montrent la distribution de chaque variable "
        "dans le dataset. La **ligne rouge en pointillés** représente votre valeur."
    )
    fig_comp = plot_comparaison_profil(donnees, df, feature_names)
    st.pyplot(fig_comp)

    # ── Recommandations ──
    st.markdown("### 💡 Recommandations génériques")
    st.markdown(
        '<div class="legal-notice">'
        "⚕️ Ces recommandations sont <strong>génériques et informatives</strong>. "
        "Elles ne remplacent pas un avis médical personnalisé."
        "</div>",
        unsafe_allow_html=True,
    )

    recommandations = generer_recommandations(shap_vals, feature_names)
    for reco in recommandations:
        icon = "⬆️" if reco["impact"] == "augmente" else "⬇️"
        with st.expander(f"{icon} {reco['variable']} — {reco['impact']} le risque"):
            st.markdown(reco["texte"])


# ══════════════════════════════════════════════════════════════
# PAGE 4 — EXPLORER LES DONNÉES
# ══════════════════════════════════════════════════════════════
elif page_active == "explorer":
    st.title("📊 Explorer les données")

    if not st.session_state.get("consentement_donne", False):
        st.warning("⚠️ Vous devez d'abord accepter les conditions d'utilisation sur la page **Accueil**.")
        st.stop()

    tab1, tab2, tab3 = st.tabs([
        "📈 Visualisations descriptives",
        "🎯 Performance du modèle",
        "🔎 Transparence",
    ])

    # ── Tab 1 : Visualisations descriptives ──
    with tab1:
        st.markdown("### Distribution de la variable cible")
        fig1 = plot_distribution_classes(df)
        st.pyplot(fig1)

        st.markdown("### Distributions des variables par classe")
        fig2 = plot_distributions_features(df)
        st.pyplot(fig2)

        st.markdown("### Matrice de corrélation")
        fig3 = plot_correlation_matrix(df)
        st.pyplot(fig3)

    # ── Tab 2 : Performance du modèle ──
    with tab2:
        st.markdown("### Évaluation du modèle retenu : Arbre de Décision")

        # Recréer le split pour les métriques
        X = df.drop("Outcome", axis=1)
        y = df["Outcome"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        scaler_eval = StandardScaler()
        X_train_sc = scaler_eval.fit_transform(X_train)
        X_test_sc = scaler_eval.transform(X_test)

        y_pred = model.predict(X_test_sc)
        y_proba = model.predict_proba(X_test_sc)[:, 1]

        # Métriques
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        col_m1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.1%}")
        col_m2.metric("Précision", f"{precision_score(y_test, y_pred):.1%}")
        col_m3.metric("Rappel", f"{recall_score(y_test, y_pred):.1%}")
        col_m4.metric("F1-score", f"{f1_score(y_test, y_pred):.1%}")
        col_m5.metric("AUC-ROC", f"{roc_auc_score(y_test, y_proba):.3f}")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Matrice de confusion")
            fig_cm = plot_confusion_matrix(y_test, y_pred)
            st.pyplot(fig_cm)
        with col_b:
            st.markdown("#### Courbe ROC")
            fig_roc = plot_roc_curve(y_test, y_proba)
            st.pyplot(fig_roc)

    # ── Tab 3 : Transparence ──
    with tab3:
        st.markdown("### 🔎 Transparence du modèle")

        st.markdown("""
        #### Modèle utilisé
        Le modèle retenu est un **Arbre de Décision** (`DecisionTreeClassifier`) avec les paramètres suivants :
        - Profondeur maximale : **5 niveaux**
        - Pondération des classes : **`balanced`** (compense le déséquilibre 65/35)
        - Graine aléatoire : **42** (reproductibilité)

        #### Pourquoi ce modèle ?
        L'Arbre de Décision a été choisi pour deux raisons principales :
        1. **Meilleur rappel** (92.6%) : il rate le moins de vrais cas de diabète, ce qui est crucial
           pour un outil de sensibilisation médicale.
        2. **Interprétabilité native** : on peut visualiser et suivre chaque règle de décision,
           ce qui est une exigence du RGPD (Art. 22) pour les traitements automatisés.
        """)

        st.markdown("#### ⚠️ Limites connues")
        st.markdown("""
        - **Biais de population** : le modèle est entraîné sur le dataset *Pima Indians*,
          composé exclusivement de femmes d'origine Pima (Arizona). Il n'est **pas représentatif
          de la population française**.
        - **Biais d'âge** : les performances sont moins bonnes pour les moins de 30 ans
          (précision de 56%) que pour les 30-50 ans (précision de 94%).
        - **Biais d'antécédents** : le modèle détecte mieux les patients avec de forts antécédents
          familiaux (rappel 100%) que ceux sans antécédents (rappel 85%).
        - **Variable Pregnancies** : non pertinente pour tous les profils.
          Une option "Non applicable" est proposée, remplacée par la valeur médiane.
        - **Précision modérée** (78%) : environ 1 alerte sur 5 est une fausse alerte.
        """)

        st.markdown("#### 📊 Dataset utilisé")
        st.markdown("""
        - **Source** : Pima Indians Diabetes Dataset (UCI / Kaggle)
        - **Taille** : 768 observations, 8 variables prédictives, 1 variable cible
        - **Classes** : 500 non-diabétiques (65.1%) / 268 diabétiques (34.9%)
        - **Aucune donnée personnelle réelle** n'est utilisée — ce dataset est public et anonyme.
        """)


# ── Footer ──
st.markdown(
    '<div class="footer-text">'
    "MediPredict — Projet développé à des fins pédagogiques pour SantéCo (fictive)<br>"
    "Cet outil ne constitue pas un avis médical."
    "</div>",
    unsafe_allow_html=True,
)
