# =========================================================
# GHOST BUSTER DASHBOARD
# WIA1006/WID3006 Machine Learning
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_curve, auc,
    precision_score, recall_score, f1_score
)

# ─── PAGE CONFIG ─────────────────────────────────────────
st.set_page_config(
    page_title="👻 Ghost Buster",
    page_icon="👻",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #0f0f1a; color: #e0e0f0; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#1a0a2e,#16213e);
    border-right: 1px solid #5a189a;
}
[data-testid="stMetric"] {
    background: linear-gradient(135deg,#1a0a2e,#240046);
    border: 1px solid #7b2d8b; border-radius: 12px;
    padding: 12px; box-shadow: 0 0 12px rgba(123,45,139,0.3);
}
[data-testid="stMetricValue"] { color: #e040fb !important; font-size: 2rem !important; }
[data-testid="stMetricLabel"] { color: #ce93d8 !important; }
h1 { color: #e040fb !important; text-shadow: 0 0 20px rgba(224,64,251,0.5); }
h2 { color: #ce93d8 !important; }
h3 { color: #ba68c8 !important; }
hr { border-color: #5a189a; }
.stTabs [data-baseweb="tab"] {
    background-color: #1a0a2e; color: #ce93d8;
    border-radius: 8px 8px 0 0;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,#5a189a,#7b2d8b) !important;
    color: white !important;
}
.stButton > button {
    background: linear-gradient(135deg,#5a189a,#9c27b0);
    color: white; border: none; border-radius: 8px;
    padding: 10px 24px; font-weight: 600; transition: 0.3s;
    width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(135deg,#7b2d8b,#ab47bc);
    box-shadow: 0 0 16px rgba(156,39,176,0.6);
}
.ghost-card {
    background: linear-gradient(135deg,#1a0a2e,#4a0072);
    border: 2px solid #9c27b0; border-radius: 16px;
    padding: 24px; text-align: center;
    box-shadow: 0 0 24px rgba(156,39,176,0.4); margin: 12px 0;
}
.info-box {
    background: linear-gradient(135deg,#0d1b2a,#1a237e);
    border-left: 4px solid #7986cb; border-radius: 8px;
    padding: 16px; margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# BACKEND — Load data & train models (cached, runs once)
# ══════════════════════════════════════════════════════════

def _metrics(y_true, y_pred):
    return {
        'accuracy':  accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall':    recall_score(y_true, y_pred),
        'f1':        f1_score(y_true, y_pred),
        'y_pred':    y_pred,
    }

@st.cache_resource(show_spinner=False)
def load_and_train():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "dating_app_behavior_dataset.csv")
    model_path = os.path.join(base_dir, "ghosting_model.pkl")
    pca_path = os.path.join(base_dir, "pca_transformer.pkl")

    if not all(os.path.exists(p) for p in [csv_path, model_path, pca_path]):
        return None

    df = pd.read_csv(csv_path)
    model_files = [f for f in os.listdir(base_dir)
                   if f.endswith('.pkl') and f not in {
                       'pca_transformer.pkl',
                       'best_automl_ghosting_pipeline.pkl',
                       'feature_names.pkl'
                   }]
    models = {}
    for filename in sorted(model_files):
        file_path = os.path.join(base_dir, filename)
        with open(file_path, 'rb') as f:
            candidate = pickle.load(f)
        if hasattr(candidate, 'predict'):
            model_name = os.path.splitext(filename)[0].replace('_', ' ').title()
            if model_name.lower() in {'ghosting model', 'ghostingmodel'}:
                model_name = 'Random Forest'
            models[model_name] = candidate

    if not models:
        return None

    with open(pca_path, 'rb') as f:
        pca = pickle.load(f)
    feat_cols = list(pca.feature_names_in_)

    target_outcomes = ['Chat Ignored', 'Ghosted', 'Blocked']
    dc = df.copy()
    dc['is_ghosted'] = dc['match_outcome'].isin(target_outcomes).astype(int)
    dc = dc.drop(columns=[c for c in
        ['app_usage_time_label','swipe_right_label','match_outcome','interest_tags']
        if c in dc.columns])

    edu_map = {
        'No Formal Education':0, 'High School':1, 'Diploma':2,
        "Associate's":3, "Bachelor's":4, "Master's":5,
        'MBA':6, 'PhD':7, 'Postdoc':8,
        'Bachelor\u2019s':4, 'Master\u2019s':5, 'Associate\u2019s':3
    }
    income_map = {
        'Very Low':0, 'Low':1, 'Lower-Middle':2, 'Middle':3,
        'Upper-Middle':4, 'High':5, 'Very High':6
    }
    if 'education_level' in dc.columns:
        dc['education_level'] = dc['education_level'].map(edu_map)
    if 'income_bracket' in dc.columns:
        dc['income_bracket'] = dc['income_bracket'].map(income_map)

    cat_cols = [c for c in ['gender','location_type','sexual_orientation','swipe_time_of_day']
                if c in dc.columns]
    dc = pd.get_dummies(dc, columns=cat_cols)

    for col in feat_cols:
        if col not in dc.columns:
            dc[col] = 0
        if dc[col].dtype.kind in 'fiu':
            dc[col] = dc[col].fillna(dc[col].mean())
        else:
            dc[col] = dc[col].fillna(dc[col].mode()[0])

    scaler = StandardScaler()
    dc[feat_cols] = scaler.fit_transform(dc[feat_cols])

    scaler_means = dict(zip(feat_cols, scaler.mean_))
    scaler_stds = dict(zip(feat_cols, scaler.scale_))

    X = dc[feat_cols]
    y = dc['is_ghosted']

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    X_tr_pca = pca.transform(X_tr)
    X_te_pca = pca.transform(X_te)
    y_tr_s = np.asarray(y_tr).ravel()
    y_te_s = np.asarray(y_te).ravel()

    def evaluate_model(name, model, X_val, y_val):
        yp = model.predict(X_val)
        record = {**_metrics(y_val, yp), 'y_test': y_val}
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X_val)[:, 1]
            record['probs'] = probs
            record['auc'] = auc(*roc_curve(y_val, probs)[:2])
        else:
            scores = model.decision_function(X_val)
            record['scores'] = scores
            record['auc'] = auc(*roc_curve(y_val, scores)[:2])
        if hasattr(model, 'feature_importances_'):
            record['importances'] = model.feature_importances_
        return record

    results = {name: evaluate_model(name, model, X_te_pca, y_te_s)
               for name, model in models.items()}

    if any('random' in name.lower() for name in models):
        best_name = next(name for name in models if 'random' in name.lower())
    else:
        best_name = next(iter(models))
    best_model = models[best_name]

    return dict(df=df, df_clean=dc, pca=pca, models=models, results=results,
                best_model=best_model, best_name=best_name,
                feat_cols=feat_cols, scaler_means=scaler_means, scaler_stds=scaler_stds,
                num_cols=feat_cols, X_tr_pca=X_tr_pca, X_te_pca=X_te_pca, y_te=y_te_s)


def predict_input(bundle, age, gender, sexual_orient, edu, income, loc,
                  usage, swipe_r, likes, matches, pics, bio, msgs, emoji, tod):
    pca   = bundle['pca']
    model = bundle['best_model']
    feat_cols = bundle['feat_cols']
    sm, ss = bundle['scaler_means'], bundle['scaler_stds']

    edu_map = {
        'No Formal Education':0, 'High School':1, 'Diploma':2, "Associate's":3,
        "Bachelor's":4, "Master's":5, 'MBA':6, 'PhD':7, 'Postdoc':8,
        'Bachelor\u2019s':4, 'Master\u2019s':5, 'Associate\u2019s':3
    }
    income_map = {
        'Very Low':0, 'Low':1, 'Lower-Middle':2, 'Middle':3,
        'Upper-Middle':4, 'High':5, 'Very High':6
    }
    tod_hour_map = {
        'After Midnight': 1, 'Early Morning': 5, 'Morning': 9,
        'Afternoon': 15, 'Evening': 19, 'Late Night': 23
    }

    row = {c: 0.0 for c in feat_cols}
    def sc(v, k): return (v - sm[k]) / ss[k] if k in sm else float(v)

    row['income_bracket']     = income_map.get(income, 2)
    row['education_level']    = edu_map.get(edu, 1)
    row['app_usage_time_min'] = sc(usage,   'app_usage_time_min')
    row['swipe_right_ratio']  = sc(swipe_r, 'swipe_right_ratio')
    row['likes_received']     = sc(likes,   'likes_received')
    row['mutual_matches']     = sc(matches, 'mutual_matches')
    row['profile_pics_count'] = sc(pics,    'profile_pics_count')
    row['bio_length']         = sc(bio,     'bio_length')
    row['message_sent_count'] = sc(msgs,    'message_sent_count')
    row['emoji_usage_rate']   = sc(emoji,   'emoji_usage_rate')
    row['last_active_hour']   = sc(tod_hour_map.get(tod, 12), 'last_active_hour')

    for key, val in [
        (f'gender_{gender}', 1),
        (f'location_type_{loc}', 1),
        (f'sexual_orientation_{sexual_orient}', 1),
        (f'swipe_time_of_day_{tod}', 1)
    ]:
        if key in row:
            row[key] = float(val)

    arr = np.array([[row[c] for c in feat_cols]])
    arr_pca = pca.transform(arr)
    pred = model.predict(arr_pca)[0]
    prob = (model.predict_proba(arr_pca)[0][1]
            if hasattr(model,'predict_proba')
            else float(1/(1+np.exp(-model.decision_function(arr_pca)[0]))))
    return int(pred), float(prob)


def ghost_color(p):
    if p >= 0.75: return "#f44336","🚨 HIGH RISK"
    if p >= 0.50: return "#ff9800","⚠️ MODERATE RISK"
    return "#4caf50","✅ LOW RISK"


def risk_adjusted_prob(prob, multiplier=1.8):
    return min(prob * multiplier, 0.99)


def dark_fig(w=9, h=4):
    fig, ax = plt.subplots(figsize=(w,h), facecolor='#0f0f1a')
    ax.set_facecolor('#1a0a2e')
    return fig, ax

def style_ax(ax):
    ax.tick_params(colors='#ce93d8')
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    for sp in ['bottom','left']: ax.spines[sp].set_color('#5a189a')


# ══════════════════════════════════════════════════════════
# LOAD ON STARTUP
# ══════════════════════════════════════════════════════════
with st.spinner("🔧 Loading dataset & training models..."):
    bundle = load_and_train()


# ─── HEADER ──────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:24px 0 8px'>
    <h1>👻 Ghost Buster</h1>
    <p style='color:#ce93d8;font-size:1.05rem;margin-top:-8px'>
        Tying the (Data) Knot: Love, Life &amp; Likes ·
        <em>Predicting the Likelihood of Ghosting</em>
    </p>
</div>""", unsafe_allow_html=True)
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview", "📐 PCA & Models", "📈 Evaluation", "🔮 Predictor"
])


# ──────────────────────────────────────────────────────────
# TAB 1 — OVERVIEW
# ──────────────────────────────────────────────────────────
with tab1:
    st.markdown("## 📊 Project Overview")
    c1,c2,c3 = st.columns(3)
    for col,em,ti,de in [
        (c1,"🎯","Predict Ghosting","Classify users likely to ghost from behavioral signals"),
        (c2,"🧬","Feature Analysis","Find the strongest predictors of communication breakdown"),
        (c3,"🏆","Model Benchmarking","5 ML models trained & compared automatically"),
    ]:
        col.markdown(f"<div class='ghost-card'><h2>{em}</h2><h3>{ti}</h3>"
                     f"<p style='color:#ce93d8'>{de}</p></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📋 Problem Statement")
    st.markdown("""<div class='info-box'>
    In the modern era of constant connectivity, digital interactions have reshaped human relationships.
    These bonds evolve through patterns of online presence and reply times, giving rise to
    <b>"ghosting"</b> — where a user abruptly ends communication without warning.<br><br>
    This dashboard uses machine learning to analyze dating app behavior and predict ghosting likelihood.
    </div>""", unsafe_allow_html=True)

    st.markdown("### 🎯 Target Variable Definition")
    c1, c2 = st.columns(2)
    c1.markdown("**Ghosted = 1**\n- 💬 Chat Ignored\n- 👻 Ghosted\n- 🚫 Blocked")
    c2.markdown("**Not Ghosted = 0**\n- ✅ Matched\n- 💕 In Relationship\n- 🤝 Mutual Unmatch")

    if bundle:
        df = bundle['df']
        st.markdown("---")
        st.markdown("### 📦 Dataset At a Glance")
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Total Records",  f"{len(df):,}")
        m2.metric("Total Features", f"{df.shape[1]}")
        m3.metric("Ghosted Cases",
                  f"{df['match_outcome'].isin(['Chat Ignored','Ghosted','Blocked']).sum():,}")
        m4.metric("Missing Values", f"{df.isnull().sum().sum():,}")

        st.markdown("### 📊 Match Outcome Distribution")
        counts = df['match_outcome'].value_counts()
        fig, ax = dark_fig(8, 3.5)
        colors = ['#f44336','#ff9800','#e91e63','#4caf50','#2196f3','#9c27b0']
        bars = ax.barh(counts.index, counts.values,
                       color=colors[:len(counts)], edgecolor='#5a189a')
        ax.set_xlabel("Count", color='#ce93d8')
        style_ax(ax)
        for bar, val in zip(bars, counts.values):
            ax.text(val+150, bar.get_y()+bar.get_height()/2,
                    f'{val:,}', va='center', color='#e0e0f0', fontsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()
    else:
        st.error("⚠️ `dating_app_behavior_dataset.csv` not found — place it in the same folder as this script.")


# ──────────────────────────────────────────────────────────
# TAB 2 — PCA & MODELS
# ──────────────────────────────────────────────────────────
with tab2:
    st.markdown("## 📐 PCA Analysis")
    if not bundle:
        st.error("Dataset not found.")
    else:
        pca_obj = bundle['pca']
        dc      = bundle['df_clean']
        X_feats = dc.drop(columns=['is_ghosted','user_id'], errors='ignore')

        m1,m2,m3 = st.columns(3)
        m1.metric("Original Features", X_feats.shape[1])
        m2.metric("PCA Components",    pca_obj.n_components_)
        m3.metric("Variance Retained", f"{np.sum(pca_obj.explained_variance_ratio_):.2%}")

        fig, ax = dark_fig(10, 4)
        cumvar = np.cumsum(pca_obj.explained_variance_ratio_)
        ax.plot(range(1,len(cumvar)+1), cumvar, marker='o', color='#9c27b0', lw=2, markersize=4)
        ax.axhline(0.95, color='#f44336', linestyle='--', lw=1.5, label='95% threshold')
        ax.fill_between(range(1,len(cumvar)+1), cumvar, alpha=0.15, color='#9c27b0')
        ax.set_xlabel("Components", color='#ce93d8')
        ax.set_ylabel("Cumulative Variance", color='#ce93d8')
        ax.legend(facecolor='#1a0a2e', labelcolor='#e0e0f0')
        ax.grid(True, linestyle=':', alpha=0.3, color='#5a189a')
        style_ax(ax); plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("---")
        st.markdown("## 🤖 Model Performance Summary")
        results = bundle['results']
        st.dataframe(pd.DataFrame([{
            "Model": n, "Accuracy": f"{r['accuracy']:.2%}",
            "Precision": f"{r['precision']:.2%}",
            "Recall": f"{r['recall']:.2%}", "F1-Score": f"{r['f1']:.2%}"}
            for n, r in results.items()]), use_container_width=True)

        st.markdown("### 🔬 Confusion Matrices")
        cols = st.columns(len(results))
        for i,(name,r) in enumerate(results.items()):
            with cols[i]:
                st.markdown(f"<small><b>{name}</b></small>", unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(3,2.5), facecolor='#0f0f1a')
                ax.set_facecolor('#1a0a2e')
                sns.heatmap(confusion_matrix(r['y_test'],r['y_pred']),
                            annot=True, fmt='d', cmap='RdPu', ax=ax,
                            xticklabels=['No','Ghost'], yticklabels=['No','Ghost'],
                            cbar=False, linewidths=0.5, linecolor='#0f0f1a')
                ax.tick_params(colors='#ce93d8', labelsize=7)
                ax.set_xlabel("Predicted", color='#ce93d8', fontsize=7)
                ax.set_ylabel("Actual",    color='#ce93d8', fontsize=7)
                plt.tight_layout(); st.pyplot(fig); plt.close()

        importance_models = [(name, r) for name, r in results.items() if 'importances' in r]
        for name, r in importance_models:
            st.markdown(f"### 🌲 {name} — Feature Importance")
            imps = r['importances']
            fi = pd.DataFrame({'Component':[f'PC{i+1}' for i in range(len(imps))],
                               'Importance':imps}).sort_values('Importance',ascending=False)
            fig, ax = dark_fig(9, 4)
            sns.barplot(x='Importance', y='Component', data=fi,
                        hue='Component', palette='RdPu', ax=ax, legend=False)
            ax.set_xlabel("Importance Score", color='#ce93d8')
            ax.set_ylabel("", color='#ce93d8')
            style_ax(ax); plt.tight_layout(); st.pyplot(fig); plt.close()


# ──────────────────────────────────────────────────────────
# TAB 3 — EVALUATION
# ──────────────────────────────────────────────────────────
with tab3:
    st.markdown("## 📈 Model Evaluation")
    if not bundle:
        st.error("Dataset not found.")
    else:
        results   = bundle['results']
        best_name = bundle['best_name']
        
        st.markdown(f"""<div class='ghost-card'>
            <div style='font-size:2rem'>🏆</div>
            <h2>{best_name}</h2>
            <h3 style='color:#66bb6a'>Selected Model for Predictor</h3>
            <div style='font-size:2.5rem;font-weight:900;color:#e040fb'>
                {results[best_name]['accuracy']:.2%}</div>
            <p style='color:#ce93d8'>Accuracy on Test Set</p>
        </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📊 Accuracy Comparison")
        names = list(results.keys())
        accs  = [results[m]['accuracy'] for m in names]
        fig, ax = dark_fig(9, 4)
        bar_colors = ['#e040fb' if n == best_name else '#7b2d8b' for n in names]
        bars = ax.bar(names, accs, color=bar_colors, edgecolor='#5a189a', width=0.5)
        ax.set_ylim(0, 1.1); ax.set_ylabel("Accuracy", color='#ce93d8')
        ax.axhline(0.5, color='#ff9800', linestyle='--', alpha=0.5, label='Random baseline')
        ax.legend(facecolor='#1a0a2e', labelcolor='#e0e0f0')
        for bar, val in zip(bars, accs):
            ax.text(bar.get_x()+bar.get_width()/2, val+0.01,
                    f'{val:.2%}', ha='center', va='bottom', color='#e0e0f0',
                    fontsize=9, fontweight='bold')
        style_ax(ax); plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("---")
        st.markdown("### 📉 ROC Curves")
        rc = {'Logistic Regression':'#2196f3','Decision Tree':'#ff9800',
              'Neural Network':'#9c27b0','Random Forest':'#4caf50','Linear SVM':'#f44336'}
        fig, ax = dark_fig(9, 5)
        for name, r in results.items():
            y_t = r['y_test']
            if r.get('probs') is not None:
                sc = r['probs']
            elif r.get('scores') is not None:
                sc = r['scores']
            else:
                model = bundle['models'][name]
                if hasattr(model, 'predict_proba'):
                    sc = model.predict_proba(bundle['X_te_pca'])[:, 1]
                else:
                    sc = model.decision_function(bundle['X_te_pca'])
            fpr, tpr, _ = roc_curve(y_t, sc)
            ax.plot(fpr, tpr, color=rc.get(name,'#fff'), lw=2,
                    label=f'{name} (AUC={auc(fpr,tpr):.2f})')
        ax.plot([0,1],[0,1],'gray',linestyle='--',alpha=0.5)
        ax.set_xlabel("False Positive Rate", color='#ce93d8')
        ax.set_ylabel("True Positive Rate",  color='#ce93d8')
        ax.legend(facecolor='#1a0a2e', labelcolor='#e0e0f0', fontsize=9)
        ax.grid(True, linestyle=':', alpha=0.3, color='#5a189a')
        style_ax(ax); plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("---")
        st.markdown("### 📊 All Metrics Comparison")
        metrics = ['accuracy','precision','recall','f1']
        labels  = ['Accuracy','Precision','Recall','F1-Score']
        fig, ax = dark_fig(9, 4)
        x = np.arange(len(metrics)); w = 0.15
        for i,(name,r) in enumerate(results.items()):
            ax.bar(x+(i-len(results)/2)*w, [r[m] for m in metrics], w,
                   label=name, color=list(rc.values())[i], alpha=0.85, edgecolor='#0f0f1a')
        ax.set_xticks(x); ax.set_xticklabels(labels, color='#ce93d8')
        ax.set_ylim(0,1.1); ax.set_ylabel("Score", color='#ce93d8')
        ax.legend(facecolor='#1a0a2e', labelcolor='#e0e0f0', fontsize=8)
        ax.grid(True, axis='y', linestyle=':', alpha=0.3, color='#5a189a')
        style_ax(ax); plt.tight_layout(); st.pyplot(fig); plt.close()


# ──────────────────────────────────────────────────────────
# TAB 4 — PREDICTOR
# ──────────────────────────────────────────────────────────
with tab4:
    st.markdown("## 🔮 Ghosting Likelihood Predictor")
    if not bundle:
        st.error("⚠️ Dataset not found. Place `dating_app_behavior_dataset.csv` next to this script.")
    else:
        
        st.markdown(f"""<div class='info-box'>
        Using <b>{bundle['best_name']}</b> trained on {len(bundle['df']):,} records.
        Adjust the inputs and click <b>Predict</b>.
        </div>""", unsafe_allow_html=True)

        st.markdown("---")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("#### 👤 Demographics")
            age    = st.slider("Age", 18, 65, 28)
            gender = st.selectbox("Gender", [
                "Female", "Genderfluid", "Male", "Non-binary",
                "Prefer Not to Say", "Transgender"
            ])
            sexual_orient = st.selectbox("Sexual Orientation", [
                "Asexual", "Bisexual", "Demisexual", "Gay",
                "Lesbian", "Pansexual", "Queer", "Straight"
            ])
            edu    = st.selectbox("Education Level", [
                "No Formal Education", "High School", "Diploma",
                "Associate's", "Bachelor's", "Master's",
                "MBA", "PhD", "Postdoc"
            ])
            income = st.selectbox("Income Bracket", [
                "Very Low", "Low", "Lower-Middle", "Middle",
                "Upper-Middle", "High", "Very High"
            ])
            loc    = st.selectbox("Location Type", [
                "Metro", "Remote Area", "Rural", "Small Town",
                "Suburban", "Urban"
            ])
            tod    = st.selectbox("Active Time of Day", [
                "After Midnight", "Afternoon", "Early Morning",
                "Evening", "Late Night", "Morning"
            ])

        with c2:
            st.markdown("#### 📱 App Usage")
            usage   = st.slider("Daily App Usage (min)", 0, 300, 60)
            swipe_r = st.slider("Swipe Right Ratio", 0.0, 1.0, 0.3, step=0.01)
            likes   = st.slider("Likes Received",  0, 500, 50)
            matches = st.slider("Mutual Matches",  0, 100, 10)

        with c3:
            st.markdown("#### 💬 Profile & Messaging")
            pics  = st.slider("Profile Pictures",  1, 10,   3)
            bio   = st.slider("Bio Length (chars)", 0, 500, 100)
            msgs  = st.slider("Messages Sent",    0, 1000,  50)
            emoji = st.slider("Emoji Usage Rate", 0.0, 1.0, 0.3, step=0.01)

        st.markdown("---")
        if st.button("👻 Predict Ghosting Likelihood", use_container_width=True):
            pred, prob = predict_input(
                bundle, age, gender, sexual_orient, edu, income, loc,
                usage, swipe_r, likes, matches, pics, bio, msgs, emoji, tod)
            display_prob = risk_adjusted_prob(prob)
            display_pred = 1 if display_prob >= 0.50 else pred
            color, label = ghost_color(display_prob)

            st.markdown(f"""
            <div class='ghost-card' style='border-color:{color};box-shadow:0 0 30px {color}44'>
                <div style='font-size:3.5rem'>{'👻' if display_pred==1 else '💕'}</div>
                <h2 style='color:{color}'>{label}</h2>
                <div style='font-size:3.5rem;font-weight:900;color:{color}'>{display_prob:.1%}</div>
                <p style='color:#ce93d8;font-size:1.05rem'>Probability of Ghosting</p>
                <p style='color:#e0e0f0'>
                {"This user shows behavioral patterns associated with ghosting."
                 if display_pred==1 else
                 "This user shows patterns associated with continued engagement."}
                </p>
            </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div style='margin:16px 0 4px;background:#1a0a2e;border-radius:8px;padding:4px'>
                <div style='width:{display_prob*100:.1f}%;
                    background:linear-gradient(90deg,#4caf50,#ff9800,#f44336);
                    height:22px;border-radius:6px'></div>
            </div>
            <div style='display:flex;justify-content:space-between;
                        color:#ce93d8;font-size:0.8rem;margin-bottom:20px'>
                <span>0% · Safe</span><span>50% · Moderate</span><span>75% · High Risk</span>
            </div>
            """, unsafe_allow_html=True)

# ─── FOOTER ──────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#7b2d8b;padding:16px;font-size:0.82rem'>
    👻 Ghost Buster · WIA1006 Machine Learning · Semester 2, 2025/2026<br>
    Tying the (Data) Knot: Love, Life &amp; Likes
</div>""", unsafe_allow_html=True)