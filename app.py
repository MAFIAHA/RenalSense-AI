import os

import streamlit as st
st.set_page_config(page_title="RenalSense AI", layout="wide", page_icon="🏥")

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    import os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(BASE_DIR, "kidney_model.pkl"), "rb") as f:
        return pickle.load(f)

data = load_model()
models        = data["models"]
model_metrics = data["model_metrics"]
imputer       = data["imputer"]
FEATURES      = data["features"]
LABELS        = data["feature_labels"]
RANGES        = data["normal_ranges"]
importances   = data["importances"]

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.result-card{padding:28px;border-radius:14px;color:white;text-align:center;margin-bottom:16px;}
.flag-box{padding:8px 14px;border-radius:8px;margin:4px 0;font-size:14px;}
.metric-row{display:flex;gap:12px;margin-bottom:12px;}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏥 RenalSense AI — Kidney Health Dashboard")
st.caption("Machine Learning Decision Support System for Chronic Kidney Disease (CKD) Detection")
st.write("---")

# ── Sidebar: All 14 Inputs ────────────────────────────────────────────────────
with st.sidebar:
    st.header("📋 Patient Input Panel")
    algo_name = st.selectbox("🤖 Select ML Model", list(models.keys()))
    st.divider()

    st.subheader("Basic Vitals")
    age  = st.slider("Age (years)",           1,    100,   45)
    bp   = st.slider("Blood Pressure (mm Hg)",50,   180,   70)

    st.subheader("Urine Analysis")
    sg   = st.select_slider("Specific Gravity",
                             options=[1.005,1.010,1.015,1.020,1.025], value=1.020)
    al   = st.select_slider("Albumin (0–5)",   options=[0,1,2,3,4,5], value=0)
    su   = st.select_slider("Sugar (0–5)",     options=[0,1,2,3,4,5], value=0)

    st.subheader("Blood Tests")
    bgr  = st.number_input("Blood Glucose Random (mg/dL)",  70,  500, 120)
    bu   = st.number_input("Blood Urea (mg/dL)",            10,  200,  40)
    sc   = st.number_input("Serum Creatinine (mg/dL)",    0.4,   15.0, 1.0, step=0.1)
    sod  = st.number_input("Sodium (mEq/L)",              100,   160,  138)
    pot  = st.number_input("Potassium (mEq/L)",           2.0,   8.0,  4.2, step=0.1)
    hemo = st.slider("Hemoglobin (g/dL)",    3.0,  18.0, 14.0)

    st.subheader("Cell Counts")
    pcv  = st.slider("Packed Cell Volume (%)",10,   55,   40)
    wc   = st.number_input("White Cell Count (cells/cumm)", 2200, 26400, 7800)
    rc   = st.number_input("Red Cell Count (millions/cmm)", 2.1,  8.0,   5.0, step=0.1)

# ── Prediction ────────────────────────────────────────────────────────────────
user_values = [age, bp, sg, al, su, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc]
user_input  = imputer.transform(np.array([user_values]))
model       = models[algo_name]
prediction  = model.predict(user_input)[0]
probability = model.predict_proba(user_input)[0]

# Abnormal flag check
def is_abnormal(feat, val):
    lo, hi = RANGES[feat]
    if lo == 0 and hi == 0:
        return val > 0        # albumin/sugar: abnormal if > 0
    return val < lo or val > hi

patient_dict = dict(zip(FEATURES, user_values))
abnormal = {f: v for f, v in patient_dict.items() if is_abnormal(f, v)}

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔬 Diagnosis", "📊 Patient Profile", "🤖 Model Comparison", "ℹ️ About"])

# ═══════════════════════════════════════════════════════
#  TAB 1 — DIAGNOSIS
# ═══════════════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns([1, 1.1], gap="large")

    with col1:
        st.subheader("Diagnostic Result")
        conf = probability[1] if prediction == 1 else probability[0]

        if prediction == 1:
            st.markdown(f"""
            <div class="result-card" style="background:#d9534f;">
                <h1 style="margin:0;">⚠️ AT RISK</h1>
                <p style="font-size:1.3rem;">CKD Indicators Detected</p>
                <hr style="border-color:rgba(255,255,255,0.4)">
                <h2>Confidence: {conf*100:.1f}%</h2>
            </div>""", unsafe_allow_html=True)
            st.error("**Clinical Note:** CKD biomarkers detected. Specialist referral strongly recommended.")
        else:
            st.markdown(f"""
            <div class="result-card" style="background:#2ecc71;">
                <h1 style="margin:0;">✅ HEALTHY</h1>
                <p style="font-size:1.3rem;">No CKD Indicators Found</p>
                <hr style="border-color:rgba(255,255,255,0.4)">
                <h2>Confidence: {conf*100:.1f}%</h2>
            </div>""", unsafe_allow_html=True)
            st.success("**Clinical Note:** Biomarkers are within normal physiological ranges.")

        # Confidence bar
        st.write("**Prediction Confidence**")
        col_h, col_r = st.columns(2)
        col_h.metric("Healthy Probability",  f"{probability[0]*100:.1f}%")
        col_r.metric("At Risk Probability",  f"{probability[1]*100:.1f}%")

        # Abnormal flags
        if abnormal:
            st.write("**⚠️ Flagged Abnormal Values**")
            for feat, val in abnormal.items():
                lo, hi = RANGES[feat]
                if lo == 0 and hi == 0:
                    note = f"Expected 0 — got {val}"
                else:
                    note = f"Normal: {lo}–{hi} | Patient: {val}"
                st.markdown(
                    f'<div class="flag-box" style="background:#fff3cd;border-left:4px solid #f0ad4e;">'
                    f'🔶 <b>{LABELS[feat]}</b> — {note}</div>',
                    unsafe_allow_html=True)
        else:
            st.success("All values within normal ranges ✓")

        with st.expander("🔧 Technical Model Details"):
            m = model_metrics[algo_name]
            st.write(f"**Algorithm:** {algo_name}")
            st.write(f"**5-Fold CV Accuracy:** {m['cv_accuracy']}% ± {m['cv_std']}%")
            st.write(f"**Validation Accuracy:** {m['val_accuracy']}%")
            st.write("**Dataset:** UCI Chronic Kidney Disease (280 train / 120 test samples)")
            st.write("**Features used:** 14 numeric biomarkers")

    with col2:
        st.subheader("Confidence Gauge")

        fig, ax = plt.subplots(figsize=(5, 2.8))
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
        # Background bar
        ax.barh(0.5, 1, height=0.35, color='#ecf0f1', align='center')
        color = '#d9534f' if prediction == 1 else '#2ecc71'
        ax.barh(0.5, conf, height=0.35, color=color, align='center')
        ax.text(conf/2, 0.5, f"{conf*100:.1f}%",
                ha='center', va='center', fontsize=16, fontweight='bold', color='white')
        label = "Risk Probability" if prediction == 1 else "Healthy Confidence"
        ax.set_title(label, fontsize=13)
        st.pyplot(fig); plt.close(fig)

        # Probability pie
        st.write("**Probability Distribution**")
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        ax2.pie([probability[0], probability[1]],
                labels=['Healthy', 'At Risk'],
                colors=['#2ecc71', '#d9534f'],
                autopct='%1.1f%%', startangle=90,
                wedgeprops={'edgecolor':'white', 'linewidth':1.5})
        ax2.set_title("Model Output Distribution")
        st.pyplot(fig2); plt.close(fig2)

# ═══════════════════════════════════════════════════════
#  TAB 2 — PATIENT PROFILE
# ═══════════════════════════════════════════════════════
with tab2:
    st.subheader("Patient Biomarker Analysis — Normal Range Comparison")

    display_feats = ['bp','hemo','pcv','bgr','bu','sc','sod','pot']
    disp_labels   = [LABELS[f] for f in display_feats]
    patient_vals  = [patient_dict[f] for f in display_feats]
    normal_mid    = []
    for f in display_feats:
        lo, hi = RANGES[f]
        normal_mid.append((lo+hi)/2 if lo != hi else 0)

    x     = np.arange(len(disp_labels))
    width = 0.35

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    bars_n = ax3.bar(x - width/2, normal_mid, width, label='Normal Midpoint', color='#3498db', alpha=0.7)
    bars_p = ax3.bar(x + width/2, patient_vals, width, label='Patient Value',
                     color=['#d9534f' if is_abnormal(f, patient_dict[f]) else '#2ecc71'
                            for f in display_feats])

    ax3.set_ylabel('Value', fontsize=11)
    ax3.set_title('Key Biomarkers: Patient vs Normal', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(disp_labels, rotation=20, ha='right', fontsize=9)
    ax3.legend(fontsize=10)
    ax3.grid(axis='y', alpha=0.3)
    normal_patch = mpatches.Patch(color='#2ecc71', label='Normal patient value')
    risk_patch   = mpatches.Patch(color='#d9534f', label='Abnormal patient value')
    ax3.legend(handles=[bars_n, normal_patch, risk_patch], fontsize=9)
    plt.tight_layout()
    st.pyplot(fig3); plt.close(fig3)

    # Summary table
    st.subheader("Full Biomarker Summary Table")
    rows = []
    for feat in FEATURES:
        val = patient_dict[feat]
        lo, hi = RANGES[feat]
        if lo == 0 and hi == 0:
            status = "⚠️ Abnormal" if val > 0 else "✅ Normal"
            normal_str = "0 (none)"
        else:
            status = "⚠️ Abnormal" if is_abnormal(feat, val) else "✅ Normal"
            normal_str = f"{lo} – {hi}"
        rows.append({"Biomarker": LABELS[feat], "Patient Value": val,
                     "Normal Range": normal_str, "Status": status})

    import pandas as pd
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════
#  TAB 3 — MODEL COMPARISON
# ═══════════════════════════════════════════════════════
with tab3:
    st.subheader("Model Performance Comparison")

    c1, c2, c3 = st.columns(3)
    cols = [c1, c2, c3]
    for i, (name, m) in enumerate(model_metrics.items()):
        with cols[i]:
            border = "3px solid #3498db" if name == algo_name else "1px solid #ddd"
            tag = " ← selected" if name == algo_name else ""
            st.markdown(f"**{name}{tag}**")
            st.metric("CV Accuracy",  f"{m['cv_accuracy']}%", delta=f"±{m['cv_std']}%")
            st.metric("Val Accuracy", f"{m['val_accuracy']}%")

    # Feature Importance
    st.divider()
    st.subheader("🎯 Feature Importance (Random Forest)")
    st.caption("Shows which biomarkers the model relies on most for CKD prediction")

    sorted_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    feat_names   = [LABELS[f] for f, _ in sorted_feats]
    feat_vals    = [v for _, v in sorted_feats]

    fig4, ax4 = plt.subplots(figsize=(8, 5))
    colors = ['#d9534f' if v > 0.10 else '#3498db' if v > 0.05 else '#bdc3c7'
              for v in feat_vals]
    bars = ax4.barh(feat_names[::-1], feat_vals[::-1], color=colors[::-1])
    ax4.set_xlabel('Importance Score', fontsize=11)
    ax4.set_title('Feature Importances for CKD Prediction', fontsize=13, fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)
    # Add value labels
    for bar, val in zip(bars, feat_vals[::-1]):
        ax4.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                 f'{val:.3f}', va='center', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig4); plt.close(fig4)

    # Confusion Matrix for selected model
    st.divider()
    st.subheader(f"Confusion Matrix — {algo_name}")
    cm = np.array(model_metrics[algo_name]['confusion_matrix'])
    fig5, ax5 = plt.subplots(figsize=(4, 3))
    im = ax5.imshow(cm, cmap='Blues')
    ax5.set_xticks([0,1]); ax5.set_yticks([0,1])
    ax5.set_xticklabels(['Healthy','CKD']); ax5.set_yticklabels(['Healthy','CKD'])
    ax5.set_xlabel('Predicted'); ax5.set_ylabel('Actual')
    ax5.set_title('Confusion Matrix (Validation Set)')
    for i in range(2):
        for j in range(2):
            ax5.text(j, i, str(cm[i,j]), ha='center', va='center',
                     fontsize=16, fontweight='bold',
                     color='white' if cm[i,j] > cm.max()/2 else 'black')
    plt.tight_layout()
    st.pyplot(fig5); plt.close(fig5)

# ═══════════════════════════════════════════════════════
#  TAB 4 — ABOUT
# ═══════════════════════════════════════════════════════
with tab4:
    st.subheader("About RenalSense AI")
    st.markdown("""
**RenalSense AI** is a Machine Learning-based Clinical Decision Support System (CDSS)
designed to assist healthcare professionals in the early detection of Chronic Kidney Disease (CKD).

---

### 📂 Dataset
- **Source:** UCI Machine Learning Repository — Chronic Kidney Disease Dataset
- **Training samples:** 280 patients | **Test samples:** 120 patients
- **Features used:** 14 numeric biomarkers (out of 25 available attributes)
- **Class distribution:** ~62% CKD, ~38% Not CKD

### 🤖 Models Implemented
| Model | Description |
|-------|-------------|
| **Random Forest** | Ensemble of 150 decision trees. Best overall accuracy (98.57%). |
| **Gradient Boosting** | Sequential boosting for maximum precision. |
| **SVM (RBF Kernel)** | Support Vector Machine with radial basis function for non-linear separation. |

### 🔬 Methodology
1. Data loading and cleaning (handling `?` and missing values via median imputation)
2. Feature engineering on 14 numeric biomarkers
3. 80/20 stratified train-validation split
4. 5-Fold cross-validation for all models
5. Prediction with confidence probability output

### ⚠️ Disclaimer
This tool is for **educational and research purposes only**.
It is not a substitute for professional medical diagnosis.
Always consult a qualified nephrologist for clinical decisions.
    """)
