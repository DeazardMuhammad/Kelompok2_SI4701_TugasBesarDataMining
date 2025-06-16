import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Dashboard Prediksi Pelanggan Loyal",
    page_icon="💰",
    layout="wide"
)

st.title("Customer Loyalty Prediction App")
st.write("Masukkan data pelanggan untuk memprediksi kemungkinan menjadi pelanggan loyal.")

st.markdown("---")
st.subheader("Evaluasi Model")

# 1. Load Model yang sudah dilatih sebelumnya
try:
    # Coba load model yang sudah diperbaiki terlebih dahulu
    try:
        model = joblib.load("model_tubes_loyalitas_fixed.pkl")
        st.success("✅ Model berhasil dimuat! (Model yang sudah diperbaiki)")
    except:
        # Jika tidak ada, gunakan model lama
        model = joblib.load("model_tubes_loyalitas.pkl")
        st.warning("✅ Model berhasil dimuat!")
    
    st.info("🔧 **Perbaikan yang telah dilakukan:**")
    st.info("• Menghapus fitur 'Response' untuk menghindari data leakage")
    st.info("• Menambahkan fitur 'Income' sebagai prediktor yang lebih baik")
    st.info("• Target variable sekarang berdasarkan kampanye sebelumnya saja")
    
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.info("Pastikan file model ada di direktori yang sama.")
    st.info("Jalankan 'python retrain_model.py' untuk membuat model yang diperbaiki.")
    st.stop()

# 2. Load dan persiapkan data sama seperti saat melatih model sebelumnya
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv("marketing_campaign.csv", sep='\t')
    
    # 2.1 Drop Column yang tidak diperlukan (sesuai notebook)
    df.drop(columns=[
        'ID',
        'Z_CostContact', 'Z_Revenue', 'NumDealsPurchases', 'NumWebPurchases', 
        'NumCatalogPurchases', 'NumStorePurchases', 'Recency', 'Kidhome', 'Teenhome',
        'NumWebVisitsMonth'  # Tambahkan ini sesuai notebook
    ], inplace=True)
    
    # Menambahkan column total spent dan total accepted campaign
    df['TotalSpent'] = df[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntSweetProducts', 'MntGoldProds']].sum(axis=1)
    
    # PERBAIKAN: Tambah kolom TotalAcceptedCmp TANPA Response (menghindari data leakage)
    cmp_cols = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']
    df['TotalAcceptedCmp'] = df[cmp_cols].sum(axis=1)
    
    # PERBAIKAN: Buat target variable Loyal berdasarkan kampanye sebelumnya saja
    df['Loyal'] = (df['TotalAcceptedCmp'] > 0).astype(int)
    
    # Handle missing values (sesuai notebook)
    df.dropna(subset=['Income'], inplace=True)
    
    return df

# Load data
df = load_and_preprocess_data()

# 2.3 Pisahkan data menjadi fitur dan target
features = [
    'TotalSpent',
    'Complain',
    'Income'  # Tambahkan Income sebagai fitur prediktif
]

target = 'Loyal'

X = df[features]
y = df[target]

# 2.4 Buat prediksi menggunakan model yang sudah dilatih
y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]

# 2.5 Hitung metrik evaluasi
acc = accuracy_score(y, y_pred)
prec = precision_score(y, y_pred)
rec = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)
roc_auc = roc_auc_score(y, y_prob)

# Tampilkan metrik evaluasi
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.success(f"Accuracy: **{acc:.3f}**")

with col2:
    st.info(f"Precision: **{prec:.3f}**")

with col3:
    st.warning(f"Recall: **{rec:.3f}**")

with col4:
    st.error(f"F1-Score: **{f1:.3f}**")

with col5:
    st.success(f"ROC AUC: **{roc_auc:.3f}**")

# Plot options
st.subheader("📊 Visualisasi Model")
plot_option = st.selectbox("Pilih grafik untuk ditampilkan:", ["Pilih", "ROC AUC Curve", "Confusion Matrix", "Feature Importance"])

if plot_option == "ROC AUC Curve":
    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(y, y_prob)
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}', linewidth=2)
    ax2.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)

elif plot_option == "Confusion Matrix":
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)

elif plot_option == "Feature Importance":
    st.subheader("Feature Importance")
    # Feature importance untuk Naive Bayes
    feature_importance = []
    for feature in features:
        loyal_mean = df[df['Loyal'] == 1][feature].mean()
        not_loyal_mean = df[df['Loyal'] == 0][feature].mean()
        importance = abs(loyal_mean - not_loyal_mean) / (loyal_mean + not_loyal_mean + 1e-8)
        feature_importance.append(importance)
    
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    importance_df.plot(x='Feature', y='Importance', kind='bar', ax=ax, color='skyblue')
    ax.set_title("Pentingnya Fitur")
    ax.set_xlabel("Fitur")
    ax.set_ylabel("Importance")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

st.markdown("---")
st.subheader("🎯 Prediksi Loyalitas Pelanggan")

st.info("""
**📋 Informasi Input:**
- **Total Pengeluaran**: Total belanja pelanggan untuk semua produk (prediktor kuat loyalitas)
- **Pendapatan Tahunan**: Kemampuan finansial pelanggan (prediktor loyalitas)
- **Pernah Komplain**: Indikator kepuasan pelanggan (prediktor negatif loyalitas)

**🔧 Perbaikan Model:**
Model ini sekarang menggunakan fitur yang benar tanpa data leakage, sehingga probabilitas yang dihasilkan lebih realistis dan dapat dipercaya.
""")

# Input form untuk prediksi
with st.form("prediction_form"):
    st.write("Masukkan data pelanggan untuk memprediksi kemungkinan menjadi pelanggan loyal:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        total_spent = st.number_input("Total Pengeluaran ($)", min_value=0, value=500000, step=50000, help="Total pengeluaran pelanggan untuk semua produk")
        income = st.number_input("Pendapatan Tahunan ($)", min_value=0, value=50000000, step=1000000, help="Pendapatan tahunan pelanggan")
    
    with col2:
        complain = st.selectbox("Pernah Komplain?", options=[0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak", help="Apakah pelanggan pernah mengajukan komplain")
    
    submit = st.form_submit_button("🔮 Prediksi Loyalitas", use_container_width=True)

if submit:
    try:
        # Buat dataframe dari input yang sudah dimasukkan
        input_df = pd.DataFrame([{
            "TotalSpent": total_spent,
            "Complain": complain,
            "Income": income
        }])
        
        # Prediksi
        prob = model.predict_proba(input_df)[0]
        pred = model.predict(input_df)[0]
        
        st.subheader("🎯 Hasil Prediksi")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if pred == 1:
                st.success("✅ **PELANGGAN LOYAL**")
            else:
                st.error("❌ **TIDAK LOYAL**")
        
        with col2:
            st.info(f"🎲 Probabilitas Loyal: **{prob[1]:.1%}**")
        
        with col3:
            st.warning(f"🎲 Probabilitas Tidak Loyal: **{prob[0]:.1%}**")
        
        # Visualisasi probabilitas
        prob_df = pd.DataFrame({
            'Kategori': ['Tidak Loyal', 'Loyal'],
            'Probabilitas': [prob[0], prob[1]]
        })
        
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['green' if x == 'Loyal' else 'red' for x in prob_df['Kategori']]
        prob_df.plot(x='Kategori', y='Probabilitas', kind='bar', ax=ax, color=colors)
        ax.set_title("Distribusi Probabilitas Prediksi")
        ax.set_ylabel("Probabilitas")
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Rekomendasi
        st.subheader("💡 Rekomendasi")
        if pred == 1:
            st.success("""
            **Strategi untuk Pelanggan Loyal:**
            - 🎁 Berikan program reward dan loyalty points
            - 📧 Kirim penawaran eksklusif dan early access
            - 🤝 Pertahankan kualitas layanan yang excellent
            - 📊 Monitor kepuasan pelanggan secara berkala
            """)
        else:
            st.warning("""
            **Strategi untuk Meningkatkan Loyalitas:**
            - 🎯 Personalisasi kampanye pemasaran
            - 💰 Tawarkan diskon dan promosi menarik
            - 📱 Tingkatkan pengalaman digital dan website
            - 🔄 Follow up dengan program retensi pelanggan
            """)
    
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.info("Pastikan input data sesuai dengan format yang diharapkan oleh model. Coba lagi dengan nilai yang berbeda.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Dashboard Prediksi Loyalitas Pelanggan | Dibuat dengan Streamlit</p>
</div>
""", unsafe_allow_html=True) 