import streamlit as st
import pandas as pd
from pycaret.classification import *
from pycaret.classification import setup, compare_models, pull, save_model, load_model, predict_model
import matplotlib.pyplot as plt
import seaborn as sns


# Başlık
st.title("AutoML Sınıflandırma Arayüzü - PyCaret ile")

# Veri kümesi yükleme
uploaded_file = st.file_uploader("Veri Kümesini Yükleyin (CSV formatında)", type="csv")
uploaded_metadata = st.file_uploader("Metadata Yükleyin (CSV formatında)", type="csv")
if uploaded_file is not None and uploaded_metadata is not None:
    data = pd.read_csv(uploaded_file)
    metadata = pd.read_csv(uploaded_metadata)
    
    # Veri ve meta bilgiyi gösterme
    tab1, tab2, tab3, tab4 = st.tabs(["Veri Kümesi", "Tanımlayıcı İstatistikler", "Veri Seti Özeti", "Değişken Tanımlamaları"])
    with tab1:
        st.write("Veri Kümesi", data)
    with tab2:
        st.write(data.describe().T)
    with tab3:
        info = pd.DataFrame({
            'Veri Tipi': data.dtypes,
            'Eksik Değerler': data.isnull().sum(),
            'Toplam Değer': data.count(),
            'Benzersiz Değerler': data.nunique()
        })
        st.table(info)
    with tab4:
        st.write(metadata.T)

    # PyCaret ayarları
    st.sidebar.header("PyCaret Model Ayarları")
    target = st.sidebar.selectbox("Hedef Değişkeni Seçin", data.columns)

    # PyCaret kurulumu
    if st.sidebar.button("Modeli Kur ve Eğit"):
        # Model kurulumunu yapıyoruz
        setup(data=data, target=target, silent=True, verbose=False, session_id=42)
        
        # Modelleri karşılaştırıyoruz
        best_model = compare_models()
        
        # Model performans metrikleri
        st.subheader("Model Karşılaştırması Sonuçları")
        comparison_df = pull()
        st.write(comparison_df)

        # En iyi modeli göster
        st.subheader("En İyi Model:")
        st.write(best_model)

        # Performans metriklerini görselleştirme
        tab1, tab2 = st.tabs(["Confusion Matrix", "ROC Curve"])
        with tab1:
            plot_model(best_model, plot='confusion_matrix')
            st.pyplot()
        with tab2:
            plot_model(best_model, plot='auc')
            st.pyplot()

        # Modeli kaydetme
        save_model(best_model, 'best_classification_model')
        st.success("Model başarıyla kaydedildi.")
        
        # Yeni veri tahmini
        st.sidebar.subheader("Yeni Veri Üzerinde Tahmin")
        uploaded_new_data = st.sidebar.file_uploader("Yeni Veri Yükleyin (CSV)", type="csv")
        if uploaded_new_data:
            new_data = pd.read_csv(uploaded_new_data)
            predictions = predict_model(best_model, data=new_data)
            st.write("Tahmin Sonuçları:", predictions)
