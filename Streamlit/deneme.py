import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report


def siniflandirma_sayfasi():

    # Başlık
    st.title("CATML Sınıflandırma Arayüzü")

    # Veri kümesi yükleme
    uploaded_file = st.file_uploader("Veri Kümesini Yükleyin (CSV formatında)", type="csv")
    uploaded_metadata = st.file_uploader("Metadata Yükleyin (... formatında)", type="csv")
    if uploaded_file is not None and uploaded_metadata is not None:
        data = pd.read_csv(uploaded_file)
        metadata = pd.read_csv(uploaded_metadata)
        tab1, tab2, tab3, tab4 =st.tabs(["Veri Kümesi", "Tanımlayıcı İstatistikler", "Veri Seti Özeti", "Değişken Tanımlamaları"])
        with tab1:
            st.write("Veri Kümesi", data)

        with tab2:
            st.write(data.describe().T)
        with tab3:
            info = pd.DataFrame({
                #'Değişken': data.columns,
                'Veri Tipi': data.dtypes,
                'Eksik Değerler': data.isnull().sum(),
                'Toplam Değer': data.count(),
                'Benzersiz Değerler': data.nunique()})
            
            st.table(info)
        with tab4:
            st.write(metadata.T)

        # Ön işleme seçenekleri
        st.sidebar.header("Veri Ön İşleme Adımları")
        missing_option = st.sidebar.selectbox("Eksik Değer İşleme", ("Kaldır", "Ortalama ile Doldur", "Medyan ile Doldur"))
        scaling_option = st.sidebar.checkbox("Özellik Ölçekleme (StandardScaler)")
        encoding_option = st.sidebar.checkbox("Kategorik Değişkenleri Kodla (Label Encoding)")

        # Eksik değer işleme
        if missing_option == "Kaldır":
            data = data.dropna()
        elif missing_option == "Ortalama ile Doldur":
            data = data.fillna(data.mean())
        elif missing_option == "Medyan ile Doldur":
            data = data.fillna(data.median())

        # Kategorik değişkenleri kodlama
        if encoding_option:
            label_encoders = {}
            for column in data.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                data[column] = le.fit_transform(data[column])
                label_encoders[column] = le

        # Özellik ölçekleme
        if scaling_option:
            scaler = StandardScaler()
            numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
            data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

        # Özellik ve hedef değişkenleri seçme
        target = st.selectbox("Hedef Değişkeni Seçin", data.columns)
        X = data.drop(columns=[target])
        y = data[target]

        # Veri kümesini ayırma
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model seçimi
        model_name = st.sidebar.selectbox("Model", ("Lojistik Regresyon", "Rasgele Orman", "SVM", "K-En Yakın Komşu"))
        

        # Model tanımlama
        def get_model(model_name):
            if model_name == "Lojistik Regresyon":
                return LogisticRegression(max_iter=200)
            elif model_name == "Rasgele Orman":
                return RandomForestClassifier()
            elif model_name == "SVM":
                return SVC()
            elif model_name == "K-En Yakın Komşu":
                return KNeighborsClassifier()

        model = get_model(model_name)

        # Model eğitimi
        if st.button("Modeli Eğit"):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Metrikleri hesapla
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            # Sonuçları göster
            st.write(f"Seçilen Model: {model_name}")
            st.write("Doğruluk (Accuracy):", accuracy)
            st.write("Kesinlik (Precision):", precision)
            st.write("Duyarlılık (Recall):", recall)
            st.write("F1 Skoru:", f1)
    # Görselleştirme sekmeleri
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["Confusion Matrix", "ROC-AUC Eğrisi", "Özellik Önem Düzeyleri", "Classification Report", "Hiper Parametreler"])

            with tab1:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
                st.pyplot(fig)

            with tab2:
                st.subheader("ROC-AUC Eğrisi")
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_prob)
                    roc_auc = auc(fpr, tpr)
                    fig, ax = plt.subplots()
                    ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
                    ax.plot([0, 1], [0, 1], 'k--')
                    ax.set_xlabel("False Positive Rate")
                    ax.set_ylabel("True Positive Rate")
                    ax.legend()
                    st.pyplot(fig)

            with tab3:
                st.subheader("Özellik Önem Düzeyleri")
                if model_name == "Rasgele Orman":
                    feature_importances = model.feature_importances_
                    importance_df = pd.DataFrame({"Özellik": X.columns, "Önem Düzeyi": feature_importances})
                    importance_df = importance_df.sort_values(by="Önem Düzeyi", ascending=False)
                    st.write(importance_df)
                    fig, ax = plt.subplots()
                    sns.barplot(data=importance_df, x="Önem Düzeyi", y="Özellik", ax=ax)
                    st.pyplot(fig)
                else:
                    st.write("Model Seçiminizde Bu Özellik Kullanılamıyor")
            
            with tab4:
                # Classification Report
                st.write("Sınıflandırma Raporu:")
                report = classification_report(y_test, y_pred)
                st.text(report)
            
            with tab5:
                # Model hiperparametreleri
                params = model.get_params()
                st.title("Lojistik Regresyon Hiperparametreleri")

                param_df = pd.DataFrame(list(params.items()), columns=["Parametre", "Değer"])
                st.table(param_df)

