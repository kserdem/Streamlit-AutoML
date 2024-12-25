import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns


def regresyon_sayfasi():
    # Başlık
    st.title("AutoML Regresyon Arayüzü")

    # Veri kümesi yükleme
    uploaded_file = st.file_uploader("Veri Kümesini Yükleyin (CSV formatında)", type="csv")
    uploaded_metadata = st.file_uploader("Metadata Yükleyin (... formatında)", type="csv")
    if uploaded_file is not None and uploaded_metadata is not None:
        data = pd.read_csv(uploaded_file)
        metadata = pd.read_csv(uploaded_metadata)
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
                return LinearRegression()
            elif model_name == "Rasgele Orman":
                return RandomForestRegressor()
            elif model_name == "SVM":
                return SVR()
            elif model_name == "K-En Yakın Komşu":
                return KNeighborsRegressor()

        model = get_model(model_name)

        # Model eğitimi
        if st.button("Modeli Eğit"):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Metrikleri hesapla
            mse = mean_squared_error(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            # Sonuçları göster
            st.write(f"Seçilen Model: {model_name}")
            st.write("Mean Squared Error (MSE):", mse)
            st.write("Root Mean Squared Error (RMSE):", rmse)
            st.write("R-Kare (R²):", r2)
            st.write("Mean Absolute Error (MAE):", mae)

            # Görselleştirme sekmeleri
            tab1, tab2, tab3, tab4 = st.tabs(["Kayıp Fonksiyonu", "Gerçek vs Tahmin", "Özellik Önem Düzeyleri", "Hiper Parametreler"])

            with tab1:
                st.subheader("Kayıp Fonksiyonu")
                fig, ax = plt.subplots()
                ax.plot(y_test.values, label='Gerçek Değerler', color='blue')
                ax.plot(y_pred, label='Tahmin Edilen Değerler', color='red')
                ax.set_xlabel("Veri Noktaları")
                ax.set_ylabel("Hedef Değişken")
                ax.legend()
                st.pyplot(fig)

            with tab2:
                st.subheader("Gerçek vs Tahmin")
                fig, ax = plt.subplots()
                ax.scatter(y_test, y_pred)
                ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
                ax.set_xlabel('Gerçek Değerler')
                ax.set_ylabel('Tahmin Edilen Değerler')
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
                # Model hiperparametreleri
                params = model.get_params()
                st.title("Model Hiperparametreleri")
                param_df = pd.DataFrame(list(params.items()), columns=["Parametre", "Değer"])
                st.table(param_df)
