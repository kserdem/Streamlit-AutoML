import streamlit as st
from veritabani import veritabani_olustur, kullanıcı_ekle, kullanıcı_doğrula
import siniflandirma 
import regresyon 




# Veritabanını başlat
veritabani_olustur()

# Kullanıcı giriş durumu
if "giriş_yapıldı" not in st.session_state:
    st.session_state["giriş_yapıldı"] = False

# Giriş sayfası
def giriş_sayfası():
    st.title("CATML")
    st.header("Giriş Yap / Kayıt Ol")
    st.image("pamuk.jpeg", width=200)

    seçim = st.radio("Bir işlem seçin:", ("Giriş Yap", "Kayıt Ol"))

    if seçim == "Giriş Yap":
        kullanıcı_adı = st.text_input("Kullanıcı Adı")
        şifre = st.text_input("Şifre", type="password")
        if st.button("Giriş"):
            if kullanıcı_doğrula(kullanıcı_adı, şifre):
                st.session_state["giriş_yapıldı"] = True
                st.success("Başarıyla giriş yapıldı!")
            else:
                st.error("Kullanıcı adı veya şifre hatalı!")

    elif seçim == "Kayıt Ol":
        yeni_kullanıcı_adı = st.text_input("Yeni Kullanıcı Adı")
        yeni_şifre = st.text_input("Şifre", type="password")
        if st.button("Kayıt Ol"):
            try:
                kullanıcı_ekle(yeni_kullanıcı_adı, yeni_şifre)
                st.success("Kayıt başarılı! Şimdi giriş yapabilirsiniz.")
            except sqlite3.IntegrityError:
                st.error("Bu kullanıcı adı zaten mevcut.")

# Ana uygulama
def ana_uygulama():
    st.title("Kullanmak İstediğiniz ML Modelini Seçiniz")
    secenek = st.selectbox("Model türünü seçin:", ["Sınıflandırma", "Regresyon"])
    if secenek == "Sınıflandırma":
        siniflandirma.siniflandirma_sayfasi()

    elif secenek == "Regresyon":
        regresyon.regresyon_sayfasi()


# Giriş yapıldıysa ana sayfaya yönlendir, değilse giriş sayfasını göster
if st.session_state["giriş_yapıldı"]:
    ana_uygulama()
    #siniflandirma_sayfasi()
else:
    giriş_sayfası()
