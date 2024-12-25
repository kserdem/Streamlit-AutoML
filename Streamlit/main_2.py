import streamlit as st

# Kullanıcı girişi kontrolü için bir oturum durumu tanımlıyoruz
if "giriş_yapıldı" not in st.session_state:
    st.session_state["giriş_yapıldı"] = False

# Giriş sayfası fonksiyonu
def giriş_sayfası():
    st.title("CATML")
    st.header("Kullanıcı Giriş Sayfası")
    st.image("pamuk.jpeg", width=200)  # Logo ekleyin
    
    # Kullanıcı adı ve şifre girişi
    kullanıcı_adı = st.text_input("Kullanıcı Adı")
    şifre = st.text_input("Şifre", type="password")
    
    # Basit giriş doğrulaması (Örneğin, kullanıcı adı: admin, şifre: 1234)
    if st.button("Giriş"):
        if kullanıcı_adı == "admin" and şifre == "1234":
            st.session_state["giriş_yapıldı"] = True
            st.success("Başarıyla giriş yapıldı!")
        else:
            st.error("Kullanıcı adı veya şifre hatalı!")

# Ana uygulama fonksiyonu
def ana_uygulama():
    st.title("Sınıflandırma Sayfası")
    st.write("Burada sınıflandırma işlemlerinizi yapabilirsiniz.")
    # Buraya sınıflandırma ile ilgili içerikleri ekleyebilirsiniz.

# Giriş yapıldıysa ana uygulamaya yönlendir
if st.session_state["giriş_yapıldı"]:
    ana_uygulama()
else:
    giriş_sayfası()
