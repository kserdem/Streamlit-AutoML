import streamlit as st
import pandas as pd
tabel=pd.DataFrame({"Colm1": [1,2,3,4,5,6], "colm2": [2,25,44,78,9,78]})

#Başlık oluşturma

st.title('Merhabalar')
st.subheader(" Alt Başlık 1")

st.header('Header')

st.text("Merhabalar bu br düz yazı")
#https://www.markdownguide.org/cheat-sheet/
st.markdown("**denemeeeeeeee**")

st.caption("Caption")

#https://katex.org/docs/supported.html
st.latex(r"\begin{pmatrix}a&b\\c&d\end{pmatrix}")

json ={"a":"1,2,3"}
st.json(json)

code='''st.title('Merhabalar')
st.subheader(" Alt Başlık 1")'''

st.code(code,language="python")

st.write(" ## H2")

st.metric(label="Wind Speed", value="120ms⁻¹ ",delta="1.4ms⁻¹ ")

st.table(tabel)

st.dataframe(tabel)

st.image("indir.jpg",caption="Tatlişko", width=680)

#st.audio() ile mp3 yükleyebiliriz
#st.video() ile video yükleyebiliriz














