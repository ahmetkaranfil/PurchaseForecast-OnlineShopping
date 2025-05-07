import streamlit as st
import joblib
import pandas as pd

st.title("🛒 Online Alışverişte Satın Alma Tahmini")

# Model Bilgilendirme Bölümü
with ((st.expander("📈 **Model Bilgileri ve Performans Karşılaştırması**"))):
    st.markdown("""
    ### 🧪 Model Performansları
    - 🔹 **Logistic Regression**: %85.36 doğruluk oranı
    - 🔹 **Decision Tree**: %84.24 doğruluk oranı
    - 🔹 **KNN**: %86.72 doğruluk oranı
    - 🔹 **Random Forest**: %89.04 doğruluk oranı
    """)

    st.success("✅ En iyi model: Random Forest (%89.04)")

# Model dosyasını yükleyelim
data = joblib.load("model.pkl")
model = data['model']
scaler = data['scaler']
selector = data['selector']
selected_features = data['selected_features']
le = data['label_encoder']  # LabelEncoder eklendi

# Modelin eğitiminde kullanılan orijinal tüm özellik isimlerini sırayla buraya yazalım
all_feature_names = [
    'Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
    'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates',
    'PageValues', 'SpecialDay', 'Month', 'OperatingSystems',
    'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend'
]

# Kullanıcıdan alınacak 8 temel özellik
user_inputs = {
    'Administrative': st.number_input("📄 İdari Sayfa Sayısı (İletişim, Kullanım Koşulları, Hakkımızda gibi sayfaların sayısı)", min_value=0, key="administrative"),
    'Informational': st.number_input("📄 Bilgilendirici Sayfa Sayısı (Sıkça Sorulan Sorular, Ürün Karşılaştırmaları, Yardım gibi sayfaların sayısı)", min_value=0, key="informational"),
    'ProductRelated': st.number_input("🛍️ Ürünle İlgili Sayfa Sayısı (Ürün Özellikleri veya Ürün Yorumları gibi sayfaların sayısı)", min_value=0, key="product_related"),
    'BounceRates': st.number_input("📉 Hemen Çıkma Oranı (0 ile 1 arasında)", min_value=0.0, max_value=1.0),
    'ExitRates': st.number_input("🚪 Belirli Bir Süre Sonra Çıkış Oranı (0 ile 1 arasında)", min_value=0.0, max_value=1.0),
    'PageValues': st.number_input("💲 Sayfa Değeri (TL cinsinden)", min_value=0),
    'SpecialDay': st.selectbox("🎉 Özel Gün Ziyareti mi?", options=[1, 0], format_func=lambda x: "Evet" if x == 1 else "Hayır"),
    'Month': st.selectbox("🗓️ Ay", options=list(range(12)), format_func=lambda x: ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"][x]),
    'VisitorType': le.transform([st.selectbox("👥 Ziyaretçi Türü", options=le.classes_)])[0]
}

# Diğer tüm eksik özellikleri default ile tamamlayalım
default_values = {
    'Administrative_Duration': 80.82,      # float
    'Informational_Duration': 34.47,       # float
    'ProductRelated_Duration': 1194.75,    # float
    'OperatingSystems': 2,                 # int
    'Browser': 2,                          # int
    'Region': 3,                           # int
    'TrafficType': 4,                      # int
    'Weekend': 0.13                        # float
}

# input_df'i doğru sırada oluşturalım
combined_input = {**default_values, **user_inputs}
ordered_input = {feature: combined_input[feature] for feature in all_feature_names}
input_df = pd.DataFrame([ordered_input])

# Tahmin butonu
if st.button("📊 Tahmin Et"):
    input_scaled = scaler.transform(input_df)
    input_selected = selector.transform(input_scaled)
    prediction = model.predict(input_selected)
    probabilities = model.predict_proba(input_selected)[0]
    prob_satinalma = probabilities[1] * 100
    prob_satinalmama = probabilities[0] * 100

    if prediction[0] == 1:
        st.success("📈 Tahmin: 🟢 Satın alacak (Revenue = 1)")
    else:
        st.error("📉 Tahmin: 🔴 Satın almayacak (Revenue = 0)")

    st.info(f"🧮 Satın alma olasılığı: **%{prob_satinalma:.2f}**")
    st.info(f"🚫 Satın almama olasılığı: **%{prob_satinalmama:.2f}**")
