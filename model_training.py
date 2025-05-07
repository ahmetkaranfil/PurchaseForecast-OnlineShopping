import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    # 1. Veri Yükleme
    df = pd.read_csv(r"C:\Users\pv\Downloads\online_shoppers_purchasing_intention_dataset\online_shoppers_intention.csv")

    # 2. Veri Ön İşleme
    df.dropna(inplace=True)
    df['Weekend'] = df['Weekend'].astype(int)
    df['Revenue'] = df['Revenue'].astype(int)

    month_map = {'Jan': 0, 'Feb': 1, 'Mar': 2, 'Apr': 3, 'May': 4, 'June': 5,
                 'Jul': 6, 'Aug': 7, 'Sep': 8, 'Oct': 9, 'Nov': 10, 'Dec': 11}
    df['Month'] = df['Month'].map(month_map)

    le = LabelEncoder()
    df['VisitorType'] = le.fit_transform(df['VisitorType'])

    X = df.drop('Revenue', axis=1)
    y = df['Revenue']

    # 3. Normalizasyon
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # 4. Özellik Seçimi
    kbest = SelectKBest(score_func=chi2, k=9)
    X_selected = kbest.fit_transform(X_scaled, y)
    selected_features = X_scaled.columns[kbest.get_support()]
    print("✅ Seçilen özellikler:", selected_features.tolist())

    # 5. Eğitim Başlangıcı
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    # 6. SMOTE ile dengeleme
    X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)

    # 7. Modeller
    models = {
        'Logistic Regression': LogisticRegression(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'KNN': KNeighborsClassifier()
    }

    best_model = None
    best_accuracy = 0
    best_model_name = ""

    # 8. Eğitim ve Değerlendirme
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        print(f"\n📊 {name}")
        print(f"Doğruluk: {acc:.4f}")

        # Confusion matrix görseli
        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"{name} - Confusion Matrix")
        plt.xlabel("Tahmin")
        plt.ylabel("Gerçek")
        plt.tight_layout()
        plt.savefig(f"{name.replace(' ', '_')}_confusion_matrix.png")
        plt.close()

        # En iyi modeli sakla
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_model_name = name

    # 9. Kaydetme
    print(f"\n✅ En iyi model: {best_model_name} ({best_accuracy:.4f} doğruluk ile kaydedildi)")
    joblib.dump({
        'model': best_model,
        'scaler': scaler,
        'selector': kbest,
        'selected_features': selected_features.tolist(),
        'label_encoder': le
    }, "model.pkl")


if __name__ == "__main__":
    main()
