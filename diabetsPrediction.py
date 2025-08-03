                    ### Diabets Prediction with Logistic Regression ###

# İş Problemi:
# Özellikleri belirtildiğinde kişilerin diyabet hastası olup
# olmadıklarını tahmin edebilecek bir makine öğrenmesi
# modeli geliştirebilir misiniz?

# Veri seti ABD'deki Ulusal Diyabet-Sindirim-Böbrek Hastalıkları Enstitüleri'nde tutulan büyük veri setinin
# parçasıdır. ABD'deki Arizona Eyaleti'nin en büyük 5. şehri olan Phoenix şehrinde yaşayan 21 yaş ve üzerinde olan
# Pima Indian kadınları üzerinde yapılan diyabet araştırması için kullanılan verilerdir. 768 gözlem ve 8 sayısal
# bağımsız değişkenden oluşmaktadır. Hedef değişken "outcome" olarak belirtilmiş olup; 1 diyabet test sonucunun
# pozitif oluşunu, 0 ise negatif oluşunu belirtmektedir.

# Değişkenler
# Pregnancies: Hamilelik sayısı
# Glucose: Glikoz.
# BloodPressure: Kan basıncı.
# SkinThickness: Cilt Kalınlığı
# Insulin: İnsülin.
# BMI: Beden kitle indeksi.
# DiabetesPedigreeFunction: Soyumuzdaki kişilere göre diyabet olma ihtimalimizi hesaplayan bir fonksiyon.
# Age: Yaş (yıl)
# Outcome: Kişinin diyabet olup olmadığı bilgisi. Hastalığa sahip (1) ya da değil (0)

# 1. Exploratory Data Analysis
# 2. Data Preprocessing
# 3. Model & Prediction
# 4. Model Evaluation
# 5. Model Validation: Holdout
# 6. Model Validation: 10-Fold Cross Validation
# 7. Prediction for A New Observation


    # 1.Gerekli Kütüphaneler

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report,RocCurveDisplay


    # 2.Yardımcı Fonksiyonlar

#Bu fonksiyon, bir sütundaki verilerin belirli bir yüzdelik dilimine göre aykırı kabul edilebilecek
#değerlerin aralığını tanımlayan iki değer (low_limit ve up_limit) döndürür.

def outlier_thresholds(df, col, q1 = 0.05, q3 = 0.95):
    q1_val = df[col].quantile(q1)
    q3_val = df[col].quantile(q3)
    iqr = q3_val - q1_val
    low_limit = q1_val - 1.5 * iqr
    up_limit = q3_val + 1.5 * iqr
    return low_limit, up_limit

#Bu fonksiyon, aykırı değerleri tespit edip, onları belirlediğimiz alt ve üst sınırlara sabitleyerek veri setini temizler.

def replace_with_thresholds(df, col):
    low, up = outlier_thresholds(df, col)
    df[col] = np.where(df[col] < low, low, np.where(df[col] > up, up, df[col]))

#Bu kod, bir modelin tahminlerinin ne kadar doğru olduğunu gösteren tabloyu (karmaşıklık matrisi) görselleştirir ve doğruluk oranını başlıkta belirtir.

def plot_confusion_matrix(y_true, y_pred):
    acc = round(accuracy_score(y_true, y_pred), 2)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot = True, fmt= "d")
    plt.xlabel("Prediction")
    plt.ylabel("Actual")
    plt.title(f"Accuary: {acc}")
    plt.show()

    # 3.Veri Yükleme ve İlk İnceleme

#Local'den yüklemek daha kolay oldugu ıcın ben bu yolu tercih ettim fakat api kullanarak yapmak istiyorsanız:
#https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset/data adresinden veri setine ulaşabilirsiniz.

df = pd.read_csv(r"Dosya Yolu")

#Bu kod, veri setinizdeki "Outcome" sütununu tahmin edilecek sonuç olarak belirlerken,
#diğer tüm sütunları bu tahmini yapmak için kullanılacak bilgiler (özellikler) olarak ayırır.

target = "Outcome";
features = [col for col in df.columns if col != target]

    # 4.Eksik ve Aykırı Değer İşleme

#Bu işlem, modeli eğitmeden önce veri setindeki aykırı değerleri temizlemek ve veri setini daha sağlam hale getirmek için kullanılır.

for col in features:
    replace_with_thresholds(df, col)


    # 5.Ölçekleme

#Bu kod, modelin öğrenme sürecini olumsuz etkileyebilecek aykırı değerlerin etkisini
#azaltmak için veri setindeki özelliklerin değer aralıklarını RobustScaler ile standartlaştırır.

scaler = RobustScaler()
df[features] = scaler.fit_transform(df[features])


    # 6.Modelleme

#Bu kod lojistik regresyon modelini oluşturur,
#eğitir ve ardından hem doğrudan sınıf tahminleri (y_pred) hem de bu tahminlere ait olasılık değerlerini (y_prob) hesaplar.

x = df[features]
y = df[target]

log_model = LogisticRegression().fit(x, y)
y_pred = log_model.predict(x)
y_prob = log_model.predict_proba(x)[:, 1]


    # 7.Başarı Değerlendirmesi

#Bu üç satır kod, bir modelin eğitim aşamasındaki başarısını kapsamlı bir şekilde incelememizi
#ve performansını çeşitli açılardan değerlendirmemizi sağlar.

print("Training Classificattion Report: \n", classification_report(y, y_pred))
plot_confusion_matrix(y, y_pred)
print("ROC AUC Score(Train): ", roc_auc_score(y, y_prob))

    # 8.Holdout Doğrulama

#Bu kod, modeli eğitim verileriyle eğittikten sonra, görmediği test verileriyle gerçek performansını
# ölçmek ve raporlamak için standart bir süreç izler.

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 17)

log_model = LogisticRegression().fit(x_train, y_train)
y_pred = log_model.predict(x_test)
y_prob = log_model.predict_proba(x_test)[:, 1]

print("Holdout Classification Report: \n", classification_report(y_test,y_pred))
RocCurveDisplay.from_estimator(log_model, x_test, y_test)
plt.title("ROC Curve (Test Set)")
plt.plot([0, 1], [0, 1], "r--")
plt.show()
print("ROC AUC Score (Test):", roc_auc_score(y_test, y_prob))

    # 9. 10-Fold Cross Validation

#Bu yöntem, modelimizin performansının tek bir rastgele test seti yerine, veri setinin farklı bölümlerinde nasıl bir performans
#gösterdiğini görmemizi sağlar. Bu sayede modelin genelleme yeteneği hakkında daha güvenilir bir değerlendirme elde ederiz.

cv_result = cross_validate(log_model, x, y, cv = 10, scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"])

print("CV Accuracy:", cv_result["test_accuracy"].mean())
print("CV Precision:", cv_result["test_precision"].mean())
print("CV Recall:", cv_result["test_recall"].mean())
print("CV F1 Score:", cv_result["test_f1"].mean())
print("CV ROC AUC:", cv_result["test_roc_auc"].mean())

    # 10. Yeni Gözlem Üzerinden Tahmin

#Bu kod, modelin gerçek dünyadaki yeni verilerle karşılaştığında nasıl bir tahmin yapacağını simüle etmemize yarar.

new_sample = x.sample(1, random_state = 45)
prediction = log_model.predict(new_sample)
print("New Sample Prediction:", prediction)
