<img src="https://twemoji.maxcdn.com/v/latest/svg/1f1ec-1f1e7.svg" width="20"/> English
<br>
<h3>🩺 Diabetes Prediction Project<h3>
This project's goal is to develop a machine learning model that predicts whether a person has diabetes based on specific health-related features. The Logistic Regression algorithm, a common choice for classification problems, was used for this purpose.

🚀 Project Goal
The primary objective of this project is to use a dataset collected from Pima Indian women to build a reliable model that can predict the risk of diabetes for a new individual.

📊 Dataset
Data Source: Pima Indian Diabetes Dataset (Kaggle)

Observations: 768

Variables: 8 independent variables (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age) and 1 target variable (Outcome).

🧠 Methodology
Data Preprocessing: Outlier thresholds were defined to clean the data, and RobustScaler was used for feature scaling to handle potential outliers.

Model Development:

The dataset was split into 80% training and 20% testing sets using the Holdout Method.

A Logistic Regression model was trained on the training data.

Model Validation and Evaluation:

The model's performance was evaluated on the test set using the Holdout method. Its accuracy, precision, and recall were analyzed with the Classification Report, Confusion Matrix, and ROC Curve.

10-Fold Cross-Validation was performed to obtain a more robust measure of the model's generalization ability.

Prediction: The trained model was used to predict the diabetes status of a new, unseen observation.

✨ Results
Holdout Validation Results:

Accuracy: 0.77

ROC AUC Score: 0.83

Cross-Validation (10-Fold CV) Results:

Mean Accuracy: 0.77

Mean Precision: 0.72

Mean Recall: 0.58

Mean F1 Score: 0.64

Mean ROC AUC Score: 0.83

The results indicate that the model performs well in predicting diabetes risk. The high ROC AUC score suggests a strong ability to distinguish between diabetic and non-diabetic individuals.
<br><br><br>

<img src="https://twemoji.maxcdn.com/v/latest/svg/1f1f9-1f1f7.svg" width="20"/> Türkçe
<br>
<h3>🩺 Diyabet Tahmini Projesi<h3>
Bu proje, bir kişinin belirli sağlık özelliklerine dayanarak diyabet hastası olup olmadığını tahmin eden bir makine öğrenimi modelinin geliştirilmesini amaçlar. Projede, sınıflandırma problemleri için yaygın olarak kullanılan Lojistik Regresyon algoritması kullanılmıştır.

🚀 Proje Amacı
Projenin temel amacı, Pima Indian kadınları üzerinde toplanmış olan bir veri setini kullanarak, yeni bir birey için diyabet riskini tahmin edebilecek güvenilir bir model oluşturmaktır.

📊 Kullanılan Veri Seti
Veri Kaynağı: Pima Indian Diyabet Veri Seti (Kaggle)

Gözlem Sayısı: 768

Değişkenler: 8 bağımsız değişken (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age) ve 1 hedef değişken (Outcome).

🧠 Metodoloji
Veri Ön İşleme: Olası aykırı değerlerin etkisini azaltmak için eşik değerleri belirlenerek veriler temizlendi ve RobustScaler kullanılarak ölçekleme yapıldı.

Model Geliştirme:

Veri seti, Holdout Yöntemi ile %80 eğitim ve %20 test setlerine ayrıldı.

Eğitim seti üzerinde bir Lojistik Regresyon modeli eğitildi.

Model Doğrulaması ve Değerlendirmesi:

Modelin performansı, Holdout yöntemi ile test setinde değerlendirildi. Sınıflandırma Raporu, Karmaşıklık Matrisi ve ROC Eğrisi kullanılarak modelin doğruluğu, hassasiyeti ve duyarlılığı incelendi.

Modelin genelleme yeteneğini daha güvenilir bir şekilde ölçmek için 10-Katlı Çapraz Doğrulama (10-Fold Cross-Validation) yöntemi uygulandı.

Tahmin: Eğitilmiş model, yeni bir bireyin diyabet durumunu tahmin etmek için kullanıldı.

✨ Sonuçlar
Holdout Doğrulama Sonuçları:

Doğruluk (Accuracy): 0.77

ROC AUC Skoru: 0.83

Çapraz Doğrulama (10-Fold CV) Sonuçları:

Ortalama Doğruluk: 0.77

Ortalama Hassasiyet (Precision): 0.72

Ortalama Duyarlılık (Recall): 0.58

Ortalama F1 Skoru: 0.64

Ortalama ROC AUC Skoru: 0.83

Elde edilen sonuçlar, modelin diyabet riskini tahmin etme konusunda iyi bir performans sergilediğini göstermektedir. Özellikle ROC AUC skoru, modelin pozitif ve negatif sınıfları ayırma yeteneğinin güçlü olduğunu belirtir.
