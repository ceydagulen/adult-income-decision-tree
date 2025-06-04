
#  Adult Income Sınıflandırması – Karar Ağacı Modeli

Bu proje, **UCI Adult (Census Income)** veri seti kullanılarak bireylerin yıllık gelirinin **50.000$'ın üzerinde olup olmadığını** tahmin etmeyi amaçlar. Python ile geliştirilen bu sistemde **karar ağacı modeli**, hiperparametre optimizasyonu ve kapsamlı görselleştirme teknikleri kullanılmıştır.

---

## Proje Yapısı

adult-income-decision-tree/

├── adult.data            # Eğitim verisi

├── adult.test            # Test verisi

├── main.py               # Ana Python kodu

├── requirements.txt      # Gereken kütüphaneler

├── README.md             # Proje açıklamaları

└── gorseller/            # Otomatik oluşturulan grafik çıktılar



---

##  Kurulum Talimatları

### 1. Python Ortamı

Bu proje Python 3.8+ sürümü ile uyumludur. Sisteminizde yüklü değilse, [python.org](https://www.python.org/downloads/) üzerinden kurabilirsiniz.

### 2. Gerekli Kütüphaneleri Yükleme

Aşağıdaki komutu terminalde çalıştırarak gerekli bağımlılıkları yükleyin:

```bash
pip install -r requirements.txt
``````


Eğer requirements.txt dosyası yoksa, bu kütüphaneler yüklenmelidir:

```bash
pip install pandas numpy matplotlib scikit-learn
``````
## Nasıl Çalıştırılır?

adult.data ve adult.test dosyalarının proje klasöründe olduğundan emin olun.

Aşağıdaki komutla projeyi çalıştırın:
```bash
python main.py
``````

Kod çalıştıktan sonra tüm çıktı görselleri gorseller/ klasörüne otomatik kaydedilecektir.


## Üretilen Grafikler

Gelir sınıfı histogramı

Sürekli değişken korelasyon matrisi (ısı haritası)

Boxplot analizleri

Karışıklık matrisleri (eğitim & test)

Feature importance grafiği

Karar ağacı görselleştirmesi

ROC eğrisi

## Model Performansı (Test Seti)
| Metrik      | Değer     |
|-------------|-----------|
| Accuracy    | 0.857     |
| Precision   | 0.736     |
| Recall      | 0.616     |
| F1 Skoru    | 0.671     |
| ROC AUC     | 0.895     |

## Sonuç

Bu çalışmada, **Decision Tree** algoritması kullanılarak **Adult veri seti** üzerinde gelir sınıflandırması yapılmıştır. Uygulanan **veri ön işleme**, **hiperparametre optimizasyonu** ve **model değerlendirme** adımları sonucunda elde edilen performans metrikleri, modelin dengeli ve güvenilir bir sınıflandırma sağladığını göstermektedir. ROC eğrisi ve karar ağacı yapısı da modelin sınıflandırma gücünü görsel olarak desteklemektedir. Yapılan bu analiz, veri madenciliği süreçlerinin dikkatli bir şekilde uygulandığında, doğru sonuçlar üretme potansiyelini açıkça ortaya koymaktadır.
