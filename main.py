

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    roc_curve, auc
)

# ---------- 1) ÇIKTI KLASÖRÜ -------------------------------------------------
output_dir = "gorseller"
os.makedirs(output_dir, exist_ok=True)  # Görseller bu klasöre kaydedilecek

# ---------- 2) VERİ OKUMA ve TEMİZLEME ---------------------------------------
cols = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

train = pd.read_csv("adult.data", header=None, names=cols,
                    na_values=" ?", skipinitialspace=True)
test  = pd.read_csv("adult.test", header=0, names=cols,
                    na_values=" ?", skipinitialspace=True, comment="|")

train_clean = train.dropna()   # Eksik satırları sil
test_clean  = test.dropna()

test_clean["income"] = test_clean["income"].str.rstrip(".")  # '.' karakterini at

# Kategorik sütunlardaki boşlukları temizle
cat_cols = ["workclass", "education", "marital-status", "occupation",
            "relationship", "race", "sex", "native-country", "income"]
for col in cat_cols:
    train_clean[col] = train_clean[col].str.strip()
    test_clean[col]  = test_clean[col].str.strip()

# ---------- 3) ONE-HOT ENCODING & HEDEF İKİLİLEŞTİRME ------------------------
X_train = train_clean.drop("income", axis=1)
y_train = train_clean["income"]
X_test  = test_clean.drop("income", axis=1)
y_test  = test_clean["income"]

# Aynı dummy sütun sırasını garanti etmek için birleştir- böl
combined     = pd.concat([X_train, X_test], axis=0)
combined_enc = pd.get_dummies(combined, drop_first=False)

X_train_enc = combined_enc.iloc[:len(X_train)].reset_index(drop=True)
X_test_enc  = combined_enc.iloc[len(X_train):].reset_index(drop=True)

y_train_enc = y_train.map({"<=50K": 0, ">50K": 1})
y_test_enc  = y_test.map({"<=50K": 0, ">50K": 1})

# ---------- 4) HİPERPARAMETRE ARAMALI KARAR AĞACI ----------------------------
dt = DecisionTreeClassifier(random_state=42)
param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 50, 100],
    "min_samples_leaf": [1, 50, 100]
}
grid_search = GridSearchCV(dt, param_grid, cv=5, scoring="f1",
                           n_jobs=-1, verbose=1)
grid_search.fit(X_train_enc, y_train_enc)
best_dt = grid_search.best_estimator_

# ---------- 5) TEST METRİKLERİ ----------------------------------------------
y_test_pred = best_dt.predict(X_test_enc)

print("\n== Test Seti Metrikleri ==")
print(f"Accuracy : {accuracy_score(y_test_enc, y_test_pred):.4f}")
print(f"Precision: {precision_score(y_test_enc, y_test_pred):.4f}")
print(f"Recall   : {recall_score(y_test_enc, y_test_pred):.4f}")
print(f"F1 Score : {f1_score(y_test_enc, y_test_pred):.4f}\n")


# 6) GÖRSELLER 


# ------ 6.1 Histogram: Gelir Dağılımı ---------------------------------------
plt.figure(figsize=(6, 4))
counts = y_train_enc.value_counts().sort_index()
plt.bar(["<=50K", ">50K"], counts, color="steelblue")
plt.title("Fig.1: Eğitim Seti - Gelir Sınıfı Dağılımı")
plt.xlabel("Gelir Sınıfı");  plt.ylabel("Örnek Sayısı")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "fig1_income_histogram.png"))
plt.close()  # belleği boşalt

# ------ 6.2 Korelasyon Isı Haritası -----------------------------------------
continuous_cols = ["age", "fnlwgt", "education-num",
                   "capital-gain", "capital-loss", "hours-per-week"]
corr = train_clean[continuous_cols].corr()

plt.figure(figsize=(6, 5))
plt.imshow(corr, interpolation="nearest", cmap="coolwarm")
plt.title("Fig.2: Sürekli Değişken Korelasyon Matrisi")
plt.colorbar()
ticks = np.arange(len(continuous_cols))
plt.xticks(ticks, continuous_cols, rotation=45, ha="right")
plt.yticks(ticks, continuous_cols)
# Hücrelere korelasyon değeri yaz
for i in range(len(continuous_cols)):
    for j in range(len(continuous_cols)):
        val = f"{corr.iloc[i, j]:.2f}"
        color = "white" if abs(corr.iloc[i, j]) > 0.5 else "black"
        plt.text(j, i, val, ha="center", va="center", color=color)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "fig2_corr_heatmap.png"))
plt.close()

# ------ 6.3 Boxplot’lar ------------------------------------------------------
for col in continuous_cols:
    plt.figure(figsize=(4, 3))
    plt.boxplot(train_clean[col], vert=True)
    plt.title(f"Boxplot: {col}")
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"boxplot_{col}.png"))
    plt.close()

# ------ 6.4 Karışıklık Matrisi ----------------------------------------------
def plot_cm(cm, title, filename):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title(title);  plt.colorbar()
    ticks = np.arange(2)
    plt.xticks(ticks, ["<=50K", ">50K"])
    plt.yticks(ticks, ["<=50K", ">50K"])
    plt.ylabel("Gerçek");  plt.xlabel("Tahmin")
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > thresh else "black"
            plt.text(j, i, cm[i, j], ha="center", va="center", color=color)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

plot_cm(confusion_matrix(y_train_enc, best_dt.predict(X_train_enc)),
        "Karışıklık Matrisi (Eğitim)", "fig3_cm_train.png")
plot_cm(confusion_matrix(y_test_enc, y_test_pred),
        "Karışıklık Matrisi (Test)",    "fig3_cm_test.png")

# ------ 6.5 Özellik Önemleri -------------------------------------------------
importances   = best_dt.feature_importances_
feat_names    = X_train_enc.columns
idx_top10     = np.argsort(importances)[::-1][:10]

plt.figure(figsize=(8, 5))
plt.barh(range(10), importances[idx_top10][::-1], color="seagreen")
plt.yticks(range(10), feat_names[idx_top10][::-1])
plt.xlabel("Önem Puanı")
plt.title("Fig.6: En Önemli 10 Özellik")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "fig6_feature_importance.png"))
plt.close()

# ------ 6.6 Tam Karar Ağacı --------------------------------------------------
from sklearn.tree import plot_tree
plt.figure(figsize=(22, 10))
plot_tree(best_dt,
          feature_names=X_train_enc.columns,
          class_names=["<=50K", ">50K"],
          filled=True, impurity=False, proportion=True,
          rounded=True, fontsize=6)
plt.title("Fig.7: En İyi Karar Ağacının Tam Görünümü")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "fig7_decision_tree.png"), dpi=300)
plt.close()

# ------ 6.7 ROC Eğrisi -------------------------------------------------------
y_proba = best_dt.predict_proba(X_test_enc)[:, 1]  # pozitif sınıf olasılığı
fpr, tpr, _ = roc_curve(y_test_enc, y_proba)
roc_auc     = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, lw=2, label=f"ROC eğrisi (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("Yanlış Pozitif Oranı (FPR)")
plt.ylabel("Doğru Pozitif Oranı (TPR)")
plt.title("Fig.8: Karar Ağacı – ROC Eğrisi")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "fig8_roc_curve.png"))
plt.close()

print(f"Tüm şekiller '{output_dir}' klasörüne kaydedildi.")
