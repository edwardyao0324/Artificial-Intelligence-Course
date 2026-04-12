import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy.stats import gaussian_kde
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

plt.rcParams['font.family'] = 'Microsoft JhengHei'
plt.rcParams['axes.unicode_minus'] = False

FILE = r"C:\Users\Edward\Desktop\Artificial Intelligence Course\synthetic_ehr_500.xlsx"  

# 載入資料
df = pd.read_excel(FILE)

# 數值欄位（排除 ID、日期、目標）
NUM_FEATURES = [
    "age", "bmi", "systolic_bp", "diastolic_bp", "heart_rate",
    "respiratory_rate", "temperature_c", "spo2",
    "diabetes", "hypertension", "ckd", "copd",
    "hemoglobin_g_dl", "wbc_k_ul", "platelet_k_ul",
    "creatinine_mg_dl", "sodium_mmol_l", "potassium_mmol_l",
    "glucose_mg_dl", "crp_mg_l",
    "icu_transfer", "readmission_30d"
]
TARGET = "length_of_stay_days"

# 類別欄位 One-Hot Encoding
CAT_FEATURES = ["admission_type", "department", "sex",
                "smoking_status", "primary_dx_group", "discharge_disposition"]

df_model = df[NUM_FEATURES + CAT_FEATURES + [TARGET]].copy()
df_model = pd.get_dummies(df_model, columns=CAT_FEATURES, drop_first=True)

# 補缺失值（數值欄位用中位數，類別欄位用眾數）(可有可無)
df_model.fillna(df_model.median(numeric_only=True), inplace=True)

X = df_model.drop(columns=[TARGET]) # 特徵
y = df_model[TARGET] # 目標

# ── Train/Test Split（80/20，固定 random_state）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"Train size: {len(X_train)}  |  Test size: {len(X_test)}")
print(f"Features: {X.shape[1]}")

# 模型選擇
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest":     RandomForestRegressor(
                            n_estimators=200,
                            max_depth=8,
                            min_samples_leaf=5,
                            random_state=42,
                            n_jobs=-1),
    "XGBoost":           XGBRegressor(
                            n_estimators=300,
                            max_depth=5,
                            learning_rate=0.05,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            random_state=42,
                            verbosity=0,
                            device="cuda",
                            tree_method="hist"),
}

# 訓練、評估
results = {}

for name, model in models.items():
    # Linear Regression 需標準化
    if name == "Linear Regression":
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_train)
        Xte = scaler.transform(X_test)
    else:
        Xtr, Xte = X_train, X_test

    model.fit(Xtr, y_train)
    y_pred_train = model.predict(Xtr)
    y_pred_test  = model.predict(Xte)

    results[name] = {
        "Train MSE": mean_squared_error(y_train, y_pred_train),
        "Test MSE":  mean_squared_error(y_test,  y_pred_test),
        "Train MAE": mean_absolute_error(y_train, y_pred_train),
        "Test MAE":  mean_absolute_error(y_test,  y_pred_test),
        "Test R²":   r2_score(y_test, y_pred_test),
        "y_pred":    y_pred_test,   # 保留預測值供畫圖
    }


# 評估報告表格
metric_cols = ["Train MSE", "Test MSE", "Train MAE", "Test MAE", "Test R²"]
report = pd.DataFrame(results).T[metric_cols].round(4)

print("\n" + "=" * 65)
print("  模型評估報告")
print("=" * 65)
print(report.to_string())
print("=" * 65)

# 圖
# EDA 圖一：缺失值水平長條圖
miss = df.isnull().sum()
miss_pct = miss / len(df) * 100
miss_df = (pd.DataFrame({"count": miss, "pct": miss_pct})
           .query("count > 0")
           .sort_values("pct"))

fig_miss, ax_miss = plt.subplots(figsize=(8, max(3, len(miss_df) * 0.5)))
fig_miss.suptitle("各欄位缺失值比例", fontsize=14, fontweight="bold")
bars = ax_miss.barh(miss_df.index, miss_df["pct"],
                    color=sns.color_palette("RdYlGn_r", len(miss_df)))
ax_miss.set_xlabel("缺失比例 (%)")
for bar, (cnt, pct) in zip(bars, zip(miss_df["count"], miss_df["pct"])):
    ax_miss.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                 f"{cnt} 筆 ({pct:.1f}%)", va="center", fontsize=9)
fig_miss.tight_layout()

# EDA 圖二：LOS 分布（直方圖 + KDE）
los = df[TARGET].dropna()
fig_los, ax_los = plt.subplots(figsize=(9, 5))
fig_los.suptitle("住院天數分布（直方圖 + KDE）", fontsize=14, fontweight="bold")
ax_los.hist(los, bins=20, density=True,
            color="#4C72B0", edgecolor="white", alpha=0.75, label="直方圖")
kde = gaussian_kde(los)
import numpy as np
x_range = np.linspace(los.min(), los.max(), 300)
ax_los.plot(x_range, kde(x_range), color="#DD8452", lw=2.5, label="KDE")
ax_los.axvline(los.mean(),   color="red",   ls="--", lw=1.5,
               label=f"平均數 = {los.mean():.1f}")
ax_los.axvline(los.median(), color="green", ls=":",  lw=1.5,
               label=f"中位數 = {los.median():.1f}")
ax_los.set_xlabel("住院天數 (天)")
ax_los.set_ylabel("密度")
ax_los.legend()
fig_los.tight_layout()

# EDA 圖三：數值特徵相關係數熱力圖
num_cols = df[NUM_FEATURES + [TARGET]].select_dtypes(include=[np.number]).columns
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
fig_heat, ax_heat = plt.subplots(figsize=(14, 11))
fig_heat.suptitle("數值型特徵相關矩陣", fontsize=14, fontweight="bold")
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.4, annot_kws={"size": 7}, ax=ax_heat)
ax_heat.tick_params(axis="x", rotation=45, labelsize=8)
ax_heat.tick_params(axis="y", rotation=0,  labelsize=8)
fig_heat.tight_layout()

# EDA 圖四：Boxplot 各診斷群住院天數
order = (df.groupby("primary_dx_group")[TARGET]
         .median()
         .sort_values(ascending=False)
         .index.tolist())
fig_box, ax_box = plt.subplots(figsize=(13, 6))
fig_box.suptitle("各診斷群住院天數分布（依中位數由高到低）",
                 fontsize=14, fontweight="bold")
sns.boxplot(data=df, x="primary_dx_group", y=TARGET, order=order,
            palette="tab10", flierprops={"marker": "o", "markersize": 4, "alpha": 0.5},
            ax=ax_box)
sns.stripplot(data=df, x="primary_dx_group", y=TARGET, order=order,
              color="gray", size=2.5, alpha=0.35, jitter=True, ax=ax_box)
ax_box.set_xlabel("診斷群")
ax_box.set_ylabel("住院天數 (天)")
ax_box.tick_params(axis="x", rotation=25)
medians = df.groupby("primary_dx_group")[TARGET].median()
for i, grp in enumerate(order):
    ax_box.text(i, medians[grp] + 0.3, f"{medians[grp]:.0f}天",
                ha="center", va="bottom", fontsize=8, fontweight="bold")
fig_box.tight_layout()

palette = ["#4C72B0", "#55A868", "#DD8452"]
model_names = list(results.keys())
model_names_zh = ["線性回歸", "隨機森林", "XGBoost"]

# 測試集指標長條圖
fig1, axes1 = plt.subplots(1, 3, figsize=(14, 5))
fig1.suptitle("三個模型測試集指標比較", fontsize=14, fontweight="bold")

metrics_info = [
    ("Test MSE", "測試集 MSE（均方誤差）",   "MSE 值"),
    ("Test MAE", "測試集 MAE（平均絕對誤差）","MAE 值"),
    ("Test R²",  "測試集 R²（決定係數）",    "R² 值"),
]
for ax, (key, title, ylabel) in zip(axes1, metrics_info):
    vals = [results[m][key] for m in model_names]
    bars = ax.bar(model_names_zh, vals, color=palette, edgecolor="white", width=0.5)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticklabels(model_names_zh, rotation=0, ha="center", fontsize=10)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.015,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10)

fig1.tight_layout()

# Actual vs Predicted 
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
fig2.suptitle("預測值 vs 實際值（測試集）", fontsize=14, fontweight="bold")

for ax, name, name_zh, color in zip(axes2, model_names, model_names_zh, palette):
    y_pred = results[name]["y_pred"]
    ax.scatter(y_test, y_pred, alpha=0.5, s=30, color=color, edgecolors="none")
    lo, hi = y_test.min() - 0.5, y_test.max() + 0.5
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="完美預測線")
    ax.set_title(f"{name_zh}　R²={results[name]['Test R²']:.3f}", fontsize=11)
    ax.set_xlabel("實際住院天數", fontsize=10)
    ax.set_ylabel("預測住院天數", fontsize=10)
    ax.legend(fontsize=9)

fig2.tight_layout()

# Feature Importance（隨機森林 & XGBoost）
fig3, axes3 = plt.subplots(1, 2, figsize=(16, 7))
fig3.suptitle("特徵重要性（前 15 名）", fontsize=14, fontweight="bold")

for ax, name, name_zh, color in zip(
        axes3,
        ["Random Forest", "XGBoost"],
        ["隨機森林", "XGBoost"],
        ["#4C72B0", "#DD8452"]):
    imp = pd.Series(models[name].feature_importances_, index=X.columns)
    top15 = imp.sort_values(ascending=True).tail(15)
    top15.plot.barh(ax=ax, color=color)
    ax.set_title(f"{name_zh} — 前 15 重要特徵", fontsize=12)
    ax.set_xlabel("重要性分數", fontsize=10)
    ax.tick_params(labelsize=9)

fig3.tight_layout()

plt.show()      


print(report[["Train MSE", "Test MSE", "Train MAE", "Test MAE", "Test R²"]].round(3).to_string())