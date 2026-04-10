
from faker import Faker
import random
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

fake=Faker('zh_TW')

scores=[] #放所有分數
data_list = []  # 放所有資料

def task1():
    for i in range(100):
        sleep=random.randint(4,9)
        attendance=random.randint(60,100)
        study=random.randint(5,40)
        score=random.randint(0,100)
        social=random.choice(["low", "medium", "high"])
        scores.append(score) #放分數值
        data_list.append({
            "分數": score,
            "學習時間": study,
            "出勤率": attendance,
            "睡眠時間": sleep,
            "社交活動": social
        })
        print(fake.name(), "：" , str(score), "分，學習時間：", str(study), "小時", "，出勤率：", str(attendance), "%", "，睡眠時間：", str(sleep), "小時", "，社交活動：", social)
        print("-----------")
task1()


df = pd.DataFrame(data_list) #我DataFrame

avg_score=sum(scores)/len(scores)
print("樣本數：", len(scores), "人",",最高分：", max(scores), "分", ",最低分：", min(scores), "分", ",全班平均分數：", avg_score, "分", ",分數標準差：", np.std(scores, ddof=1), "分")


sns.set_theme(style="darkgrid")
plt.rcParams['font.family'] = 'Microsoft JhengHei' #中文
plt.rcParams['axes.unicode_minus'] = False  #負號

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("學生分數分析圖", fontsize=16, fontweight='bold')

# 分數分布圖
sns.histplot(df[["分數"]], bins=20, kde=True, ax=axes[0, 0])
axes[0, 0].set_title("分數分布圖")

# 相關性熱力圖
numeric_cols = df.select_dtypes(include=['float64', 'int64'])  # 選擇數值型列
corr = numeric_cols.corr()  # 算係數矩陣
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=axes[0, 1])
axes[0, 1].set_title("所有欄位與 exam_score 的相關性熱力圖")

# 分數 vs 學習時間
sns.scatterplot(data=df, x="學習時間", y="分數", hue="社交活動", ax=axes[1, 0])
axes[1, 0].set_title("分數 vs 學習時間")

# 分數 vs 社交活動
sns.boxplot(data=df, x="社交活動", y="分數", ax=axes[1, 1])
axes[1, 1].set_title("分數 vs 社交活動")

plt.tight_layout()
plt.show()