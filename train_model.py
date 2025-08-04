# %%
# %pip install --upgrade plotly

# %%


# %%
import pandas as pd
import numpy as np

import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
# TODO: Add missing import statements

# %%
pd.options.display.float_format = '{:,.2f}'.format

# %%
data = pd.read_csv('boston.csv', index_col=0)

# %%
import pandas as pd
from sklearn.datasets import fetch_openml

# 下載 Boston Housing 資料集
boston = fetch_openml(name="boston", version=1, as_frame=True)

# 轉成 DataFrame
data = boston.frame
data.rename(columns={'MEDV': 'PRICE'}, inplace=True)  # 目標欄位

# %%
print("Shape of data:", data.shape)
print("\nColumn names:", data.columns.tolist())
print("\nAny NaN values?\n", data.isnull().sum())
print("\nNumber of duplicates:", data.duplicated().sum())

# %% [markdown]
# ## Descriptive Statistics
# 

# %%
# Descriptive Statistics


avg_students_per_teacher = data['PTRATIO'].mean()
print("Average students per teacher:", avg_students_per_teacher)


avg_price = data['PRICE'].mean()
print("Average house price:", avg_price)

print("CHAS unique values:", data['CHAS'].unique())

if str(data['CHAS'].dtype) == 'category':
    data['CHAS'] = data['CHAS'].astype(int)

print("CHAS min:", data['CHAS'].min(), "max:", data['CHAS'].max())


print("Min rooms:", data['RM'].min(), "Max rooms:", data['RM'].max())


# %%


# %%


# %% [markdown]
# #### House Prices 💰

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml


boston = fetch_openml(name="boston", version=1, as_frame=True)
data = boston.frame
data.rename(columns={"MEDV": "PRICE"}, inplace=True)  # 目標欄位改名為 PRICE


sns.set(style="whitegrid")


features = ['PRICE', 'RM', 'DIS', 'RAD']


fig, axes = plt.subplots(2, 2, figsize=(14, 8))  # 2x2 網格
axes = axes.flatten()

for i, feature in enumerate(features):
    sns.histplot(data[feature], kde=True, ax=axes[i], color="skyblue")
    axes[i].set_title(f"Distribution of {feature}")
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel("Count")

plt.tight_layout()
plt.show()


# %% [markdown]
# #### Distance to Employment - Length of Commute 🚗

# %%
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")

sns.displot(data['DIS'], kde=True, aspect=2, color="skyblue")
plt.title("Distance to Employment - Length of Commute (DIS)")
plt.xlabel("DIS")
plt.ylabel("Count")
plt.show()



# %% [markdown]
# #### Number of Rooms

# %%
# RM: 房間數
sns.displot(data['RM'], kde=True, aspect=2, color="lightgreen")
plt.title("Number of Rooms (RM)")
plt.xlabel("RM")
plt.ylabel("Count")
plt.show()


# %% [markdown]
# #### Access to Highways 🛣

# %%
# RAD: 高速公路可達性
sns.displot(data['RAD'], kde=True, aspect=2, color="salmon")
plt.title("Access to Highways (RAD)")
plt.xlabel("RAD")
plt.ylabel("Count")
plt.show()


# %%
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.renderers.default = 'browser'

if pd.api.types.is_categorical_dtype(data['CHAS']):
    data['CHAS'] = data['CHAS'].astype(str)

data['CHAS_num'] = pd.to_numeric(data['CHAS'], errors='coerce')

data['CHAS_binary'] = (data['CHAS_num'] > 0.5).astype(int)
data['CHAS_label'] = data['CHAS_binary'].map({0: 'No', 1: 'Yes'})

chas_counts = data['CHAS_label'].value_counts().reset_index()
chas_counts.columns = ['Property Located Next to the River?', 'Number of Homes']

fig = px.bar(
    chas_counts,
    x='Property Located Next to the River?',
    y='Number of Homes',
    color='Property Located Next to the River?',
    text='Number of Homes',
    title='Next to Charles River?'
)
fig.update_traces(textposition='outside')
fig.update_layout(showlegend=False)
fig.show()


# %%


# %% [markdown]
# Relationships in the Data

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 假設 data 是你前面讀入的 DataFrame
# 如果你的 DataFrame 名稱不是 data，請改成正確名稱

# 先確認數值型欄位
numeric_cols = data.select_dtypes(include=['number']).columns
print("Numeric columns:", numeric_cols)

# 只挑數值型欄位繪製 pairplot
sns.pairplot(data[numeric_cols])
plt.show()


# %%
sns.pairplot(data[['RM', 'LSTAT', 'NOX', 'DIS', 'PRICE']])
plt.show()


# %% [markdown]
# #### Distance from Employment vs. Pollution
# 
# 

# %%
sns.jointplot(
    data=data,
    x="DIS", 
    y="NOX", 
    kind="scatter", 
    joint_kws={"alpha": 0.5}
)
plt.show()

# %%
sns.jointplot(
    data=data,
    x="INDUS", 
    y="NOX", 
    kind="scatter", 
    joint_kws={"alpha": 0.5}
)

# %%
sns.jointplot(
    data=data,
    x="LSTAT", 
    y="RM", 
    kind="scatter", 
    joint_kws={"alpha": 0.5}
)
plt.show()

# %%
sns.jointplot(
    data=data,
    x="LSTAT", 
    y="PRICE", 
    kind="scatter", 
    joint_kws={"alpha": 0.5}
)
plt.show()

# %%
sns.jointplot(
    data=data,
    x="RM", 
    y="PRICE", 
    kind="scatter", 
    joint_kws={"alpha": 0.5}
)
plt.show()


# %% [markdown]
# # Split Training & Test Dataset
# 
# 
# 

# %%
from sklearn.model_selection import train_test_split

X = data.drop(columns=["PRICE"])

y = data["PRICE"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,       
    random_state=10      
)

print("訓練集大小:", X_train.shape, y_train.shape)
print("測試集大小:", X_test.shape, y_test.shape)

# %%


# %%
X_encoded = pd.get_dummies(X, drop_first=True)
print(X_encoded.dtypes)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_encoded, y)

r2_train = model.score(X_encoded, y)
print("訓練集 R²:", r2_train)


# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd


X = data.drop(columns=["PRICE"])
y = data["PRICE"]


X_encoded = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=10
)


model = LinearRegression()
model.fit(X_train, y_train)


r2_train = model.score(X_train, y_train)
print("訓練集 R²:", r2_train)


# %%
import pandas as pd

# 1. 取得特徵名稱與係數
coeff_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Coefficient": model.coef_
})

# 2. 按絕對值排序，更容易看出影響力
coeff_df["Abs_Coefficient"] = coeff_df["Coefficient"].abs()
coeff_df = coeff_df.sort_values(by="Abs_Coefficient", ascending=False)

print("截距 (intercept):", model.intercept_)
print(coeff_df)


# %%


# %%
import matplotlib.pyplot as plt

y_train_pred = model.predict(X_train)


residuals = y_train - y_train_pred


# %%
plt.figure(figsize=(8,6))
plt.scatter(y_train, y_train_pred, alpha=0.6)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'c', lw=2)  
plt.xlabel("Actual Price ($1000s)")
plt.ylabel("Predicted Price ($1000s)")
plt.title("Actual vs Predicted (Training Set)")
plt.show()


# %%
import seaborn as sns
import numpy as np
from scipy.stats import skew

residual_mean = np.mean(residuals)
residual_skew = skew(residuals)

print("Residual Mean:", residual_mean)
print("Residual Skewness:", residual_skew)


# %%
sns.displot(residuals, kde=True, height=5, aspect=1.5)
plt.title("Distribution of Residuals")
plt.xlabel("Residuals")
plt.show()


# %%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.displot(data['PRICE'], kde=True, height=5, aspect=1.5)
plt.title("Original PRICE Distribution")
plt.show()

orig_skew = data['PRICE'].skew()
print("Original PRICE Skewness:", orig_skew)

log_price = np.log(data['PRICE'])


sns.displot(log_price, kde=True, height=5, aspect=1.5)
plt.title("Log-Transformed PRICE Distribution")
plt.show()

log_skew = log_price.skew()
print("Log PRICE Skewness:", log_skew)


# %%


# %%


# %%
import matplotlib.pyplot as plt


price = data['PRICE']

log_price = np.log(price)

plt.figure(figsize=(6,4))
plt.scatter(price, log_price, alpha=0.7)
plt.xlabel("Original PRICE")
plt.ylabel("Log(PRICE)")
plt.title("Original PRICE vs Log PRICE")
plt.grid(True, alpha=0.3)
plt.show()


# %%
import pandas as pd

X_num = pd.get_dummies(X, drop_first=True)

print(X_num.dtypes)  


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

y_log = np.log(y)

X_train, X_test, y_train_log, y_test_log = train_test_split(
    X_num, y_log, test_size=0.2, random_state=10
)

model_log = LinearRegression()
model_log.fit(X_train, y_train_log)

print("訓練集 R² (log target):", model_log.score(X_train, y_train_log))


# %%


# %%


# %%
import matplotlib.pyplot as plt
import seaborn as sns


y_train_pred = model.predict(X_train)
y_train_log_pred = model_log.predict(X_train)


residuals = y_train - y_train_pred
residuals_log = y_train_log - y_train_log_pred

fig, axes = plt.subplots(2, 2, figsize=(14, 10))


axes[0, 0].scatter(y_train, y_train_pred, color='indigo', alpha=0.6)
axes[0, 0].plot([y_train.min(), y_train.max()],
                [y_train.min(), y_train.max()],
                color='cyan', linewidth=2)
axes[0, 0].set_xlabel('Actual PRICE')
axes[0, 0].set_ylabel('Predicted PRICE')
axes[0, 0].set_title('Original Prices: Actual vs Predicted')


axes[1, 0].scatter(y_train_pred, residuals, color='indigo', alpha=0.6)
axes[1, 0].axhline(0, color='cyan', linewidth=2)
axes[1, 0].set_xlabel('Predicted PRICE')
axes[1, 0].set_ylabel('Residuals')
axes[1, 0].set_title('Original Prices: Residuals vs Predicted')


axes[0, 1].scatter(y_train_log, y_train_log_pred, color='navy', alpha=0.6)
axes[0, 1].plot([y_train_log.min(), y_train_log.max()],
                [y_train_log.min(), y_train_log.max()],
                color='cyan', linewidth=2)
axes[0, 1].set_xlabel('Actual log(PRICE)')
axes[0, 1].set_ylabel('Predicted log(PRICE)')
axes[0, 1].set_title('Log Prices: Actual vs Predicted')


axes[1, 1].scatter(y_train_log_pred, residuals_log, color='navy', alpha=0.6)
axes[1, 1].axhline(0, color='cyan', linewidth=2)
axes[1, 1].set_xlabel('Predicted log(PRICE)')
axes[1, 1].set_ylabel('Residuals')
axes[1, 1].set_title('Log Prices: Residuals vs Predicted')

plt.tight_layout()
plt.show()


# %%


# %%
y_train_log_pred = model_log.predict(X_train)
residuals_log = y_train_log - y_train_log_pred

import numpy as np
from scipy.stats import skew

mean_log_residuals = np.mean(residuals_log)
skew_log_residuals = skew(residuals_log)

print("Residual Mean (log prices):", mean_log_residuals)
print("Residual Skewness (log prices):", skew_log_residuals)

# %% [markdown]
# 

# %%


# %%
from sklearn.metrics import r2_score

y_test_pred = model.predict(X_test)
r2_test_original = r2_score(y_test, y_test_pred)

y_test_log_pred = model_log.predict(X_test)
r2_test_log = r2_score(y_test_log, y_test_log_pred)

print("Test R² (original prices):", r2_test_original)
print("Test R² (log prices):", r2_test_log)


# %%


# %%
import pandas as pd
import numpy as np

features = data.drop(['PRICE'], axis=1)

features_numeric = features.apply(pd.to_numeric, errors='coerce')

average_vals = features_numeric.mean().values

property_stats = pd.DataFrame(
    data=average_vals.reshape(1, len(features_numeric.columns)),
    columns=features_numeric.columns
)

print(property_stats)


# %%
import numpy as np
import pandas as pd


feature_names = model_log.feature_names_in_


avg_features = pd.DataFrame(X, columns=feature_names).mean()


avg_features = avg_features.fillna(0).values.reshape(1, -1)

print("Model expects features:", model_log.n_features_in_)
print("Avg features shape:", avg_features.shape)


log_price_pred = model_log.predict(avg_features)
print("Predicted log price:", log_price_pred[0])


price_pred = np.exp(log_price_pred)
print("Predicted price ($):", price_pred[0])


# %%
# 房屋特徵
next_to_river = True                   # 是否臨河
nr_rooms = 8                           # 房間數
students_per_classroom = 20            # 每班學生數
distance_to_town = 5                   # 距離市中心
pollution = data.NOX.quantile(q=0.75)  # 高污染
amount_of_poverty = data.LSTAT.quantile(q=0.25)  # 低貧困


# %%
import numpy as np
import pandas as pd

X_encoded = pd.get_dummies(X) 
feature_names = model_log.feature_names_in_

# 2️⃣ 建立平均特徵 DataFrame
avg_features = pd.DataFrame([X_encoded.mean()], columns=X_encoded.columns)

for col in feature_names:
    if col not in avg_features.columns:
        avg_features[col] = 0

avg_features = avg_features[feature_names]


avg_features["CHAS_1"] = 1 if next_to_river else 0        #
avg_features["RM"] = nr_rooms                             #
avg_features["PTRATIO"] = students_per_classroom          # 
avg_features["DIS"] = distance_to_town                    # 
avg_features["NOX"] = data.NOX.quantile(q=0.75)           # 
avg_features["LSTAT"] = data.LSTAT.quantile(q=0.25)       # 


log_price_pred = model_log.predict(avg_features)[0]
print("Predicted log price:", log_price_pred)


price_pred = np.exp(log_price_pred)
print("Predicted price ($):", price_pred)


# %%
def predict_custom_house(next_to_river, nr_rooms, students_per_classroom, 
                         distance_to_town, pollution_level, poverty_level):
    # 建立一個輸入特徵 DataFrame
    input_data = pd.DataFrame([{
        'next_to_river': 1 if next_to_river else 0,
        'nr_rooms': nr_rooms,
        'students_per_classroom': students_per_classroom,
        'distance_to_town': distance_to_town,
        'pollution_level': 1 if pollution_level == 'high' else 0,
        'poverty_level': 1 if poverty_level == 'high' else 0
    }])

    # 使用你訓練好的模型預測
    prediction = model.predict(input_data)[0]
    return prediction


# %%
import os
import joblib

os.makedirs("../models", exist_ok=True)


joblib.dump(model, "../models/model.pkl")


feature_names = X_train.columns.tolist()
joblib.dump(feature_names, "../models/feature_names.pkl")

print("✅ 模型與特徵名稱已經成功保存！")


# %%
import numpy as np
import pandas as pd
import joblib


model = joblib.load("../models/model.pkl")
feature_names = joblib.load("../models/feature_names.pkl")

def predict_custom_house(next_to_river, nr_rooms, students_per_classroom,
                         distance_to_town, pollution_level, poverty_level):

    input_data = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)


    if 'nr_rooms' in input_data.columns:
        input_data['nr_rooms'] = nr_rooms
    if 'students_per_classroom' in input_data.columns:
        input_data['students_per_classroom'] = students_per_classroom
    if 'distance_to_town' in input_data.columns:
        input_data['distance_to_town'] = distance_to_town

    if 'next_to_river' in input_data.columns:
        input_data['next_to_river'] = 1 if next_to_river else 0

    for level in ['high', 'low']:
        col = f'pollution_level_{level}'
        if col in input_data.columns:
            input_data[col] = 1 if pollution_level == level else 0

    for level in ['high', 'low']:
        col = f'poverty_level_{level}'
        if col in input_data.columns:
            input_data[col] = 1 if poverty_level == level else 0

 
    prediction = model.predict(input_data)[0]
    return prediction


# %%
price = predict_custom_house(
    next_to_river=True,
    nr_rooms=8,
    students_per_classroom=20,
    distance_to_town=5,
    pollution_level='high',
    poverty_level='low'
)

print(f"預測價格: {price:.2f} 美元")


# %%
import os

print(os.getcwd())  

# %%
import glob

print(glob.glob("*.pkl"))

# %%
import joblib
import os


os.makedirs("models", exist_ok=True)


joblib.dump(model, "models/model.pkl")
joblib.dump(feature_names, "models/feature_names.pkl")



