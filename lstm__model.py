import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ========================
# LOAD DATA
# ========================

data = pd.read_csv("player_transfer_value_with_sentiment.csv")

print("Dataset Loaded")
print(data.head())


# ========================
# FEATURES AND TARGET
# ========================

features = ["goals","assists","shots","dribbles","tackles_total","interceptions"]

X = data[features]

y = data["market_value_eur"]


# ========================
# TRAIN TEST SPLIT
# ========================

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)


# ========================
# MODEL
# ========================

model = LinearRegression()

model.fit(X_train,y_train)


# ========================
# PREDICTIONS
# ========================

predictions = model.predict(X_test)


# ========================
# METRICS
# ========================

mae = mean_absolute_error(y_test,predictions)

rmse = np.sqrt(mean_squared_error(y_test,predictions))

r2 = r2_score(y_test,predictions)

accuracy = r2 * 100


print("\n===== MODEL RESULTS =====")

print("MAE:",mae)

print("RMSE:",rmse)

print("R2 Score:",r2)

print("Model Accuracy:",accuracy,"%")



# ========================
# PREDICTION GRAPH
# ========================

plt.figure(figsize=(8,5))

plt.scatter(y_test,predictions)

plt.xlabel("Actual Market Value")

plt.ylabel("Predicted Market Value")

plt.title("Actual vs Predicted Transfer Value")

plt.show()



# ========================
# ACCURACY GRAPH
# ========================

plt.figure(figsize=(6,4))

plt.bar(["Accuracy"],[accuracy])

plt.ylim(0,100)

plt.title("Model Accuracy")

plt.show()



# ========================
# SAVE JSON FOR DASHBOARD
# ========================

results = []

for i in range(len(predictions)):

    results.append({

        "actual_value":float(y_test.iloc[i]),

        "predicted_value":float(predictions[i])

    })


with open("prediction_data.json","w") as f:

    json.dump(results,f,indent=4)


print("\nPrediction data saved for dashboard")