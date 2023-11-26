import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt

df1=pd.read_excel("C:/Users/sampa/Desktop/Output_Synthetic.xlsx")


df1['Confidence_Score'] = 0.2 * df1['likes_count_normalized'] + \
                          0.2 * df1['comments_count_normalized'] + \
                          0.2 * df1['shares_count_normalized'] + \
                          0.1 * df1['Combined_Sentiment_nltk'] + \
                          0.3 * df1['Scaled Rank']

Conf = df1.groupby('source')['Confidence_Score'].mean().reset_index()
Conf = Conf.sort_values(by='Confidence_Score')

Conf.to_csv("C:/Users/sampa/Desktop/conf_sources.xlsx")


plt.figure(figsize=(12, 6))
plt.bar(Conf['source'], Conf['Confidence_Score'], color='gray')
plt.xlabel('Source')
plt.ylabel('Mean Confidence Score')
plt.title('Source vs. Confidence Score')
plt.xticks(rotation=45, ha='right')
plt.show()


print(Conf)

df1['likes_count_normalized'] = df1['likes_count_normalized'].fillna(df1['likes_count_normalized'].mean())
df1['comments_count_normalized'] = df1['comments_count_normalized'].fillna(df1['comments_count_normalized'].mean())
df1['shares_count_normalized'] = df1['shares_count_normalized'].fillna(df1['shares_count_normalized'].mean())
df1['Combined_Sentiment_nltk'] = df1['Combined_Sentiment_nltk'].fillna(df1['Combined_Sentiment_nltk'].mean())
df1['Scaled Rank'] = df1['Scaled Rank'].fillna(df1['Scaled Rank'].mean())
df1['Confidence_Score']= df1['Confidence_Score'].fillna(df1['Confidence_Score'].mean())




X = df1[['likes_count_normalized', 'comments_count_normalized', 'shares_count_normalized', 'Combined_Sentiment_nltk', 'Scaled Rank']]
y = df1['Confidence_Score']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


mae = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nLinear Regression")
print(f"Mean Squared Error (MSE): {mae:.2f}")
print(f"R-squared (R2): {r2:.2f}")



decision_tree_model = DecisionTreeRegressor()

decision_tree_model.fit(X_train, y_train)

y_pred_dt = decision_tree_model.predict(X_test)


random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)

random_forest_model.fit(X_train, y_train)

y_pred_rf = random_forest_model.predict(X_test)


mae_dt = mean_squared_error(y_test, y_pred_dt)

r2_dt = r2_score(y_test, y_pred_dt)


mae_rf = mean_squared_error(y_test, y_pred_rf)

r2_rf = r2_score(y_test, y_pred_rf)

print("\nDecision Tree Regression:")
print(f"Mean Squared Error (MSE): {mae_dt:.2f}")
print(f"R-squared (R2): {r2_dt:.2f}")

print("\nRandom Forest Regression:")
print(f"Mean Squared Error (MSE): {mae_rf:.2f}")
print(f"R-squared (R2): {r2_rf:.2f}")


knn_model = KNeighborsRegressor(n_neighbors=5)

knn_model.fit(X_train, y_train)

y_pred_knn = knn_model.predict(X_test)


mae_knn = mean_squared_error(y_test, y_pred_knn)

r2_knn = r2_score(y_test, y_pred_knn)

print("\nK-Nearest Neighbors (KNN) Regression:")
print(f"Mean Squared Error (MSE): {mae_knn:.2f}")
print(f"R-squared (R2): {r2_knn:.2f}")



model_names = ["Linear Regression", "Decision Tree", "Random Forest", "K-Nearest Neighbors"]

r2_scores = [r2, r2_dt, r2_rf, r2_knn]
mse_values = [mae, mae_dt, mae_rf, mae_knn]
sorted_r2_scores, sorted_mse_values, sorted_model_names = zip(*sorted(zip(r2_scores, mse_values, model_names)))

plt.figure(figsize=(10, 5))
plt.bar(sorted_model_names, sorted_r2_scores, color='green')
plt.xlabel('Regression Models')
plt.ylabel('R-squared (R2)')
plt.title('R2 Score Comparison for Regression Models')
plt.ylim(0.9, 1)
plt.show()


plt.figure(figsize=(10, 5))
plt.bar(sorted_model_names, sorted_mse_values, color='orange')
plt.xlabel('Regression Models')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('MSE Comparison for Regression Models')
plt.show()


