import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pickle


with open('assets/perplexity_results.pkl', 'rb') as f:
    data = pickle.load(f)

data = dict(sorted(data.items()))
num_topics_list = list(data.keys())
perplexities = list(data.values())

plt.figure(figsize=(10, 6))
plt.plot(num_topics_list, perplexities, marker='o')
plt.xlabel("Темы")
plt.ylabel("Перплексия")
plt.title("Зависимость перплексии от числа тем")
plt.legend()
plt.grid(True)

best_degree = None
best_r_squared = float('-inf')

for degree in range(1, 6):
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(np.array(num_topics_list).reshape(-1, 1))
    regressor = LinearRegression().fit(X_poly, perplexities)
    predictions = regressor.predict(X_poly)

    current_r_squared = r2_score(perplexities, predictions)

    if current_r_squared > best_r_squared:
        best_r_squared = current_r_squared
        best_degree = degree
        best_predictions = predictions

poly_features_best = PolynomialFeatures(degree=best_degree)
X_poly_best = poly_features_best.fit_transform(np.array(num_topics_list).reshape(-1, 1))
regressor_best = LinearRegression().fit(X_poly_best, perplexities)
predictions_best = regressor_best.predict(X_poly_best)

plt.plot(num_topics_list, predictions_best, color='red', linewidth=2, label=f"Полином ({best_degree}-го порядка)")
plt.title(f"Аппроксимация полиномом {best_degree}-ой степени, r^2: {best_r_squared:.4f}")
plt.savefig('assets/perp_graph.png', dpi=100, bbox_inches='tight')