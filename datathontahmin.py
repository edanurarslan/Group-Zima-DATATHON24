import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# Veri yükleme
train_data = pd.read_csv('C:/Users/90552/Downloads/newprocessed_train.csv')
test_data = pd.read_csv('C:/Users/90552/Downloads/test_x.csv')

# Hedef değişken ve özellikleri ayırma
X = train_data.drop(columns=['Degerlendirme Puani'])
y = train_data['Degerlendirme Puani']

# Veriyi eğitim ve doğrulama için bölme
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model: RandomForestRegressor
rf = RandomForestRegressor(random_state=42)

# Hiperparametre optimizasyonu için GridSearch
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# En iyi model
best_rf = grid_search.best_estimator_

# Doğrulama seti üzerinde tahmin yapma
y_val_pred = best_rf.predict(X_val)

# MSE hesaplama
mse = mean_squared_error(y_val, y_val_pred)
print(f"Validation MSE: {mse}")

# Test setindeki Degerlendirme Puani tahminleri
test_data_scaled = MinMaxScaler().fit_transform(test_data)  # Test verilerini aynı şekilde ölçeklendiriyoruz
predictions = best_rf.predict(test_data_scaled)

# Sonuçları submission.csv dosyasına kaydetme
submission = pd.DataFrame({
    'id': test_data['id'],  # Test verisindeki 'id' sütunu
    'Degerlendirme Puani': predictions
})

submission.to_csv('C:/Users/90552/Downloads/submission.csv', index=False)

print("Submission.csv dosyası oluşturuldu.")

