import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# CSV dosyasını okuma
train_data = pd.read_csv('C:/Users/90552/Downloads/train.csv', low_memory=False)

# Universite Not Ortalamasi sütununda aralıklı değerlerin ortalamasını hesaplama
def calculate_average(value):
    if isinstance(value, str) and '-' in value:
        try:
            lower, upper = value.split('-')
            average = (float(lower.strip()) + float(upper.strip())) / 2
            return average
        except ValueError:
            return None
    return value

train_data['Universite Not Ortalamasi'] = train_data['Universite Not Ortalamasi'].apply(calculate_average)
train_data['Universite Not Ortalamasi'] = pd.to_numeric(train_data['Universite Not Ortalamasi'], errors='coerce')
train_data['Universite Not Ortalamasi'] = train_data['Universite Not Ortalamasi'].fillna(train_data['Universite Not Ortalamasi'].mean())

# Sayısal veriler için ortalama ile doldurma (Degerlendirme Puani hariç)
numeric_cols = train_data.select_dtypes(include=['float64', 'int64']).columns.drop('Degerlendirme Puani')
numeric_imputer = SimpleImputer(strategy='mean')
train_data[numeric_cols] = numeric_imputer.fit_transform(train_data[numeric_cols])

# Kategorik veriler için en sık görülen değerle doldurma
categorical_cols = train_data.select_dtypes(include=['object']).columns
categorical_imputer = SimpleImputer(strategy='most_frequent')
train_data[categorical_cols] = categorical_imputer.fit_transform(train_data[categorical_cols])

# Kategorik sütunları etiketleme (Label Encoding)
for col in categorical_cols:
    encoder = LabelEncoder()
    train_data[col] = encoder.fit_transform(train_data[col].astype(str))

# **Normalizasyon** (MinMaxScaler ile, Degerlendirme Puani hariç)
scaler = MinMaxScaler()
train_data[numeric_cols] = scaler.fit_transform(train_data[numeric_cols])

# Aykırı değerlerden temizleme (opsiyonel, sayısal sütunlar için uygulanabilir)
def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

# Aykırı değer temizleme adımı (opsiyonel, gerekirse kullanabilirsin)
for col in numeric_cols:
    train_data = remove_outliers_iqr(train_data, col)

# Veri setini kaydetme
train_data.to_csv('C:/Users/90552/Downloads/newprocessed_train.csv', index=False)

print("Veri ön işleme tamamlandı ve dosya kaydedildi.")



