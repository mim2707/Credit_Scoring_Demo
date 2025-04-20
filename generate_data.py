import pandas as pd
import numpy as np
from faker import Faker

# Инициализация Faker с локализацией
fake = Faker('ru_RU')
np.random.seed(42)

# Параметры датасета
n_samples = 10000

# Создаем синтетический датасет
data = {
    'client_id': [fake.uuid4() for _ in range(n_samples)],
    'age': np.random.randint(18, 65, n_samples),
    'region': np.random.choice(['Tashkent', 'Samarkand', 'Fergana', 'Bukhara', 'Andijan'], n_samples),
    'income': np.random.normal(5000000, 2000000, n_samples).clip(min=1000000),  # Доход в сумах
    'debt': np.random.exponential(5000000, n_samples).clip(max=50000000),  # Долги в сумах
    'transactions_monthly': np.random.normal(3000000, 1000000, n_samples).clip(min=0),  # Транзакции
    'mobile_payments': np.random.randint(0, 6, n_samples),  # Платежи за мобильную связь
    'default': np.random.choice([0, 1], n_samples, p=[0.85, 0.15])  # 15% дефолтов
}

# Создаем DataFrame
df = pd.DataFrame(data)

# Сохраняем датасет
df.to_csv('synthetic_credit_data_uz.csv', index=False)
print("Датасет сгенерирован и сохранен как 'synthetic_credit_data_uz.csv'")
print(df.head())