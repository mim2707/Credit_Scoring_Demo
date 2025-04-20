import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import shap
import matplotlib.pyplot as plt
import io
import base64

# Функция для расчета кредитного лимита
def calculate_credit_limit(income, default_proba):
    base_limit = income * 0.4  # 40% от дохода
    risk_adjustment = 1 - default_proba
    return base_limit * risk_adjustment

# Загрузка и подготовка данных
@st.cache_data
def load_and_train_model():
    # Загружаем синтетические данные
    df = pd.read_csv('synthetic_credit_data_uz.csv')
    
    # Подготовка данных
    X = df[['age', 'income', 'debt', 'transactions_monthly', 'mobile_payments']]
    y = df['default']
    
    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Обучение модели
    model = xgb.XGBClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Оценка модели
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    
    return model, X_train, auc

# Функция для создания SHAP-графика
def plot_shap(model, input_data):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)
    
    # Создаем SHAP summary plot
    plt.figure()
    shap.summary_plot(shap_values, input_data, show=False)
    
    # Сохраняем график в память
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_str

# Основной код Streamlit
st.title("Демо кредитного скоринга для Узбекистана")
st.write("Введите данные клиента, чтобы оценить вероятность дефолта и предложить кредитный лимит.")

# Загрузка модели
model, X_train, auc = load_and_train_model()
st.write(f"Точность модели (AUC-ROC): {auc:.2f}")

# Ввод данных клиента
st.header("Данные клиента")
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Возраст", 18, 65, 30)
    income = st.number_input("Доход (в сумах)", min_value=1000000, max_value=20000000, value=5000000, step=100000)
    debt = st.number_input("Долги (в сумах)", min_value=0, max_value=50000000, value=0, step=100000)

with col2:
    transactions = st.number_input("Транзакции в месяц (в сумах)", min_value=0, max_value=10000000, value=3000000, step=100000)
    mobile_payments = st.slider("Платежи за мобильную связь (раз в месяц)", 0, 5, 2)

# Подготовка данных для предсказания
input_data = pd.DataFrame({
    'age': [age],
    'income': [income],
    'debt': [debt],
    'transactions_monthly': [transactions],
    'mobile_payments': [mobile_payments]
})

# Загрузка CSV-файла
st.header("Или загрузите CSV-файл")
uploaded_file = st.file_uploader("Загрузите CSV с данными клиентов", type="csv")
if uploaded_file:
    df_uploaded = pd.read_csv(uploaded_file)
    # Проверка, что CSV содержит нужные столбцы
    required_columns = ['age', 'income', 'debt', 'transactions_monthly', 'mobile_payments']
    if all(col in df_uploaded.columns for col in required_columns):
        X_uploaded = df_uploaded[required_columns]
        predictions = model.predict_proba(X_uploaded)[:, 1]
        df_uploaded['default_proba'] = predictions
        df_uploaded['credit_limit'] = df_uploaded.apply(
            lambda x: calculate_credit_limit(x['income'], x['default_proba']), axis=1
        )
        st.write("Результаты для загруженных клиентов:")
        # Показываем client_id, если он есть
        display_columns = ['client_id'] + required_columns + ['default_proba', 'credit_limit'] if 'client_id' in df_uploaded.columns else required_columns + ['default_proba', 'credit_limit']
        
        # Фильтрация по минимальной вероятности дефолта и кредитному лимиту
        min_proba = st.slider("Минимальная вероятность дефолта", 0.0, 1.0, 0.0)
        filtered_df = df_uploaded[df_uploaded['default_proba'] >= min_proba]
        min_limit = st.slider("Минимальный кредитный лимит (сум)", 0, 10000000, 0)
        filtered_df = filtered_df[filtered_df['credit_limit'] >= min_limit]
        st.dataframe(filtered_df[display_columns])
        
        # Кнопка для скачивания отфильтрованных результатов
        csv = filtered_df.to_csv(index=False)
        st.download_button("Скачать результаты", csv, "results.csv", "text/csv")
        
        # Выбор клиента из загруженного CSV
        st.subheader("Выберите клиента для детального анализа")
        if 'client_id' in filtered_df.columns:
            client_options = filtered_df['client_id'].tolist()
            selected_client = st.selectbox("Выберите клиента по ID", client_options) if client_options else st.write("Нет клиентов, соответствующих фильтру")
            if client_options:
                selected_data = filtered_df[filtered_df['client_id'] == selected_client]
        else:
            client_options = filtered_df.index.tolist()
            selected_client = st.selectbox("Выберите клиента по номеру строки", client_options) if client_options else st.write("Нет клиентов, соответствующих фильтру")
            if client_options:
                selected_data = filtered_df.loc[[selected_client]]
        
        if client_options and not selected_data.empty:
            input_data_selected = selected_data[required_columns]
            default_proba = selected_data['default_proba'].iloc[0]
            credit_limit = selected_data['credit_limit'].iloc[0]
            
            st.header("Результаты для выбранного клиента")
            st.write(f"**Вероятность дефолта**: {default_proba:.2%}")
            st.write(f"**Рекомендуемый кредитный лимит**: {credit_limit:,.0f} сум")
            
            # SHAP-визуализация для выбранного клиента
            st.subheader("Интерпретация решения")
            st.write("График ниже показывает, как каждый фактор влияет на предсказание:")
            img_str = plot_shap(model, input_data_selected)
            st.image(f"data:image/png;base64,{img_str}")
    else:
        st.error(f"CSV-файл должен содержать столбцы: {', '.join(required_columns)}")

# Кнопка для предсказания
if st.button("Оценить клиента"):
    # Предсказание
    default_proba = model.predict_proba(input_data)[:, 1][0]
    credit_limit = calculate_credit_limit(income, default_proba)
    
    # Вывод результатов
    st.header("Результаты")
    st.write(f"**Вероятность дефолта**: {default_proba:.2%}")
    st.write(f"**Рекомендуемый кредитный лимит**: {credit_limit:,.0f} сум")
    
    # SHAP-визуализация
    st.header("Интерпретация решения")
    st.write("График ниже показывает, как каждый фактор влияет на предсказание:")
    img_str = plot_shap(model, input_data)
    st.image(f"data:image/png;base64,{img_str}")

# Дополнительная информация
st.header("О модели")
st.write("""
Эта демо-версия использует модель XGBoost, обученную на синтетических данных, имитирующих клиентов в Узбекистане. 
Модель оценивает вероятность дефолта и предлагает кредитный лимит на основе дохода и риска.
""")