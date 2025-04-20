**Credit Scoring Demo for Uzbekistan**

This is a demo AI-based credit scoring model for Uzbekistan, developed to enhance my skills in AI and fintech. It predicts default probability and recommends credit limits using synthetic data.

**Features**

Minimalistic interface for manual data input and CSV uploads.

Filters by default probability and credit limit.

SHAP plots for transparent predictions.

Secure: synthetic data, HTTPS, in-memory processing.

**Technologies**

Python, Streamlit, XGBoost, SHAP, Pandas, NumPy, Matplotlib.

Synthetic data mimicking Uzbek clients.

**Setup**

Install dependencies:

pip install pandas numpy xgboost scikit-learn shap streamlit matplotlib

Place synthetic_credit_data_uz.csv in the project folder.

**Run:**

streamlit run app.py

**Usage**

Manual Input: Enter client data (age, income, etc.) and view predictions.

CSV Upload: Upload a CSV with columns: client_id, age, income, debt, transactions_monthly, mobile_payments. Filter and download results.

Client Analysis: Select a client for detailed results and SHAP plots.

**License and Usage**

This project is for demonstration purposes only. The code and materials cannot be used for any purpose without my explicit permission. Contact me for inquiries.


Try it online: https://creditscoringdemo.streamlit.app/
