import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from sklearn.preprocessing import StandardScaler

# Load the dataset
data_path = 'Restaurant_revenue.csv'
df = pd.read_csv(data_path)

label_encoder = LabelEncoder()
df['Cuisine_Type'] = label_encoder.fit_transform(df['Cuisine_Type'])

X = df[['Number_of_Customers', 'Menu_Price', 'Marketing_Spend', 'Cuisine_Type', 'Average_Customer_Spending', 'Promotions', 'Reviews']]
y = df['Monthly_Revenue']

sc = StandardScaler()
X_scaled = sc.fit_transform(X)

# Membagi data menjadi data pelatihan dan data uji
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Membuat model Linear Regression
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

# Evaluasi model Linear Regression
accuracy = model_lr.score(X_test, y_test)
print(f'Accuracy: {accuracy:.4f}')

joblib.dump(model_lr, 'linear_regression_model.pkl')
joblib.dump(sc, 'scaler.pkl')


