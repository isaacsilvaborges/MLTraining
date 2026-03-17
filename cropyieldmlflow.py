import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

# Import dataset
df = pd.read_csv("./datasets/crop_yield.csv") # From kaggle 

# Understanding how many types are in the text or boolean columns 
for column in df.columns:
    if column in ['Region', 'Soil_Type', 'Crop', 'Fertilizer_Used', 'Irrigation_Used', 'Weather_Condition']:
        print(f"{column}: {df[column].unique()}")

# Preparing the data
y = df['Yield_tons_per_hectare']
X = df.drop('Yield_tons_per_hectare', axis=1)
X = pd.get_dummies(X, drop_first=True) # This is going to create several columns and make the types boolean between 0 and 1
print(X.columns) # You will notice that, even thought theres like 4 regions, he creates 3, cus he makes the first alfabetical one as its pattern, so reduces redundancy
print("I just prepared the data...")

# Splitting the data
test_size_param = 0.2
random_state_param = 100
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_param, random_state=random_state_param)
print("I just splitted the data...")

mlflow.set_experiment(experiment_name="Productivity_Prediction")

with mlflow.start_run(run_name="Linear_Regression"):

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    print("I just built the model...")

    y_lr_train_pred = lr_model.predict(X_train)
    y_lr_test_pred = lr_model.predict(X_test)

    lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
    lr_train_r2 = r2_score(y_train, y_lr_train_pred)
    lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
    lr_test_r2 = r2_score(y_test, y_lr_test_pred)

    mlflow.log_param("test_size", test_size_param)
    mlflow.log_param("random_state", random_state_param)
    mlflow.log_param("model_type", "LinearRegression")
    
    mlflow.log_metric("train_mse", lr_train_mse)
    mlflow.log_metric("train_r2", lr_train_r2)
    mlflow.log_metric("test_mse", lr_test_mse)
    mlflow.log_metric("test_r2", lr_test_r2)

    mlflow.sklearn.log_model(lr_model, "modelo_salvo")

    print(f"Modelo com R2 de {lr_test_r2:.4f}")