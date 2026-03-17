import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

# Now we will import our csv
df = pd.read_csv("./datasets/crop_yield.csv")

# And split the data
y = df['Yield_tons_per_hectare']
X = df.drop('Yield_tons_per_hectare', axis=1)

print(df.columns)
print(df.dtypes)

# First we change the booleans to binary 
X['Fertilizer_Used'] = X['Fertilizer_Used'].astype(int)
X['Irrigation_Used'] = X['Irrigation_Used'].astype(int)

# Now I mapped the columns that are text typed
text_columns = ['Region', 'Soil_Type', 'Crop', 'Weather_Condition']

# For production, we'll basically make the same thing that the get_dummies did, but with OneHotEncoder it makes better, since it stores the categories that were seen during training, unlike dummies
# inside transformers, there will be the name of what we are doing "texts", the HotEncoder that does the same thing as the dummies, saying to drop the first column of each type, just like dummies, and to ignore anything he doesn't know, so it doesnt come up any errors and the list of the text columns, saying the remainder should be ignored, remainder of columns, that we already transformed being boolean or integers or floats
preprocessor = ColumnTransformer(transformers=[('texts', OneHotEncoder(drop='first', handle_unknown='ignore'), text_columns)], remainder='passthrough')

# So now with the Pipeline we can say what are the steps to follow
production = Pipeline(steps=[('data_preprocessing', preprocessor), ('model', LinearRegression())])

# Splitting the data
test_size_param = 0.2
random_state_param = 100
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_param, random_state=random_state_param)
print("I just splitted the data...")

mlflow.set_experiment(experiment_name="Productivity_Prediction")

with mlflow.start_run(run_name="Linear_Regression_Production"):

    production.fit(X_train, y_train)

    y_train_pred = production.predict(X_train)
    y_test_pred = production.predict(X_test)

    train_mse = mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    mlflow.log_param("test_size", test_size_param)
    mlflow.log_param("random_state", random_state_param)
    mlflow.log_param("model_type", "LinearRegression")
    
    mlflow.log_metric("train_mse", train_mse)
    mlflow.log_metric("train_r2", train_r2)
    mlflow.log_metric("test_mse", test_mse)
    mlflow.log_metric("test_r2", test_r2)

    mlflow.sklearn.log_model(production, "production_model")

    print(f"Modelo com R2 de {test_r2:.4f}")