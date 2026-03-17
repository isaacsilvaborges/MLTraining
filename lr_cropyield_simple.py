import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Import dataset
df = pd.read_csv("./datasets/crop_yield.csv") # From kaggle 

# Understanding how many types are in the text or boolean columns 
for column in df.columns:
    if column in ['Region', 'Soil_Type', 'Crop', 'Fertilizer_Used', 'Irrigation_Used', 'Weather_Condition']:
        print(f"{column}: {df[column].unique()}")

# Preparing the data
y = df['Yield_tons_per_hectare']
X = df.drop('Yield_tons_per_hectare', axis=1)
X = pd.get_dummies(X, drop_first=True) # THis is going to create several columns and make the types boolean between 0 and 1
print(X.columns) # You will notice that, even thought theres like 4 regions, he creates 3, cus he makes the first alfabetical one as its pattern, so reduces redundancy
print("I just prepared the data...")

# Splitting the data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
print("I just splitted the data...")

# Building the model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
print("I just built the model...")

# Applicating predictions to the model 
y_lr_train_pred = lr_model.predict(X_train)
y_lr_test_pred = lr_model.predict(X_test)
print("I just predicted the model...")

# Model Evaluating 
lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)
lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)

# Print
lr_results = pd.DataFrame(['Linear Regression', lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
lr_results.columns = ['Method', 'Training MSE', 'Training R2', 'Testing MSE', 'Testing R2']
print(lr_results)

# Test
while True:
    cmd = input("X: ")
    if cmd == "exit":
        break
    X_sample = [float(x) for x in cmd.split()]
    sample = pd.DataFrame([X_sample], columns=X.columns)
    prediction = lr_model.predict(sample)
    print("y: ", prediction[0])
    








