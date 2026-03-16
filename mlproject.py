import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Loading the Dataset
df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/delaney_solubility_with_descriptors.csv')

# Data Preparation
y = df['logS']
X = df.drop('logS', axis=1)

# Data Splitting 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Model Building 
lr = LinearRegression()
lr.fit(X_train, y_train)

# Prediction Application
y_lr_train_pred = lr.predict(X_train)
y_lr_test_pred = lr.predict(X_test)

# Model Evaluation
lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)

lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)

# Print
lr_results = pd.DataFrame(['Linear Regression', lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
lr_results.columns = ['Method', 'Training MSE', 'Training R2', 'Testing MSE', 'Testing R2']
print(lr_results)

while True:
    cmd = input("X: ")
    if cmd == "exit":
        break
    X_sample = [float(x) for x in cmd.split()]
    sample = pd.DataFrame([X_sample], columns=X.columns)
    prediction = lr.predict(sample)
    print("y: ", prediction[0])



