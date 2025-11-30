import sys
import time
import pandas as pd
import matplotlib.pyplot as plt

from data_collection.models.model_config_parser import parse_mode_config
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

results = {
    "Algorithm": [],
    "MSE": [],
    "R2": [],
    "Time": []
}

ALGORITHMS = ["Linear Regression","Polynomial Regression","Random Forest Regressor","Neural Network"]

def execute_config(args):
  if (len(args)<=2): #needs config path and data path
    return
  
  config_path = args[1]
  data_path = args[2]

  d = parse_mode_config(config_path=config_path)
  df = pd.read_csv(data_path,index_col=False).fillna(0)

  X = df[d.features]
  y = df[d.targets]

  # X['enviroment'], uniques = pd.factorize(X['enviroment'])
  # X['algorithm'], uniques = pd.factorize(X['algorithm'])

  for algo in ALGORITHMS:
    run_training(X, y, algo)

  plot_results(results)

def run_training(X,y,algorithm):
  start_time = time.time()
  y_pred,y_test = get_model_results(X,y,algorithm)

  mean_squared = mean_squared_error(y_test,y_pred)

  r2 = r2_score(y_test,y_pred)

  elapsed_time = time.time() - start_time

  print(f"Algorithm: {algorithm}")
  print(f"MSE: {mean_squared}")
  print(f"r2: {r2}")
  print("The closer Mean Squared Error is to 0.0, the better it is.")
  print("The closer r2 is to 1.0, the better it is.")
  print(f"Time to train: {elapsed_time} seconds")
  print()

  results["Algorithm"].append(algorithm)
  results["MSE"].append(mean_squared)
  results["R2"].append(r2)
  results["Time"].append(elapsed_time)

def get_model_results(X,y, algorithm):
  match algorithm:
    case "Linear Regression":
      return linear_regression(X,y)
    case "Polynomial Regression":
      return polynomial_regression(X,y)
    case "Random Forest Regressor":
      return random_forest_regressor(X,y)
    case "Neural Network":
      return neural_network_regressor(X,y)

def neural_network_regressor(X,y):
  X_train, X_test, y_train, y_test = preprocess_data(X,y)

  model = MLPRegressor(hidden_layer_sizes=(64,64,64),activation="relu" ,random_state=41, max_iter=20000)
  
  model.fit(X_train,y_train)

  return model.predict(X_test),y_test

def random_forest_regressor(X,y):
  X_train, X_test, y_train, y_test = preprocess_data(X,y)

  model = RandomForestRegressor()
  
  model.fit(X_train,y_train)

  return model.predict(X_test),y_test

def linear_regression(X,y):
  X_train, X_test, y_train, y_test = preprocess_data(X,y)

  model = LinearRegression()
  model.fit(X_train,y_train)

  return model.predict(X_test),y_test

def polynomial_regression(X,y):
  poly = PolynomialFeatures(degree=2, include_bias=False)
  poly_features = poly.fit_transform(X)

  X_train, X_test, y_train, y_test = preprocess_data(poly_features,y)

  model = LinearRegression()
  model.fit(X_train, y_train)
  return model.predict(X_test),y_test

def preprocess_data(X,y):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

  scaler = StandardScaler()

  X_train = scaler.fit_transform(X_train)

  X_test = scaler.transform(X_test)

  return X_train, X_test, y_train, y_test

def plot_results(results):
  df = pd.DataFrame(results)

  plt.figure(figsize=(10, 4))
  plt.subplot(1, 3, 1)
  plt.bar(df["Algorithm"], df["MSE"], color='skyblue')
  plt.title("MSE")
  plt.ylabel("Mean Squared Error")

  plt.subplot(1, 3, 2)
  plt.bar(df["Algorithm"], df["R2"], color='lightgreen')
  plt.title("R2 Score")
  plt.ylabel("R2")

  plt.subplot(1, 3, 3)
  plt.bar(df["Algorithm"], df["Time"], color='salmon')
  plt.title("Training Time (s)")
  plt.ylabel("Time (seconds)")

  plt.tight_layout()
  plt.show()

def main():
  execute_config(sys.argv)

if __name__ == "__main__":
  main()

