import yaml
import sys
import joblib
import pandas as pd
import warnings
from sklearn.exceptions import InconsistentVersionWarning


def read_config(config_path):
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
    d = dict()

    with open(config_path, 'r') as stream:
        try:
            d=yaml.safe_load(stream)
        except yaml.YAMLError as e:
            print(e)

    features = pd.DataFrame([d["Features"]])

    model = joblib.load(filename=d["Model"])
        
    result = model.predict(features)
    print(f"prediction is: {result[0]}")

    
def main():
    read_config(sys.argv[1])

if __name__ == "__main__":
    main()