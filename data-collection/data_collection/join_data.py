import sys
import pandas as pd
import os

COMBINED_DATA = os.path.join("data","combined_data.csv")

def merge(d1,d2,d3):
    df1 = pd.read_csv(d1)
    df2 = pd.read_csv(d2)
    df3 = pd.read_csv(d3)

    df = pd.merge(df3, df2, on="RunID",how="inner")
    df_full = pd.merge(df, df1, on="RunID",how="outer")

    print(df_full)

    df_full.to_csv(COMBINED_DATA,index=None)

    folder = os.getcwd()

    print("Merged combined_csv to: " + os.path.join(folder,COMBINED_DATA))

def main():
    merge(sys.argv[1],sys.argv[2],sys.argv[3])


if __name__ == "__main__":
    main()