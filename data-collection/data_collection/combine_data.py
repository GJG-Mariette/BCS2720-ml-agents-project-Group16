import pandas
import sys
import os 

'''
Would combine two csvs with shared column names

syntax

combine_csvs path/to/csv1.csv path/to/csv2.csv
'''

MASTER_DATA = os.path.join("data","master_data.csv")

def combine_data(d1,d2):
    df1 = pandas.read_csv(d1)
    df2 = pandas.read_csv(d2)

    merged_df = pandas.concat([df1, df2], join='inner', ignore_index=True)
    merged_df.to_csv(MASTER_DATA,index=None)

    folder = os.getcwd()

    print("Merged master_csv to: " + os.path.join(folder,MASTER_DATA))

def main():
    try:
        combine_data(sys.argv[1],sys.argv[2])
    except:
        print("Incorect files to be merged, please make sure the first and second paths lead to a csv file")

if __name__ == "__main__":
    main()