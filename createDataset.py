import pandas as pd
import sys
import os

def fetch_apachejit(rootPath: str):
    dataDir = os.fsencode(rootPath + "/apachejit/data")

    dfList = []

    totalRows = 0

    for file in os.listdir(dataDir):
        filename = os.fsdecode(file)
        if "commit_links" in filename:
            df = pd.read_csv(rootPath + "/apachejit/data/" + filename)
            df['project'] = df['project'].str.lower()
            df['owner'] = 'apache'
            df.rename(columns={'project':'repo'},  inplace=True)
            df = df[['fix_hash', 'bug_hash', 'owner', 'repo']]
            
            dfList.append(df)
            totalRows += df.shape[0]

    print(dfList[0])
    print(totalRows)
    df = pd.concat(dfList, ignore_index=True)
    return df

def fetch_icse2021(rootPath: str):
    dataDirPath = rootPath + "/icse2021-szz-replication-package/detailed-database/"

    df = pd.read_json(dataDirPath + "overall.json")

    print(df.columns)

    # only care about 

    for i in range(7):
        print(df.iloc[i])

    return [df]


def main():
    if(len(sys.argv) != 2):
        raise Exception("Not enough arguments")


    rootPath = sys.argv[1]

    #fetch_icse2021(rootPath)
    fetch_apachejit(rootPath)
    

if __name__ == "__main__":
    main()
