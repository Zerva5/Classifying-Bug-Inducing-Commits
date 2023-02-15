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

    #print(dfList[0])
    #print(totalRows)
    df = pd.concat(dfList, ignore_index=True)
    return df

def fetch_icse2021(rootPath: str):
    dataDirPath = rootPath + "/icse2021-szz-replication-package/detailed-database/"

    dataDF = pd.read_json(dataDirPath + "overall.json")


    #print(dataDF.columns)
    #print(dataDF.iloc[3]['bugs'])


    # for each fix there will be bugs.
    # Will need to go through each row and

    #df['fix'].iloc[0]['commit']['hash']

    rows = []

    for i in range(dataDF.shape[0]):
        # Since its json is just a bunch of dicts, not very readable :(
        fix_hash = dataDF['fix'].iloc[i]['commit']['hash']
        [author, repo] = dataDF['repository'].iloc[i].split("/")

        # iterate over all entries in the 'bugs' list, usually just one but worth it for when it isn't
        for b in range(len(dataDF.iloc[i]['bugs'])):
            bug_hash = dataDF.iloc[i]['bugs'][b]['commit']['hash']
            rows.append((fix_hash, bug_hash, author, repo))
            
            
    df = pd.DataFrame(rows, columns=['fix_hash', 'bug_hash', 'owner', 'repo'])

    return df

def getAllPairs(rootPath):
    dfList = []

    dfList.append(fetch_icse2021(rootPath))
    
    dfList.append(fetch_apachejit(rootPath))

    df = pd.concat(dfList, ignore_index=True)

    return df

def main():
    if(len(sys.argv) != 3):
        raise Exception("Not enough arguments, USAGE: python createPairs.py DATAFOLDER OUTPUTPATH")


    rootPath = sys.argv[1]

    df = getAllPairs(rootPath)

    # dfList = []

    # dfList.append(fetch_icse2021(rootPath))
    
    # dfList.append(fetch_apachejit(rootPath))

    # df = pd.concat(dfList, ignore_index=True)

    df.to_csv(sys.argv[2], index=False)
    

if __name__ == "__main__":
    main()
