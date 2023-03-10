import pandas as pd
import sys
import os
import git
import random

def fetch_apachejit(rootPath: str):
    dataDir = os.fsencode(rootPath + "/apachejit/data")

    dfList = []

    totalRows = 0

    for file in os.listdir(dataDir):
        filename = os.fsdecode(file)
        if "commit_links" in filename:
            df = pd.read_csv(rootPath + "/apachejit/data/" + filename)
            df['project'] = "apache/" + df['project'].str.lower()
            #df['owner'] = 'apache'
            df.rename(columns={'project':'repo'},  inplace=True)
            df = df[['fix_hash', 'bug_hash', 'repo']]
            
            dfList.append(df)
            totalRows += df.shape[0]

    #print(dfList[0])
    #print(totalRows)
    df = pd.concat(dfList, ignore_index=True)
    df['Y'] = 1
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
        #[author, repo] = dataDF['repository'].iloc[i].split("/")
        repo = dataDF['repository'].iloc[i]

        # iterate over all entries in the 'bugs' list, usually just one but worth it for when it isn't
        for b in range(len(dataDF.iloc[i]['bugs'])):
            bug_hash = dataDF.iloc[i]['bugs'][b]['commit']['hash']
            rows.append((fix_hash, bug_hash, repo))
            
            
    df = pd.DataFrame(rows, columns=['fix_hash', 'bug_hash', 'repo'])

    df['Y'] = 1

    return df

def getAllPairs(rootPath):
    dfList = []

    #dfList.append(fetch_icse2021(rootPath))
    
    dfList.append(fetch_apachejit(rootPath))

    df = pd.concat(dfList, ignore_index=True)
    df['Y'] = 1

    return df

def negativeRandom(pairs, n):
    numPairs = pairs.shape[0]

    dfList = []
    
    
    for i in range(numPairs):
        newRows = []
        fixSha = pairs.iloc[i]['fix_hash']

        usedIndexes = []
        
        for r in range(n):
            row = {}
            rIndex = random.choice([x for x in range(numPairs) if x != i and x not in usedIndexes])
            
            row["fix_hash"] = fixSha
            row["bug_hash"] = pairs.iloc[rIndex]['bug_hash']
            row['Y'] = 0

            newRows.append(row)

        newDF = pd.DataFrame(newRows)
        dfList.append(newDF)

    df = pd.concat(dfList)

    return df

def getRandomBugFromList(p, li):
    pass
    

def negativeRandomSameRepo(pairs, n):
    numPairs = pairs.shape[0]

    dfList = []

    for i in range(numPairs):
        tempn = n
        newRows = []
        fixSha = pairs.iloc[i]['fix_hash']

        usedIndexes = []

        selectPairs = pairs[(pairs["repo"] == pairs.iloc[i]['repo']) & (pairs['fix_hash'] != pairs.iloc[i]['fix_hash']) & (pairs['bug_hash'] != pairs.iloc[i]['bug_hash'])]


        # if(selectPairs.shape[0] > 0):
        #     print("before", tempn)

        if(selectPairs.shape[0] < n):
            tempn = selectPairs.shape[0]
            
        # if(selectPairs.shape[0] > 0):
        #     print(selectPairs.shape[0], tempn)


        for r in range(tempn):
            row = {}

            rIndex = random.choice([x for x in range(selectPairs.shape[0]) if x not in usedIndexes])

            row["fix_hash"] = fixSha
            row["bug_hash"] = selectPairs.iloc[rIndex]['bug_hash']
            row['Y'] = 0

            #print(row)
            newRows.append(row)

        newDF = pd.DataFrame(newRows)
        dfList.append(newDF)

    df = pd.concat(dfList)


    return df
    
            

def createNegativeExamples(pairs, maxNegatives):
    # Find the n closest commits either ahead or behind the correct commit that edit at least one of the same files as the correct commit
    # Find other commits that we know are bug fixing and edit the same files as the correct commit but are not the correct commit.
    # Commits made after the bug fixing commit
    # Commits that are the same repo but no similar files
    # Commits that are not the same repo
    # Be interesting to see how many bug fixing commits don't reference files in the bug creating commit
    #print(pairs)
    return pd.concat((pairs, negativeRandomSameRepo(pairs, maxNegatives)))

    # first thing is just going to be getting n random 

def main():
    if(len(sys.argv) != 5):
        raise Exception("Wrong number of arguments, USAGE: python createPairs.py DATAFOLDER OUTPUTPATH numSamples negativesPerSample")

    numSamples = int(sys.argv[3])
    numNegatives = int(sys.argv[4])
    rootPath = sys.argv[1]
    outputName = sys.argv[2]

    #df = getAllPairs(rootPath).head(20000).sample(frac=1, random_state=1).reset_index()
    df = fetch_apachejit(rootPath).head(numSamples) 
    df = df.sample(frac=1, random_state=1).reset_index()  # shuffle dataframe

    print("positive examples:", df.shape[0])

    withNegative = createNegativeExamples(df, numNegatives)

    print("negative examples:", withNegative.shape[0] - df.shape[0])
    print("total examples:", withNegative.shape[0])

    if not os.path.exists(os.path.join(rootPath, "pairs_output")):
        os.makedirs(os.path.join(rootPath, "pairs_output"))
    
    withNegative.to_csv(os.path.join(rootPath, "pairs_output", outputName), index=False)



if __name__ == "__main__":
    main()
