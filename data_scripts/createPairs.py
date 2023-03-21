import pandas as pd
import numpy as np
import sys
import os
import git
import random
from typing import Optional
import cProfile
import tqdm

def getCommitLookup(rootPath: str, maxFiles: int|None = None, maxDiffLines: int|None = None):
    """
    Reads a CSV file with commit data and returns a Pandas DataFrame with selected columns.

    Args:
    - rootPath (str): A string representing the path to the directory containing the CSV file.
    - maxFiles (int, optional): An integer representing the maximum number of files changed in a commit.
    - maxDiffLines (int, optional): An integer representing the maximum number of diff lines in a commit.

    Returns:
    - pd.DataFrame: A Pandas DataFrame with columns sha, repo, files_changed, diff_line_count, and pickle_index.
    """

    df = pd.read_csv(os.path.join(rootPath, 'all_apache_commits.csv'))

    if(maxFiles is not None):
        df = df.loc[df['files_changed'] <= maxFiles]
        
    if(maxDiffLines is not None):
        df = df.loc[df['diff_line_count'] <= maxDiffLines]

    print("commit lookup size:", df.shape[0])

    df.reset_index(drop=True, inplace=True)
    df['pickle_index'] = df.index
    df = df.set_index('sha')

    return df    


def splitPairsAndCombine(df):
    """
    Combines two columns of commit hashes into a single column and returns a new DataFrame.

    Args:
    - df (pd.DataFrame): A Pandas DataFrame with columns bug_hash, repo, and fix_hash.

    Returns:
    - pd.DataFrame: A Pandas DataFrame with columns sha and repo.
    """

    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    df1['sha'] = df['bug_hash']
    df1['repo'] = df['repo']
    df2['sha'] = df['fix_hash']
    df2['repo'] = df['repo']

    return pd.concat([df1, df2])


def fetch_apachejit(rootPath: str):
    """
    Reads multiple CSV files with commit data and returns a new DataFrame with selected columns.

    Args:
    - rootPath (str): A string representing the path to the directory containing the CSV files.

    Returns:
    - pd.DataFrame: A Pandas DataFrame with columns fix_hash, bug_hash, repo, and Y label.
    """
    
    dataDir = os.fsencode(rootPath + "/apachejit/data")
    dfList = []
    totalRows = 0

    for file in os.listdir(dataDir):
        filename = os.fsdecode(file)
        if "commit_links" in filename:
            df = pd.read_csv(rootPath + "/apachejit/data/" + filename)
            df['project'] = "apache-" + df['project'].str.lower()
            #df['owner'] = 'apache'
            df.rename(columns={'project':'repo'},  inplace=True)
            df = df[['fix_hash', 'bug_hash', 'repo']]
            
            dfList.append(df)
            totalRows += df.shape[0]

    df = pd.concat(dfList, ignore_index=True)
    df['Y'] = 1
    return df


def fetch_icse2021(rootPath: str):
    """
    Reads a JSON file with commit data and returns a new DataFrame with selected columns.

    Args:
    - rootPath (str): A string representing the path to the directory containing the JSON file.

    Returns:
    - pd.DataFrame: A Pandas DataFrame with columns fix_hash, bug_hash, repo, and Y.
    """
    
    dataDirPath = rootPath + "/icse2021-szz-replication-package/detailed-database/"
    dataDF = pd.read_json(dataDirPath + "overall.json")
    rows = []

    for i, _ in dataDF.iterrows():
        fix_hash = dataDF['fix'].iloc[i]['commit']['hash']
        repo = dataDF['repository'].iloc[i]

        # Check if there is at least one Java file in the fix commit
        fix_has_java = any(file['lang'] == 'java' for file in dataDF['fix'].iloc[i]['files'])

        if not fix_has_java:
            continue

        for b in range(len(dataDF.iloc[i]['bugs'])):
            bug_hash = dataDF.iloc[i]['bugs'][b]['commit']['hash']

            # Check if there is at least one Java file in the bug commit
            bug_has_java = any(file['lang'] == 'java' for file in dataDF.iloc[i]['bugs'][b]['files'])

            if not bug_has_java:
                continue

            rows.append((fix_hash, bug_hash, repo))

    df = pd.DataFrame(rows, columns=['fix_hash', 'bug_hash', 'repo'])
    df['Y'] = 1

    return df


def fillRow(pairs, rInd, iInd, classification):
    """
    Returns a new dictionary representing a row in a DataFrame.

    Args:
    - pairs (pd.DataFrame): A Pandas DataFrame with columns fix_hash, bug_hash, and repo.
    - rInd (int): An integer representing the index of the row to fill.
    - iInd (int): An integer representing the index of the row to use as the "fix" commit.
    - classification (int): An integer representing the classification of the row (0 or 1).

    Returns:
    - dict: A dictionary representing a row in a DataFrame.
    """
    
    row = {}
    row["fix_hash"] = pairs.iloc[iInd]['fix_hash']
    row['fix_repo'] = pairs.iloc[iInd]['repo']
    row["bug_hash"] = pairs.iloc[rInd]['bug_hash']
    row['bug_repo'] = pairs.iloc[rInd]['repo']
    row['Y'] = classification

    return row


def negativeRandom(pairs, n):
    """
    Returns a new DataFrame with randomly selected "negative" examples.

    Args:
    - pairs (pd.DataFrame): A Pandas DataFrame with columns fix_hash, bug_hash, and repo.
    - n (int): An integer representing the number of negative examples to select for each row in pairs.

    Returns:
    - pd.DataFrame: A Pandas DataFrame with columns fix_hash, bug_hash, and Y label.
    """

    numPairs = pairs.shape[0]
    dfList = []
    
    for i in range(numPairs):
        newRows = []
        fixSha = pairs.iloc[i]['fix_hash']
        usedIndexes = []
        
        for r in range(n):
            row = {}
            rIndex = random.choice([x for x in range(numPairs) if x != i and x not in usedIndexes])
            newRows.append(fillRow(pairs, rIndex, i, 0))

        newDF = pd.DataFrame(newRows)
        dfList.append(newDF)

    df = pd.concat(dfList)

    return df


def negativeRandomSameRepo(pairs, searchPairs, n):
    """
    Returns a new DataFrame with randomly selected "negative" examples that are in the same repository as the "positive" example.

    Args:
    - pairs (pd.DataFrame): A Pandas DataFrame with columns fix_hash, bug_hash, and repo.
    - searchPairs (pd.DataFrame): A Pandas DataFrame with columns fix_hash, bug_hash, and repo
    - n (int): An integer representing the number of negative examples to select for each row in pairs.

    Returns:
    - pd.DataFrame: A Pandas DataFrame with columns fix_hash, bug_hash, and Y label.
    """

    dfList = []
    repoGroups = {}     ## To cache the dataframes that are of the same repo

    for i, row in pairs.iterrows():
        tempn = n
        newRows = []
        
        usedIndexes = []

        i_fixHash = row['fix_hash']
        i_repo = row['repo']
        i_bugHash =  row['bug_hash']

        # If we haven't selected the subset of rows for a given repo then cache them
        if i_repo not in repoGroups:
            repoGroups[i_repo] = searchPairs[(searchPairs["repo"] == i_repo)]

        selectPairs = repoGroups[i_repo].loc[(searchPairs['fix_hash'] != i_fixHash) & (searchPairs['bug_hash'] != i_bugHash)]

        # if there aren't enough valid options to choose n things from then reduce n
        if(selectPairs.shape[0] < n):
            tempn = selectPairs.shape[0]

        for r in range(tempn):

            # Select random index from our valid pairs
            rIndex = random.choice(selectPairs.index)

            newRows.append(fillRow(searchPairs, rIndex, i, 0))

        # Create a dataframe from these new rows and then append that df to a list, one df per "good pair"
        newDF = pd.DataFrame(newRows)
        dfList.append(newDF)

    #concat the whole thing together
    df = pd.concat(dfList)

    return df
    
            
def createNegativeExamples(pairs, searchPairs, maxNegatives):
    """
    Returns a new DataFrame with a combination of "positive" and "negative" examples.

    Args:
    - pairs (pd.DataFrame): A Pandas DataFrame with columns fix_hash, bug_hash, and repo.
    - searchPairs (pd.DataFrame): A Pandas DataFrame with columns fix_hash, bug_hash, and repo.
    - maxNegatives (int): An integer representing the maximum number of "negative" examples to select for each "positive" example.

    Returns:
    - pd.DataFrame: A Pandas DataFrame with columns fix_hash, bug_hash, and Y label.
    """

    # Find the n closest commits either ahead or behind the correct commit that edit at least one of the same files as the correct commit
    # Find other commits that we know are bug fixing and edit the same files as the correct commit but are not the correct commit.
    # Commits made after the bug fixing commit
    # Commits that are the same repo but no similar files
    # Commits that are not the same repo
    # Be interesting to see how many bug fixing commits don't reference files in the bug creating commit
    #print(pairs)
    df = pd.concat((pairs, negativeRandomSameRepo(pairs, searchPairs, maxNegatives)))
    df = df.drop('repo', axis=1).drop('index', axis=1) # get rid of un needed columns

    return df

    # first thing is just going to be getting n random


def getPositivePairs(rootPath,  numSamples: Optional[int] = None):
    """
    Returns a new DataFrame with "positive" examples.

    Args:
    - rootPath (str): A string representing the path to the directory containing the data files.
    - numSamples (int, optional): An integer representing the maximum number of examples to include in the DataFrame.

    Returns:
    - pd.DataFrame: A Pandas DataFrame with columns fix_hash, bug_hash, repo, bug_repo, fix_repo, and Y label.
    """

    dfList = []
    optionsList = []
    repoBlacklist = []

    dfList.append(fetch_icse2021(rootPath))
    dfList.append(fetch_apachejit(rootPath))

    ## Concat all positive pairs
    df = pd.concat(dfList, ignore_index=True)

    #df = splitPairsAndCombine(df)
    df['Y'] = 1

    repoStrings = df['repo'].unique()
    repoDict = {}
    for r in repoStrings:
        try:
            repoDict[r] = git.Repo(os.path.join(rootPath, "../clones", r.replace('/', '-')))
            
        except:
            repoBlacklist.append(r)

    # Don't include repos that aren't clones
    df = df[~df['repo'].isin(repoBlacklist)]

    df['bug_repo'] = df.loc[:, 'repo']
    df['fix_repo'] = df.loc[:, 'repo']

    df = df.drop(['repo'], axis=1)

    if(numSamples is not None):
        df = df.tail(numSamples) # limit number

    return df    


def exportSupervisedCommits(rootPath, outputName):
    """
    Exports a CSV file with "positive" and "negative" examples.

    Args:
    - rootPath (str): A string representing the path to the directory containing the data files.
    - outputName (str): A string representing the name of the output CSV file.
    """

    pairs = getPositivePairs(rootPath)
    df = splitPairsAndCombine(pairs)
    df.to_csv(os.path.join(rootPath,outputName), index=False)
    

def main():
    if(len(sys.argv) != 5):
        raise Exception("Wrong number of arguments, USAGE: python createPairs.py DATAFOLDER OUTPUTPATH numSamples negativesPerSample")

    numSamples = int(sys.argv[3])
    numNegatives = int(sys.argv[4])
    rootPath = sys.argv[1]
    outputName = sys.argv[2]

    optionsList = []
    
    ## Setup positive pairs
    posPairs = getPositivePairs(rootPath)

    all_apache_commits = getCommitLookup(rootPath, maxFiles=8, maxDiffLines=80)

    #bugGood = posPairs[(posPairs['bug_hash'].isin(all_apache_commits.index))]
    #fixGood = posPairs[(posPairs['fix_hash'].isin(all_apache_commits.index))]

    posPairs = posPairs[(posPairs['bug_hash'].isin(all_apache_commits.index)) & (posPairs['fix_hash'].isin(all_apache_commits.index))]

    ## Getting the pickle index for the bug and fix commits
    posPairs = posPairs.merge(all_apache_commits, how='inner', right_on=['sha', 'repo'], left_on=['bug_hash', 'bug_repo'])
    posPairs = posPairs.drop(columns=['repo', 'files_changed', 'diff_line_count'])
    posPairs = posPairs.rename(columns={"pickle_index":"bug_index"})
    
    posPairs = posPairs.merge(all_apache_commits, how='inner', right_on=['sha', 'repo'], left_on=['fix_hash', 'bug_repo'])
    posPairs = posPairs.drop(columns=['repo', 'files_changed', 'diff_line_count'])
    posPairs = posPairs.rename(columns={"pickle_index":"fix_index"})
    
    print("positive examples:", posPairs.shape[0])

    posPairs.to_csv(os.path.join(rootPath, "pairs_output", "apache_positive_pairs.csv"))

    #withNegative = createNegativeExamples(posPairs, allSamples, numNegatives)

    #print("negative examples:", withNegative.shape[0] - posPairs.shape[0])
    #print("total examples:", withNegative.shape[0])

    #if not os.path.exists(os.path.join(rootPath, "pairs_output")):
        #os.makedirs(os.path.join(rootPath, "pairs_output"))
    
    #withNegative.to_csv(os.path.join(rootPath, "pairs_output", outputName), index=False)


if __name__ == "__main__":
    #cProfile.run('main()', sort='cumtime')
    main()
