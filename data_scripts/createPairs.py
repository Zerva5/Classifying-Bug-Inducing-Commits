import pandas as pd
import numpy as np
import sys
import os
import git
import random
from typing import Optional
import cProfile
from tqdm import tqdm

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

    print("raw search size", df.shape[0])

    if(maxFiles is not None):
        df = df.loc[df['files_changed'] <= maxFiles]
        
    if(maxDiffLines is not None):
        df = df.loc[df['diff_line_count'] <= maxDiffLines]

    print("commit lookup size:", df.shape[0])

    df.reset_index(drop=True, inplace=True)
    df['pickle_index'] = df.index
    #df = df.set_index('sha')

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
    row["fix_hash"] = pairs.iloc[iInd]['sha']
    row['fix_repo'] = pairs.iloc[iInd]['repo']
    row["bug_hash"] = pairs.iloc[rInd]['sha']
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
    - pairs (pd.DataFrame): A Pandas DataFrame with columns fix_hash, bug_hash, fix_repo, bug_repo, bug_index, fix_index, and repo.
    - searchPairs (pd.DataFrame): A Pandas DataFrame with columns sha, repo, and pickle_index
    - n (int): An integer representing the number of negative examples to select for each row in pairs.

    Returns:
    - pd.DataFrame: A Pandas DataFrame with columns fix_hash, bug_hash, fix_repo, bug_repo, fix_index, bug_index, and Y label.
    """

    def get_negative_examples(row, searchPairs, n):
        tempn = n
        newRows = []

        i_fixHash = row.fix_hash
        i_repo = row.fix_repo
        i_bugHash = row.bug_hash
        i_fixIndex = row.fix_index

        selectPairs = searchPairs.loc[(searchPairs["repo"] == i_repo) & (searchPairs['sha'] != i_fixHash) & (searchPairs['sha'] != i_bugHash)].sample(n=n, replace=True).rename(columns={'repo': 'bug_repo', 'sha': 'bug_hash', 'pickle_index': 'bug_index'}).drop(columns=['files_changed', 'diff_line_count'])
        selectPairs['fix_hash'] = i_fixHash
        selectPairs['fix_repo'] = i_repo
        selectPairs['fix_index'] = i_fixIndex

        return selectPairs

    dfList = [get_negative_examples(row, searchPairs, n) for row in tqdm(pairs.itertuples(), total=len(pairs))]
    df = pd.concat(dfList)

    return df


def negativeDifferentFixSameRepo(pairs, n):
    """
    Returns a new DataFrame with randomly selected "negative" examples that are in the same repository as the "positive" example and
    use a bug that is paired with a different fix in the same repository.

    Args:
    - pairs (pd.DataFrame): A Pandas DataFrame with columns fix_hash, bug_hash, fix_repo, bug_repo, bug_index, fix_index, and repo.
    - n (int): An integer representing the number of negative examples to select for each row in pairs.

    Returns:
    - pd.DataFrame: A Pandas DataFrame with columns fix_hash, bug_hash, fix_repo, bug_repo, fix_index, bug_index, and Y label.
    """

    def get_negative_examples(row, pairs, n):
        tempn = n
        newRows = []

        i_fixHash = row.fix_hash
        i_repo = row.fix_repo
        i_bugHash = row.bug_hash
        i_fixIndex = row.fix_index

        other_pairs_same_repo = pairs[pairs['fix_repo'] == i_repo]
        other_pairs_same_repo = other_pairs_same_repo[other_pairs_same_repo['fix_hash'] != i_fixHash]
        other_bugs = other_pairs_same_repo['bug_hash'].unique()

        if len(other_bugs) > 0:
            selectPairs = other_pairs_same_repo.loc[other_pairs_same_repo['bug_hash'].isin(other_bugs)].sample(n=n, replace=True).rename(columns={'bug_repo': 'bug_repo', 'bug_hash': 'bug_hash', 'bug_index': 'bug_index'})
            selectPairs['fix_hash'] = i_fixHash
            selectPairs['fix_repo'] = i_repo
            selectPairs['fix_index'] = i_fixIndex
            return selectPairs
        else:
            return pd.DataFrame()

    dfList = [get_negative_examples(row, pairs, n) for row in tqdm(pairs.itertuples(), total=len(pairs))]
    df = pd.concat(dfList)

    return df
    

_NoCloseCommitsFound = 0
_TotalCloseCommits = 0
def negativeCloseToFix(pairs, max_hops, n, file_tolerance = 1):

    """Returns a new DataFrame with negative examples where the bug is a commit before or after the true bug in the same repository."""

    newRowDFs = []

    # for each pair, load the commit for the bug causing commit
    # A random commit at most max_hops commits away from the bug causing commit

    repoStrings = pairs['fix_repo'].unique()
    repoDict = {}
    for r in repoStrings:
        try:
            repoDict[r] = git.Repo(os.path.join("clones", r.replace('/', '-')))
            
        except:
            raise Exception("should not have any repos that are not cloned")

    def get_negative_examples(row, pairs, max_hops, n, min_hops=0):
        newRows = []
        tempn = n

        i_fixHash = row.fix_hash
        i_repo = row.fix_repo
        i_bugHash = row.bug_hash
        i_fixIndex = row.fix_index

        ## load repo into repo object
        repo = repoDict[i_repo]

        # get the commit for the bug causing commit
        bug_commit = repo.commit(i_bugHash)

        ## get files that the bug commit changed
        bug_commit_files = bug_commit.stats.files.keys()
        #print("LEN", len(bug_commit_files))

        ## Set the number of files that the bug commit and the close commit must have in common
        commit_file_tolerance = min(file_tolerance, len(bug_commit_files))

        closeCommits = []

        # run bash command
        # git log --pretty=format:%H --max-count=1 --skip=0 --reverse
        ## Commits after the bug commit
        closeCommits.extend(repo.git.log('--pretty=format:%H', '--max-count={}'.format(max_hops), '--skip={}'.format(min_hops), '--reverse', i_bugHash + '..').strip().split('\n'))
        ## Commits before the bug commit
        closeCommits.extend(repo.git.log('--pretty=format:%H', '--max-count={}'.format(max_hops), '--skip={}'.format(min_hops + 1), i_bugHash).strip().split('\n'))

        # get the commits that changed at least 2 of the same files as the bug commit
        closeCommits = [c for c in closeCommits if len(set(repo.commit(c).stats.files.keys()).intersection(bug_commit_files)) >= commit_file_tolerance]

        # get the commits that are not the bug commit or the fix commit
        closeCommits = [c for c in closeCommits if c != i_bugHash and c != i_fixHash]

        # THis doesn't really matter tbh
        # Make sure chosen commits are not already in the pairs dataframe
        #closeCommits = [c for c in closeCommits if c not in pairs['bug_hash'].unique()]

        # select n random commits
        closeCommits = random.sample(closeCommits, min(n, len(closeCommits)))

        global _TotalCloseCommits
        _TotalCloseCommits += len(closeCommits)

        #print("Number of close commits: {}".format(len(closeCommits)))
        if len(closeCommits) == 0:
            global _NoCloseCommitsFound
            _NoCloseCommitsFound += 1



        # create a new row for each commit
        for c in closeCommits:
            newRows.append({'fix_hash': i_fixHash, 'bug_hash': c, 'fix_repo': i_repo, 'bug_repo': i_repo, 'fix_index': i_fixIndex, 'bug_index': repo.commit(c).committed_date})


        #print(newRows)
        return pd.DataFrame(newRows)

    for p in tqdm(pairs.itertuples(), total=len(pairs)):
        newRowDFs.append(get_negative_examples(p, pairs, max_hops, n))


    print("Number of new rows: {}".format(len(newRowDFs[-1])))
    print("Total close commits: {}".format(_TotalCloseCommits))
    print("Average close commits: {}".format(_TotalCloseCommits / len(pairs)))
    print("Number of no close commits found: {}".format(_NoCloseCommitsFound))
    df = pd.concat(newRowDFs)
    print("New DF")
    print(df)
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

    maxHops = 30
    
    negList = []
    #print("generating random from same repo")
    #negList.append(negativeRandomSameRepo(pairs, searchPairs, maxNegatives))
    # print("generating negative using other fixes from the same repo")
    # negList.append(negativeDifferentFixSameRepo(pairs, maxNegatives))
    negList.append(negativeCloseToFix(pairs, maxHops, maxNegatives, file_tolerance=3))
        
    df = pd.concat(negList)
    df['Y'] = 0
    return df



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
    
def fixPickleIndex(df, all_apache_commits):
    df = df.merge(all_apache_commits, how='inner', right_on=['sha', 'repo'], left_on=['bug_hash', 'bug_repo'])
    df = df.drop(columns=['repo', 'files_changed', 'diff_line_count'])
    df = df.rename(columns={"pickle_index":"bug_index"})
    
    df = df.merge(all_apache_commits, how='inner', right_on=['sha', 'repo'], left_on=['fix_hash', 'bug_repo'])
    df = df.drop(columns=['repo', 'files_changed', 'diff_line_count'])
    df = df.rename(columns={"pickle_index":"fix_index"})
    return df


def main():
    if(len(sys.argv) != 5):
        raise Exception("Wrong number of arguments, USAGE: python createPairs.py DATAFOLDER OUTPUTPATH numSamples negativesPerSample")

    numSamples = int(sys.argv[3])
    numNegatives = int(sys.argv[4])
    rootPath = sys.argv[1]
    outputName = sys.argv[2]

    optionsList = []
    
    ## Setup positive pairs
    print("Loading positive pairs")
    posPairs = getPositivePairs(rootPath)

    print("Loading all apache commits")
    all_apache_commits = getCommitLookup(rootPath, maxFiles=8, maxDiffLines=80)
    searchPairs = all_apache_commits
    all_apache_commits = all_apache_commits.set_index('sha')
    

    print("filtering positive pairs")
    posPairs = posPairs[(posPairs['bug_hash'].isin(all_apache_commits.index)) & (posPairs['fix_hash'].isin(all_apache_commits.index))]
    posPairs = fixPickleIndex(posPairs, all_apache_commits)
    
    print("positive examples:", posPairs.shape[0])

    #posPairs.to_csv(os.path.join(rootPath, "pairs_output", "apache_positive_pairs2.csv"))


    print("generating negative pairs")

    negPairs = createNegativeExamples(posPairs, searchPairs, numNegatives)
    negPairs = fixPickleIndex(negPairs, all_apache_commits)

    negPairs.to_csv(os.path.join(rootPath, "pairs_output", "apache_close_broad_negative_pairs.csv"), index=False)


    #print("negative examples:", withNegative.shape[0] - posPairs.shape[0])
    #print("total examples:", withNegative.shape[0])

    #if not os.path.exists(os.path.join(rootPath, "pairs_output")):
        #os.makedirs(os.path.join(rootPath, "pairs_output"))
    
    #withNegative.to_csv(os.path.join(rootPath, "pairs_output", outputName), index=False)


if __name__ == "__main__":
    #cProfile.run('main()', sort='cumtime')
    main()
