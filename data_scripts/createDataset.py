import sys
import os
from numpy import frombuffer
import pandas as pd
import github as gh
from github.GitCommit import GitCommit
from github.GitAuthor import GitAuthor
from github.Commit import Commit
from github.File import File

from collections import namedtuple
from dataclasses import dataclass
import git


basePath = os.path.abspath(os.path.join(sys.path[0], ".."))
    
DataFields = (
    # boolean comparison
    "sameAuthor",
    "sameRepo",
    "bugIsParentCommit",
    

    ## For individual commits
    "fix_Date",
    "bug_Date",
    "fix_LinesChanged",
    "bug_LinesChanged",
    "fix_authorOwnsRepo",
    "bug_authorOwnsRepo"
)

def getGithubUserData(namePrefix, uObj, row):
    row[namePrefix + "_collaborators"] = uObj.collaborators
    row[namePrefix + "_contributions"] = uObj.contributions
    row[namePrefix + "_disk_usage"] = uObj.disk_usage
    row[namePrefix + "_followers"] = uObj.followers
    row[namePrefix + "_following"] = uObj.following
    row[namePrefix + "_team_count"] = uObj.team_count
    row[namePrefix + "_hireable"] = uObj.hireable
    row[namePrefix + "_site_admin"] = uObj.site_admin

def getGitUserData(namePrefix, uObj:GitAuthor, row):
    row[namePrefix + "_email"] = uObj.email
    row[namePrefix + "_name"] = uObj.name

def getGitCommitData(namePrefix, cObj:GitCommit, row):
    row[namePrefix + "_message"] = cObj.message

    getGitUserData(namePrefix + "_author", cObj.author, row)

def getFileData(namePrefix, fObj, row):
    row[namePrefix + "_apiURL"] = fObj.contents_url
    row[namePrefix + "_filename"] = fObj.filename
    row[namePrefix + "_prefiousFilename"] = fObj.previous_filename
    row[namePrefix + "_sha"] = fObj.sha
    row[namePrefix + "_patch"] = fObj.patch
    row[namePrefix + "_additions"] = fObj.additions
    row[namePrefix + "_deletions"] = fObj.deletions
    row[namePrefix + "_changes"] = fObj.changes
    row[namePrefix + "_status"] = fObj.status
    


def getCommitData(namePrefix, cObj:Commit, g, row):
    _stats = cObj.stats
    _files = cObj.files

    # if(len(_files) > 1):
    #     print("MORE THAN 1 FILE", namePrefix, len(_files))

    getFileData(namePrefix + "_file", _files[0], row)

    getGitCommitData(namePrefix + "_git", cObj.commit, row)


    # General Commit stats
    row[namePrefix + "_totalAdditions"]  = _stats.additions
    row[namePrefix + "_totalDeletions"] = _stats.deletions

def create_dataset(infilePath, g):
    pairDF = pd.read_csv(infilePath)
    
    noAuthor = 0
    noCommitter = 0
    rows = []
    ## loop through all pairs
    #for i in range(pairDF.shape[0]):
    for i in range(10):
        pairRow = pairDF.iloc[-i]
        dataRow = {}
        
        repo = g.get_repo(pairRow['repo'])
        
        bug_commit = repo.get_commit(sha=pairRow['bug_hash'])
        fix_commit = repo.get_commit(sha=pairRow['fix_hash'])

        getCommitData("bug", bug_commit, g, dataRow)
        getCommitData("fix", fix_commit, g, dataRow)

        rows.append(dataRow)


        #print(dataRow)
    dataSetDF = pd.DataFrame(rows)

    return dataSetDF





## For each row, get the info of the git repo in data/repos

def getCommitInfo(repoObj, sha:str, cDict:dict, prefix:str):

    print(repoObj.commit(sha).authored_datetime)
    
    

def main():
    if(len(sys.argv) != 3):
        raise Exception("Not enough arguments, USAGE: python createDataset.py DATAFILE OUTPUTPATH")

    dir_path = os.path.dirname(os.path.realpath(__file__))

    tokenFile = open(dir_path + "/gh_access_token.txt")
    access_token = tokenFile.readline().strip()
    tokenFile.close()
    
    inFile = sys.argv[1]
    outFile = sys.argv[2]

    g = gh.Github(access_token)

    print("Github API request remaining:", g.get_rate_limit().core.remaining)

    df = create_dataset(inFile, g)

    df.to_csv(outFile)
    
     

if __name__ == "__main__":
    main()
