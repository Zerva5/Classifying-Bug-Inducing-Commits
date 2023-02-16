import sys
import os
import pandas as pd
import github as gh
from collections import namedtuple
from dataclasses import dataclass

dir_path = os.path.dirname(os.path.realpath(__file__))

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

def getUserData(namePrefix, uObj, row):
    row[namePrefix + "_collaborators"] = uObj.collaborators
    row[namePrefix + "_contributions"] = uObj.contributions
    row[namePrefix + "_disk_usage"] = uObj.disk_usage
    row[namePrefix + "_followers"] = uObj.followers
    row[namePrefix + "_following"] = uObj.following
    row[namePrefix + "_team_count"] = uObj.team_count
    row[namePrefix + "_hireable"] = uObj.hireable
    row[namePrefix + "_site_admin"] = uObj.site_admin

def getGitUserData(namePrefix, uObj, row):
    row[namePrefix + "_"] = uObj.collaborators
    row[namePrefix + "_contributions"] = uObj.contributions


def getCommitData(namePrefix, cObj, g, row):
    _author = cObj.author
    _committer = cObj.committer
    _stats = cObj.stats
    _files = cObj.files

    print(_committer)

    # commit author stats
    #https://pygithub.readthedocs.io/en/latest/github_objects/NamedUser.html#github.NamedUser.NamedUser.collaborators
    if(_author != None):
        authorPrefix = namePrefix + "_author"
        getUserData(authorPrefix, _author, row)

    if(_committer != None):
        commiterPrefix = namePrefix + "_committer"
        getUserData(commiterPrefix, _committer, row)

    print(cObj.html_url)


    # General Commit stats
    row[namePrefix + "_totalAdditions"]  = _stats.additions
    row[namePrefix + "_totalDeletions"] = _stats.deletions
    row[namePrefix + "_totalChanges"] = _stats.total

    # Commit file stats


def test(row):
    row['repo'] = 'NO'


def create_dataset(infilePath, g):
    pairDF = pd.read_csv(infilePath)
    
    noAuthor = 0
    noCommitter = 0
    ## loop through all pairs
    #for i in range(pairDF.shape[0]):
    for i in range(500):
        pairRow = pairDF.iloc[i]
        dataRow = {}
        try:
        
            repo = g.get_repo(pairRow['repo'])
        
            bug_commit = repo.get_commit(sha=pairRow['bug_hash'])
            fix_commit = repo.get_commit(sha=pairRow['fix_hash'])

        except:
            continue

        ## Seeing what portion of commits have github authors and/or commiters
        if(bug_commit.author == None):
            noAuthor += 1
        if bug_commit.committer == None:
            noCommitter += 1

        if fix_commit.author == None:
            noAuthor += 1

        if fix_commit.committer == None:
            noCommitter += 1    
            

        #getCommitData("bug", bug_commit, g, dataRow)
        #getCommitData("fix", fix_commit, g, dataRow)

        #print(dataRow)

    print("AUTHOR STATS:")
    print(noAuthor)
    print(noCommitter)
        #print(bug_commit)
        #print(fix_commit)

    

def main():
    # if(len(sys.argv) != 3):
    #     raise Exception("Not enough arguments, USAGE: python createDataset.py DATAFILE OUTPUTPATH")

    tokenFile = open(dir_path + "/gh_access_token.txt")
    access_token = tokenFile.readline().strip()
    tokenFile.close()
    
    inFile = sys.argv[1]

    g = gh.Github(access_token)

    print("Github API request remaining:", g.get_rate_limit().core.remaining)

    create_dataset(inFile, g)
    
     

if __name__ == "__main__":
    main()
