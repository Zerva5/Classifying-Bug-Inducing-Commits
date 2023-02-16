import sys
import numpy as np
import pandas as pd

def getRepoURLs(inFile, outFile):
    pairDF = pd.read_csv(inFile)

    urlArr = "git@github.com:" + pairDF['repo'].unique() + ".git"

    
    
    np.savetxt(outFile, np.flip(urlArr), fmt="%s")

def main():
    if(len(sys.argv) != 3):
        raise Exception("Not enough arguments, USAGE: python createRepoURLList.py DATAFILE OUTPUTPATH")

    inFile = sys.argv[1]
    outFile = sys.argv[2]

    getRepoURLs(inFile, outFile)

   

if __name__ == "__main__":
    main()
