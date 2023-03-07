import datetime
import time
from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.preprocessing import OneHotEncoder
from sentence_transformers import SentenceTransformer
import sys

# # Encodes the 
# def processMetaDatas(authorName: str, authorEmail: str, message: str, dateTime: datetime):
#     procName = 1
#     timestamp = time.mktime(datetime.timetuple())
#     returnDict = dict()
#     return returnDict



def processCommitMessages(commit1: str, commit2: str):
    similarity = cosineSimilarity(commit1, commit2)
    print(similarity)


def cosineSimilarity(str1: str, str2: str):
    vectors = sentences2Vecs([str1, str2])
    similarity = cosine_similarity(vectors[0].reshape(1, -1), vectors[1].reshape(1,-1))[0][0]
    return similarity

def sentences2Vecs(sentences):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    sentence_embeddings = model.encode(sentences)
    return sentence_embeddings





def main():
    if(len(sys.argv) < 4):
        raise Exception("Not enough arguments, USAGE: `python processMetadata.py -c CommitMessage1 CommitMessage2` or `python processMetadata.py -m author email time(ISO)`")
    
    if(sys.argv[1] == '-c'):
        processCommitMessages(sys.argv[2], sys.argv[3])
        return

    if(sys.argv[1] == '-m'):
        if(len(sys.argv) < 5):
            raise Exception("Not enough arguments, USAGE: `python processMetadata.py -m author email time(ISO)`")
        
    

if __name__ == "__main__":
    main()
