from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import sys
import time

# Process two names (or emails) and return 1 if they are the same and 0 if they are not
def processNames(name1: str, name2: str):
    if name1 == name2:
        return 1
    else:
        return 0

# Process two timestamnps and return their positive difference
def processTimestamps(timestamp1: int, timestamp2: int):
    difference = timestamp1 - timestamp2
    if(difference < 0):
        difference *= -1
    return difference

# Process two commit messages and return a similarity score from 0 to 1
def processCommitMessages(commit1: str, commit2: str):
    similarity = cosineSimilarity(commit1, commit2)
    print(similarity)


# Calculate the cosine similarity of two strings
def cosineSimilarity(str1: str, str2: str):
    vectors = sentences2Vecs([str1, str2])
    similarity = cosine_similarity(vectors[0].reshape(1, -1), vectors[1].reshape(1,-1))[0][0]
    normalized_similarity = round((similarity + 1) / 2, 6)
    return normalized_similarity

# Turn an array of sentences into an array of vectors using a special sentence transformer using their paraphrase-MiniLM-L6-v2 model for relatively quick performance.  
# In my small amount of testing, takes about 4 seconds to load the model the first time and perform the transformations, and then 0.3 secconds after that
def sentences2Vecs(sentences):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    sentence_embeddings = model.encode(sentences)
    return sentence_embeddings


def main():
    if(len(sys.argv) < 4):
        raise Exception("Not enough arguments, USAGE: `python processMetadata.py -c CommitMessage1 CommitMessage2`")
    
    if(sys.argv[1] == '-c'):
        processCommitMessages(sys.argv[2], sys.argv[3])
        return
        
    

if __name__ == "__main__":
    main()
