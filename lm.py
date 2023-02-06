import argparse
import numpy as np
from model import AudioDataset

parser = argparse.ArgumentParser(description='train lm')
parser.add_argument('--stm', help="stm: uid,wavpath,from,to,length,text", type=str, default=None)

if __name__ == '__main__':
    args = parser.parse_args()
    
    # Get data set
    data = AudioDataset(args.stm)
    
    # Load lexicon
    lines = list(open("lexicon.txt"))
    lexicon = list()

    for l in lines:
        l = l.split("\t")
        lexicon.append(l[0])
    
    # Train n-gram
    n = 1
    n_gram = np.zeros((len(lexicon), len(lexicon) ** n))

    count = 1
    
    # Go for each sample through transcript 
    for sample in data:
        print(f"Train LM: {count}/{len(data)}")
        count += 1
        
        corpus = sample[2].split()
        
        # Increase score for each occurring word pair at corresponding index in n-gram matrix
        for index in range(1, len(corpus)):
            if corpus[index-1] in lexicon and corpus[index] in lexicon:
                prev = lexicon.index(corpus[index-1])
                curr = lexicon.index(corpus[index])

                n_gram[prev, curr] += 1
        
        # Sample limit for training
        #if count == 100000:
        #    break
    
    # Calculate probabilities 
    sums = np.sum(n_gram, axis=0)
    n_gram = np.divide(n_gram.T, sums, out=np.zeros_like(n_gram), where=sums!=0).T
    
    # Save LM in file
    np.savetxt("n-grams/1-gram.txt", n_gram)

