import argparse
from model import AudioDataset

parser = argparse.ArgumentParser(description='create lexicon')
parser.add_argument('--stm', help="stm: uid,wavpath,from,to,length,text", type=str, default=None)

if __name__ == '__main__':
    args = parser.parse_args()

    # Load data set
    data = AudioDataset(args.stm)

    all_words = list()
    lexicon = list()
    count = 1

    # Go for each sample through transcript and add new words to list
    for sample in data:
        print(count)
        count += 1

        words = sample[2].split()
        for word in words:
            if word not in all_words:
                all_words.append(word)
    
    # Sort list with all distinct words of data set transcripts lexicographically
    all_words = sorted(all_words)

    # Bring each word in needed format and add to lexicon
    for word in all_words:
        lexicon.append(word + "\t" + " ".join(word) + " |")

    # Save lexicon in file
    with open(r"lexicon.txt", "w") as fp:
        for item in lexicon:
            fp.write("%s\n" % item)
