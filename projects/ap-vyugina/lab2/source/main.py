import pickle

from gensim.models import Word2Vec
from .utils import read_and_group, weighted_average_vector
from tqdm import tqdm


if __name__ == "__main__":
    with open('assets/term_weights.pkl', 'rb') as file:
        term_weights = pickle.load(file)
    model = Word2Vec.load('assets/word2vec.model')

    for mode in ["test"]:
        print(f"Working on {mode} part of corpus...")
        corpus_files = [
            f"assets/annotated-corpus/{mode}/1.tsv",
            f"assets/annotated-corpus/{mode}/2.tsv",
            f"assets/annotated-corpus/{mode}/3.tsv",
            f"assets/annotated-corpus/{mode}/4.tsv"
        ]

        output_file = open(f"assets/tokenized_sentences_{mode}.tsv", 'w')

        for i, cf in enumerate(corpus_files):
            sentences = read_and_group(cf)
            for j, sentence in tqdm(enumerate(sentences), total=len(sentences)):
                idx = f"{i+1}.{j}"
                tokenized_sentence = weighted_average_vector(sentence, model, term_weights)
                if tokenized_sentence is None:
                    print(idx, sentence)
                if tokenized_sentence is not None:
                    tokenized_sentence = tokenized_sentence.reshape(-1, ).tolist()[0]
                    tokenized_sentence_str = "\t".join(map(lambda x: f"{x:.12f}", tokenized_sentence))
                    output_file.write(f"{idx}\t{tokenized_sentence_str}\n")

        output_file.close()
