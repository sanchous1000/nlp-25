import numpy as np

unknown_vector = np.random.rand(30)

def read_and_group(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        sentence = []  # Временный контейнер для хранения текущего блока слов
        corpus = []         # Итоговый список групп слов
        
        for line in file:
            stripped_line = line.strip() 
            if stripped_line != '': 
                words = stripped_line.split("\t")
                sentence.append(words[0])
            else:
                corpus.append(sentence)
                sentence = []
    return corpus


def weighted_average_vector(tokens, model, term_weights):
    vectors = []
    total_weight = 0
    
    for token in tokens:
        try:
            token_embedding = model.wv[token]
        except KeyError:
            token_embedding = unknown_vector
        vectors.append(token_embedding.reshape(-1, 1) * term_weights.get(token.lower(), 0))
        total_weight += term_weights.get(token.lower(), 0)
    
    if total_weight != 0:
        return np.array(vectors).mean(axis=0) / total_weight
    else:
        return None