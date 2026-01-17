import gensim
from gensim.matutils import Sparse2Corpus
from scipy.sparse import csr_matrix
from gensim.corpora import Dictionary

import pickle


with open('assets/tf-idf.pkl', 'rb') as file:
    tfidf_matrix, feature_names = pickle.load(file)

sparse_corpus = Sparse2Corpus(csr_matrix(tfidf_matrix), documents_columns=False)
dictionary = Dictionary.from_corpus(sparse_corpus, id2word=dict(zip(range(len(feature_names)), feature_names)))

NUM_TOPICS = 4

# Создание и обучение модели LDA
lda_model = gensim.models.LdaModel(corpus=sparse_corpus,
                                   id2word=dictionary,
                                   num_topics=NUM_TOPICS,
                                   random_state=42,
                                   passes=1)

# Получение топовых слов для каждой темы
topics = lda_model.show_topics(formatted=False, num_words=10)
for topic_idx, words_probs in topics:
    print(f"Тема {topic_idx}:")
    words = ", ".join([w for w, prob in words_probs])
    print(words)
    print("-" * 50)

log_perplexity = lda_model.log_perplexity(sparse_corpus)
perplexity = 2 ** (-log_perplexity)
print(f"Логарифмическая перплексия модели: {log_perplexity:.4f}")
print(f"Перплексия модели: {perplexity:.4f}")

with open('assets/perplexity_results.pkl', 'rb') as file:
    data = pickle.load(file)
data[NUM_TOPICS] = perplexity.item()
print(data)
with open('assets/perplexity_results.pkl', 'wb') as file:
    pickle.dump(data, file)