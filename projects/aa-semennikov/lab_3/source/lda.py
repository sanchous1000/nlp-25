import gensim
import pickle
import os
import pandas as pd


with open('assets/corpus.pkl', 'rb') as f:
    corpus = pickle.load(f)
with open('assets/dictionary.pkl', 'rb') as f:
    dictionary = pickle.load(f)

TOPICS_RANGE = [2, 4, 10, 20, 40]
TOP_N_DOCS = 3 
perplexity_path = 'assets/perplexity_results.pkl'

for num_topics in TOPICS_RANGE:
    lda_model = gensim.models.LdaModel(corpus=corpus,
                                       id2word=dictionary,
                                       num_topics=num_topics,
                                       random_state=42,
                                       passes=10)

    # Получение топовых слов для каждой темы
    topics = lda_model.show_topics(formatted=False, num_words=10)
    top_words_rows = []
    for topic_idx, words_probs in topics:
        print(f"Тема {topic_idx}:")
        words = ", ".join([w for w, prob in words_probs])
        print(words)
        print("-" * 50)
        top_words_rows.append({"topic": topic_idx, "top_words": words})

    log_perp = lda_model.log_perplexity(corpus)
    perp = 2 ** (-log_perp)
    print(f"Логарифмическая перплексия модели ({num_topics} тем): {log_perp:.4f}")
    print(f"Перплексия модели ({num_topics} тем): {perp:.4f}")

    if os.path.exists(perplexity_path) and os.path.getsize(perplexity_path) > 0:
        try:
            with open(perplexity_path, 'rb') as f:
                data = pickle.load(f)
        except (EOFError, pickle.UnpicklingError):
            data = {}
    else:
        data = {}
    data[num_topics] = perp.item()
    with open(perplexity_path, 'wb') as f:
        pickle.dump(data, f)

    # Распределения тема-документ (вероятности принадлежности документа ко всем темам)
    doc_topic_dists = []
    for bow in corpus:
        dist = [0.0] * num_topics
        for topic_id, prob in lda_model.get_document_topics(bow, minimum_probability=0):
            dist[topic_id] = prob
        doc_topic_dists.append(dist)

    doc_topic_path = f'results/doc_topic_probs_{num_topics}.tsv'
    pd.DataFrame(doc_topic_dists).to_csv(doc_topic_path, sep='\t', index=False, header=False)

    top_words_path = f'results/top_words_{num_topics}.tsv'
    pd.DataFrame(top_words_rows).to_csv(top_words_path, sep='\t', index=False, header=False)

    # top-N документов
    top_docs_rows = []
    for topic_id in range(num_topics):
        scored_docs = [(doc_idx, dist[topic_id]) for doc_idx, dist in enumerate(doc_topic_dists)]
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        for doc_idx, prob in scored_docs[:TOP_N_DOCS]:
            top_docs_rows.append({"topic": topic_id, "doc_index": doc_idx, "prob": prob})

    top_docs_path = f'results/top_docs_{num_topics}.tsv'
    pd.DataFrame(top_docs_rows).to_csv(top_docs_path, sep='\t', index=False, header=False)