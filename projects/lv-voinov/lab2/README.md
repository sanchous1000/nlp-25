## Отчет по 2 лабораторной

Была обучена модель Word2Vec на тренировочном датасете.

Ранжированный список косинусного расстояния для векторов демонстрирует, что похожие токены ближе всего к изначальному, а далекие - дальше всего:

```
Базовый токен: 'football'
Ранжированный список по косинусному расстоянию:
  1. 'hockey' [похожий] -> 0.3408
  2. 'soccer' [похожий] -> 0.3504
  3. 'baseball' [похожий] -> 0.4138
  4. 'team' [та же область] -> 0.5268
  5. 'player' [та же область] -> 0.5351
  6. 'game' [та же область] -> 0.5628
  7. 'universe' [разные] -> 0.8251
  8. 'chemistry' [разные] -> 0.8639
  9. 'painting' [разные] -> 0.8842

Базовый токен: 'airplane'
Ранжированный список по косинусному расстоянию:
  1. 'aircraft' [похожий] -> 0.3949
  2. 'jet' [похожий] -> 0.4619
  3. 'plane' [похожий] -> 0.4785
  4. 'flight' [та же область] -> 0.6951
  5. 'pilot' [та же область] -> 0.7306
  6. 'airport' [та же область] -> 0.7383
  7. 'fish' [разные] -> 0.7554
  8. 'cat' [разные] -> 0.8957
  9. 'mathematics' [разные] -> 0.9603

Базовый токен: 'physics'
Ранжированный список по косинусному расстоянию:
  1. 'science' [похожий] -> 0.3290
  2. 'chemistry' [похожий] -> 0.4235
  3. 'biology' [похожий] -> 0.4628
  4. 'research' [та же область] -> 0.6337
  5. 'experiment' [та же область] -> 0.7070
  6. 'theory' [та же область] -> 0.7557
  7. 'dance' [разные] -> 0.7763
  8. 'music' [разные] -> 0.9774
  9. 'song' [разные] -> 1.0004

```

Был реализован алгоритм векторизации произвольного текста по указанному в задании алгоритму. Код:

```
def vectorize_document(text, vector_size=100):
    sentences = sent_tokenize(text)
    
    sentence_vectors = []
    
    for sentence in sentences:
        tokens = preprocess(sentence)
        
        if not tokens:
            continue
            
        token_vectors = []
        
        for token in tokens:
            if token in model.wv:
                token_vectors.append(model.wv[token])
        
        if token_vectors:
            sentence_vector = np.mean(token_vectors, axis=0)
            sentence_vectors.append(sentence_vector)
    
    if sentence_vectors:
        document_vector = np.mean(sentence_vectors, axis=0)
        return document_vector
    else:
        return np.zeros(vector_size)
```

Была выполнена векторизация тренировочной и тестовой выборки. Результаты были сохранены в файлы train_data.tsv и test_data.tsv соответственно.
