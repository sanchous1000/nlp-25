# Lab 1: Text Segmentation & Annotation

Сегментация, токенизация, стемминг и лемматизация датасета AG News.

## Что делает

- Разбивает тексты на предложения и токены (с поддержкой email, телефонов, эмодзи)
- Применяет стемминг (SnowballStemmer) и лемматизацию (WordNetLemmatizer)
- Сохраняет результаты в TSV формате

## Структура выходных данных

```
train/test
├── Business/
├── Sci_Tech/
├── Sports/
└── World/
    └── 000000.tsv (token\tstem\tlemma)
```

## Запуск

Открыть [lab1.ipynb](lab1.ipynb) и выполнить все ячейки.
