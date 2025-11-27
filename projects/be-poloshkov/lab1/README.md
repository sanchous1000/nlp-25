# Лабораторная работа №1

## Описание задания

Обработать выбранный текстовый датасет. 

Сначала токенизировать текст по регулряным выражениям, затем при помощи библиотек выполнить стемминг и лемматизацию.

Результаты сохранить в формате TSV с указанием исходного токена, стеммы и леммы.

Проанализировать полученные результаты.

## Использованные технологии и инструменты
- Датасет: Базовый новостной датасет, предоставленный преподавателем в указаниях к работе
- Модели: SnowballStemmer, WordNetLemmatizer
- Библиотеки: pandas, nltk

## Результаты работы

Для токенизации простых слов использовалось простое регулярное выражение, учитывающее специальные знаки, такие как апострофы

```python
WORD_REGEX = "\w+(?:'\w+)?"
```

Условием задания так же предусмотрено рассмотрение более сложных вариантов слов, они обрабатываются отдельными регулярными выражениями

```python
COMPLEX_TOKEN_PATTERNS = [
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # email
    r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3,4}\)?[-.\s]?\d{3}[-.\s]?\d{4}',  # phone
    r'\b(?:Dr|Mr|Mrs|Ms|Prof)\.\s+[A-Z][a-zA-Z]*',  # titles
    r'((https?):((//)|(\\\\))+[\w\d:#@%/;$()~_?\+-=\\\.&]*)', # url
]
```

По итогу получили трехэлементные аннотации для каждого предложения, кортеж (токен, стемма, лемма). Пример:

```
AP	ap	ap
Dozens	dozen	dozen
of	of	of
Rwandan	rwandan	rwandan
soldiers	soldier	soldier
flew	flew	fly
into	into	into
```

В большинстве случаев алгоритм верно определял лемму, учитывал неправильные глаголы:
```
flew	flew	fly
held	held	hold
lost	lost	lose
```

Примеры специальных токенов:

```
Dr. Phil	Dr. Phil	Dr. Phil
...
(212) 621-1630	(212) 621-1630	(212) 621-1630
...
http://www.investor.reuters.com/FullQuote.aspx?ticker=XOM.N	http://www.investor.reuters.com/FullQuote.aspx?ticker=XOM.N	http://www.investor.reuters.com/FullQuote.aspx?ticker=XOM.N
```

Были найдены и примеры омонимии:

`Captain Sourav Ganguly today said India would have forced a result in the first cricket test against south africa on a lifeless track had they won the ross and bat first`

`One of the world largest **bat** species is sad to be thriving in caprivity in a purpose built tunnel`

В данном контексте одна из гипотез: bat - bat - bat: глагол, ударить бейсбольной битой.

Другая: bat - bat - bat: существительное - летучая мышь

Другой, более показательный пример, глагол found:

`Found this really cool search engine Koders via ResearchBuzz` - в данном случае мы получили триаду (found, found, found), хотя должны были получать (found, find, find).

Произошло это потому, что found так же является самостоятельным глаголом (to found - основать, founder - основатель).


Обработка заняла:
- 94 секунды для train
- 6 секунд для test


## Выводы
Классические алгоритмы хорошо справляются с задачей до тех пор, пока не сталкиваются с лексической неопределенностью, например, омонимами. 
В данном случае без учета контекста невозможно сделать правильный анализ текста.

## Инструкция по запуску
1. Установить зависимости: `pip install -r requirements.txt`
2. В корневой директории проекта в папку dataset положить два файла: `test.csv` / `train.csv`
3. В главном файле `main.py` задать переменную `dataset_name` в `test` или `train` соответственно
4. Запустить `main.py` любым удобным интерпретатором
...