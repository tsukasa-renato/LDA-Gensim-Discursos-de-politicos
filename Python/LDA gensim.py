# Importando as bibliotecas necessárias para este trabalho
import gensim
import pandas as pd
import spacy
import nltk
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from pprint import pprint

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

nlp = spacy.load('pt_core_news_sm')  # Carregando os modelos para lematização na língua portuguesa

nltk.download('stopwords')  # Depois que baixo, não é mais necessário executar essa linha
stop_words = stopwords.words('portuguese')  # Carregando o dicionário de stopword do NLTK da língua inglesa
# Acrescentando mais palavras no dicionário de stopword
stop_words.extend(['ir', 'aqui', 'ter', 'todo', 'fazer', 'dizer', 'falar', 'estar', 'hoje', 'algum', 'outro', 'ser',
                   'querer', 'qualquer', 'nado', 'porque', 'vir', 'partir', 'governar', 'deputar', 'parlamentar', 'sr',
                   'presidente', 'vice', 'discursar', 'parecer', 'vez', 'dar', 'ex', 'sim', 'levar', 'quase', 'chance',
                   'ano', 'além', 'sob', 'termo', 'sempre', 'nenhum', 'coisa', 'frase', 'diverso'])


# Functions
def tokenization(texts):
    for word in texts:
        yield (gensim.utils.simple_preprocess(str(word), deacc=False))


def remove_stopwords(matrix):
    return [[word for word in simple_preprocess(str(line)) if word not in stop_words] for line in matrix]


def lemmatization(matrix):
    matrix_out = []
    for line in matrix:
        doc = nlp(" ".join(line))
        matrix_out.append([word.lemma_ for word in doc])
    return matrix_out


def n_grams(matrix):
    n_grams_model = gensim.models.Phrases(matrix, min_count=2, threshold=10)
    matrix_out = gensim.models.phrases.Phraser(n_grams_model)
    return [matrix_out[line] for line in matrix]


def create_dictionary(matrix):
    return Dictionary(matrix)


def create_corpus(id2word, matrix):
    return [id2word.doc2bow(line) for line in matrix]


def show_keyword_freq(dic, corp, i):
    return [[(dic[n], freq) for n, freq in cp] for cp in corp[:i]]


df = pd.read_json('Arquivos json/KimKataguiri 2019.json', encoding="utf8")
database = df.content.values.tolist()  # Converte o texto em uma lista

data_processing = list(tokenization(database))  # Converte em matriz, remove números, pontuação e letras isoladas

data_processing = remove_stopwords(data_processing)  # Remove palavras irrelevantes (stopwords)

data_processing = lemmatization(data_processing)  # Transforma a palavra em sua "palavra raiz" (Lematização)

data_processing = remove_stopwords(data_processing)

data_processing = n_grams(data_processing)  # Juntar palavras que aparece frequentemente em sequência

data_processing = n_grams(data_processing)

data_processing = remove_stopwords(data_processing)

dictionary = create_dictionary(data_processing)  # Atribui um id para cada palavra

print(len(dictionary))
dictionary.filter_extremes(no_below=2)  # mantem as palavras que estão presente em pelo menos 2 documentos
print(len(dictionary))

corpus = create_corpus(dictionary, data_processing)  # Pega a frequência de cada palavra

# Criação do modelo LDA
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=5, passes=10,
                                            random_state=100, chunksize=5)

pprint(lda_model.show_topics(num_words=10, formatted=False))

# Salvando o modelo criado
lda_model.save('LDA model/my_lda.model')
