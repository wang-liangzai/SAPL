import spacy
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from collections import Counter
import pickle
from sklearn.preprocessing import QuantileTransformer
processed_data_file = '/home/mmn/wgj/DynRT-main/DynRT/input/prepared_clean/processed_word_freq.pkl'
data_file = '/home/mmn/wgj/DynRT-main/DynRT/input/prepared_clean/train_text'
class SoftPromptInitializer:
    def __init__(self, bert_model_path, train_text_file, softprompt_length=10, freq_threshold=100):
        self.train_text_file = train_text_file
        self.bert_model_path = bert_model_path
        self.softprompt_length = softprompt_length
        self.freq_threshold = freq_threshold
        self.nlp = spacy.load('en_core_web_sm')
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_path)
        self.bert_model = BertModel.from_pretrained(bert_model_path)
        self.excluded_tokens = {'.', ',', '<', '>', '<user>', '#', '_', 'to', "'", '"', "'s", '-', ':',
                                't', 'm', 'is', 'are', 'when', ')', '(', '<', '>', '/', 'be', "'re", '`',
                                'rt', 've', "'"}

    def load_data(self, train_text_file):
        with open(train_text_file, 'rb') as f:
            train_text = pickle.load(f)
        # return [text.strip() for text in train_text]
        return train_text

    def compute_word_distribution(self, train_texts):
        word_freq = Counter()
        for text in train_texts:
            tokens = text.split()
            filtered_tokens = [token for token in tokens if token not in self.excluded_tokens]
            doc = self.nlp(' '.join(filtered_tokens))
            filtered_tokens = [token.text for token in doc if token.pos_ not in {'ADP', 'CCONJ', 'SCONJ', 'CONJ', 'PRON', 'DET'}]
            word_freq.update(filtered_tokens)
        return word_freq

    def filter_and_sort_words(self, word_freq):
        filtered_word_freq = {word: freq for word, freq in word_freq.items() if freq >= self.freq_threshold}
        sorted_word_freq = sorted(filtered_word_freq.items(), key=lambda x: x[1], reverse=True)
        return sorted_word_freq

    def generate_softprompt(self):
        train_texts = self.load_data(data_file)
        word_freq = self.compute_word_distribution(train_texts)
        sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        top_words = [word for word, _ in sorted_word_freq[:1000]]

        softprompt = torch.zeros(10, self.bert_model.config.hidden_size)

        # Split top_words into 10 groups, each with 100 words
        grouped_top_words = [top_words[i:i + 100] for i in range(0, 1000, 100)]

        for group_index, word_group in enumerate(grouped_top_words):
            group_embeddings_sum = torch.zeros(self.bert_model.config.hidden_size)

            for word in word_group:
                word_tokens = self.tokenizer.tokenize(word)
                word_ids = self.tokenizer.convert_tokens_to_ids(word_tokens)
                word_embeddings = self.bert_model.embeddings.word_embeddings(torch.tensor(word_ids))
                word_embedding_mean = torch.mean(word_embeddings, dim=0)
                group_embeddings_sum += word_embedding_mean

            group_embedding_avg = group_embeddings_sum / len(word_group)
            softprompt[group_index] = group_embedding_avg

        return softprompt



