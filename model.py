# class for summary
import nltk 
nltk.download('stopwords')
nltk.download('punkt')
from nltk import sent_tokenize, word_tokenize, PorterStemmer
from nltk.corpus import stopwords
import math
import numpy as np


def get_text(text_input):
  """
    add more extract text from url etc
  """
  return text_input

class TextSummarize:
  def __init__(self, text_input, threshold=None, ratio=None, limit=15):
    self.limit=limit
    self.ratio = ratio
    self.text_input = get_text(text_input)
    
    self.sentences = sent_tokenize(self.text_input)
    total_sentences = len(self.sentences)
    print("Summarizing a text of length: ", total_sentences)


    self.freq_matrix = self.frequency_matrix(self.sentences)
    self.tf_matrix = self.tf_matrix(self.freq_matrix)
    self.doc_per_words = self.documents_per_words(self.freq_matrix)

   
    self.idf_matrix = self.idf_matrix(self.freq_matrix, self.doc_per_words, total_sentences)
    self.tf_idf_matrix = self.tf_idf_matrix(self.tf_matrix, self.idf_matrix)

    self.sentence_scores = self.score_sentences(self.tf_idf_matrix)
    self.threshold = self.average_score(self.sentence_scores) if threshold is None else threshold

    self.summary = self.generate_summary(self.sentences, self.sentence_scores, 1.3 * self.threshold, self.ratio)

  
  def word_frequency_table(self, text_string):
    """
      dictionary of word frequency
    """
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text_string)
    ps = PorterStemmer()

    freqTable = dict()
    for word in words:
        word = ps.stem(word)
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    return freqTable


  def frequency_matrix(self, sentences):
      frequency_matrix = {}
      stopWords = set(stopwords.words("english"))
      ps = PorterStemmer()

      for sent in sentences:
          freq_table = {}
          words = word_tokenize(sent)
          for word in words:
              word = word.lower()
              word = ps.stem(word)
              if word in stopWords:
                  continue

              if word in freq_table:
                  freq_table[word] += 1
              else:
                  freq_table[word] = 1

          frequency_matrix[sent[:self.limit]] = freq_table

      return frequency_matrix


  def tf_matrix(self, freq_matrix):
      tf_matrix = {}

      for sent, f_table in freq_matrix.items():
          tf_table = {}

          count_words_in_sentence = len(f_table)
          for word, count in f_table.items():
              tf_table[word] = count / count_words_in_sentence

          tf_matrix[sent] = tf_table

      return tf_matrix


  def documents_per_words(self, freq_matrix):
      word_per_doc_table = {}

      for sent, f_table in freq_matrix.items():
          for word, count in f_table.items():
              if word in word_per_doc_table:
                  word_per_doc_table[word] += 1
              else:
                  word_per_doc_table[word] = 1

      return word_per_doc_table


  def idf_matrix(self, freq_matrix, count_doc_per_words, total_documents):
      idf_matrix = {}

      for sent, f_table in freq_matrix.items():
          idf_table = {}

          for word in f_table.keys():
              idf_table[word] = math.log10(total_documents / float(count_doc_per_words[word]))

          idf_matrix[sent] = idf_table

      return idf_matrix


  def tf_idf_matrix(self, tf_matrix, idf_matrix):
      """
        generate tf-idf
      """
      tf_idf_matrix = {}

      for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):

          tf_idf_table = {}

          for (word1, value1), (word2, value2) in zip(f_table1.items(),
                                                      f_table2.items()):  # here, keys are the same in both the table
              tf_idf_table[word1] = float(value1 * value2)

          tf_idf_matrix[sent1] = tf_idf_table

      return tf_idf_matrix


  def score_sentences(self, tf_idf_matrix):
      """
        adding the TF frequency of every non-stop word in a sentence divided by total no of words in a sentence.
      """

      sentenceValue = {}

      for sent, f_table in tf_idf_matrix.items():
          total_score_per_sentence = 0

          count_words_in_sentence = len(f_table)
          for word, score in f_table.items():
              total_score_per_sentence += score

          sentenceValue[sent] = total_score_per_sentence / count_words_in_sentence

      return sentenceValue


  def average_score(self, sentenceValue):
      """
        Find the average score from the sentence value dictionary
      """
      sumValues = 0
      for entry in sentenceValue:
          sumValues += sentenceValue[entry]

      # Average value of a sentence from original summary_text
      average = (sumValues / len(sentenceValue))

      return average


  def generate_summary(self, sentences, sentenceValue, threshold, ratio=None):
      sentence_count = 0
      summary = ''
      thresholds = []

      if ratio:
        for sentence in sentences:
          if sentence[:self.limit] in sentenceValue:
            threshold = sentenceValue[sentence[:self.limit]]
            thresholds.append(threshold)
          else:
            thresholds.append(float('-inf'))
        nt = len(thresholds)
        target_idxs = np.argsort(thresholds)[::-1][:nt]
        ratio_seq = np.arange(int(nt*ratio))
        ratio_target_idxs = target_idxs[ratio_seq]
        target_sentences = np.array(sentences)[ratio_target_idxs]
        summary = " ".join(target_sentences)
      else:
        for sentence in sentences:
            if sentence[:self.limit] in sentenceValue and sentenceValue[sentence[:self.limit]] >= (threshold):
                summary += " " + sentence
                sentence_count += 1

      return summary