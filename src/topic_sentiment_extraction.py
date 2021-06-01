from nltk.stem.snowball import SnowballStemmer
import pandas as pd
import numpy as np
from collections import Counter
from ast import literal_eval

import spacy
nlp = spacy.load("en_core_web_sm")
stemmer = SnowballStemmer(language='english')


class ReviewAnalysor:

    def __init__(self, pos, neg, df, model):
        self.pos = pos
        self.neg = neg
        self.df = df
        self.model = model

    def topic_sentiment_extraction(self, sentence):
        """ For each review, creates a dictionary for each aspect mentioned.
        Gives the sentiment towards that aspect and an opinion term.A

        Args:
            sentence (string): sentence with a review.

        Returns:
            dictionary: aspect based sentiment analysis.
        """

        sent_dict = dict()
        sentence = nlp(sentence)
        opinion_words = sorted(self.neg + self.pos)
        for token in sentence:
            # check if the word is an opinion word, then assign sentiment
            if token.text in opinion_words:
                sentiment = 1 if token.text in self.pos else -1
                # if target is an adverb modifier (i.e. pretty, highly, etc.), ignore and pass
                if (token.dep_ == "advmod"):
                    continue
                elif (token.dep_ == "amod"):
                    sent_dict[token.head.text] = (sentiment, token.text)
                # for opinion words that are adjectives, adverbs, verbs...
                else:
                    for child in token.children:
                        # if there's a adj modifier (i.e. very, pretty, etc.) add more weight to sentiment
                        if ((child.dep_ == "amod") or (child.dep_ == "advmod")) and (child.text in opinion_words):
                            sentiment *= 1.5
                        # check for negation words and flip the sign of sentiment
                        if child.dep_ == "neg":
                            sentiment *= -1
                    for child in token.children:
                        # if verb, check if there's a direct object
                        if (token.pos_ == "VERB") & (child.dep_ == "dobj"):
                            sent_dict[child.text] = (sentiment, token.text)
                            # check for conjugates (a AND b), then add both to dictionary
                            subchildren = []
                            conj = 0
                            for subchild in child.children:
                                if subchild.text == "and":
                                    conj = 1
                                if (conj == 1) and (subchild.text != "and"):
                                    subchildren.append(subchild.text)
                                    conj = 0
                            for subchild in subchildren:
                                sent_dict[subchild] = (sentiment, token.text)

                    # check for nouns
                    for child in token.head.children:
                        if (child.pos_ == "NOUN") and (child.text not in sent_dict):
                            noun = child.text
                            # Check for compound nouns
                            for subchild in child.children:
                                if subchild.dep_ == "compound":
                                    noun = subchild.text + " " + noun
                            sent_dict[noun] = (sentiment, token.text)
        return sent_dict

    def predict_topic_sentiment(self, to_path):
        self.df['topic_sentiment'] = self.df['review_sentence'].apply(
            lambda x: self.topic_sentiment_extraction(x))
        self.df.to_csv(to_path)
        print('predict_topic_sentiment done.')
        return self.df

    def aggregate(self, data, aspect):
        """ Aggregates all reviews mentioning a specific aspect.

        Args:
            data (dataframe): column of dataframe containing each aspect sentiments.
            aspect (string): example "staff","pool","food"

        Returns:
            list: all dictionaries of aspects similar to the chosen one, with sentiment and opinion.
        """

        try:
            data['topic_sentiment'] = data['topic_sentiment'].apply(
                literal_eval)
            topics = list(set([item for sublist in data['topic_sentiment'].apply(
                lambda x:list(x.keys())).tolist() for item in sublist]))
            topics_stemmed = [
                ' '.join([stemmer.stem(x) for x in topic.split()]) for topic in topics]
            aspect_topics = [topic for topic in topics_stemmed if topic in self.model.index_to_key and self.model.similarity(
                aspect, topic) > 0.6]
            topics_sentiment_opinion = [{key: value} for d in data['topic_sentiment'] for key, value in d.items(
            ) if ' '.join([stemmer.stem(x) for x in key.split()]) in aspect_topics]
            return topics_sentiment_opinion
        except ValueError:
            topics = list(set([item for sublist in data['topic_sentiment'].apply(
                lambda x:list(x.keys())).tolist() for item in sublist]))
            topics_stemmed = [
                ' '.join([stemmer.stem(x) for x in topic.split()]) for topic in topics]
            aspect_topics = [topic for topic in topics_stemmed if topic in self.model.index_to_key and self.model.similarity(
                aspect, topic) > 0.6]
            topics_sentiment_opinion = [{key: value} for d in data['topic_sentiment'] for key, value in d.items(
            ) if ' '.join([stemmer.stem(x) for x in key.split()]) in aspect_topics]
            return topics_sentiment_opinion

    # mean value of all sentiments towards one topic
    def topic_sentiment_mean(self, topics_sentiment_opinion):
        return np.mean(np.array([list(d.values())[0][0] for d in topics_sentiment_opinion]))

    # total number of reviews
    def topic_count(self, topics_sentiment_opinion):
        return len(topics_sentiment_opinion)

    # positive and negative count of one topic
    def topic_positive_negative_count(self, topics_sentiment_opinion):
        sentiment = np.array([list(d.values())[0][0]
                             for d in topics_sentiment_opinion])
        return len(sentiment[sentiment > 0]), len(sentiment[sentiment < 0])

    # positive opinions of one topic

    def topic_positive_opinion(self, topics_sentiment_opinion):
        opinions = dict(Counter([list(d.values())[
                        0][1] for d in topics_sentiment_opinion if list(d.values())[0][0] > 0]))
        return {k: v for k, v in sorted(opinions.items(), key=lambda item: item[1], reverse=True)}

    # negative opinions of one topic
    def topic_negative_opinion(self, topics_sentiment_opinion):
        opinions = dict(Counter([list(d.values())[
                        0][1] for d in topics_sentiment_opinion if list(d.values())[0][0] < 0]))
        return {k: v for k, v in sorted(opinions.items(), key=lambda item: item[1], reverse=True)}

    # extract aspects from one topic

    def extract_aspects(self, topics_sentiment_opinion):
        return list(set([stemmer.stem(list(d.keys())[0]) for d in topics_sentiment_opinion]))

    # mean value of sentiments towards each aspect in one topic
    def aspects_sentiment_mean(self, topics_sentiment_opinion):
        result = {}
        for topic in self.extract_aspects(topics_sentiment_opinion):
            result[topic] = np.mean(np.array([list(d.values())[
                                    0][0]for d in topics_sentiment_opinion if stemmer.stem(list(d.keys())[0]) == topic]))
        return result

    # count of each aspect in one topic
    def aspects_count(self, topics_sentiment_opinion):
        result = {}
        for topic in self.extract_aspects(topics_sentiment_opinion):
            result[topic] = len(
                [d for d in topics_sentiment_opinion if stemmer.stem(list(d.keys())[0]) == topic])
        return result

    # positive and negative count of aspects
    def aspects_positive_negative_count(self, topics_sentiment_opinion):
        result = {}
        for topic in self.extract_aspects(topics_sentiment_opinion):
            sentiment = np.array([list(d.values())[
                                 0][0] for d in topics_sentiment_opinion if stemmer.stem(list(d.keys())[0]) == topic])
            result[topic] = (len(sentiment[sentiment > 0]),
                             len(sentiment[sentiment < 0]))
        return result

    # positive opinions of each aspect in one topic
    def aspects_positive_opinion(self, topics_sentiment_opinion):
        result = {}
        for topic in self.extract_aspects(topics_sentiment_opinion):
            opinions = dict(Counter([list(d.values())[0][1] for d in topics_sentiment_opinion if list(
                d.values())[0][0] > 0 and stemmer.stem(list(d.keys())[0]) == topic]))
            result[topic] = {k: v for k, v in sorted(
                opinions.items(), key=lambda item: item[1], reverse=True)}
        return result

    # negative opnions of each aspect in one topic
    def aspects_negative_opinion(self, topics_sentiment_opinion):
        result = {}
        for topic in self.extract_aspects(topics_sentiment_opinion):
            opinions = dict(Counter([list(d.values())[0][1] for d in topics_sentiment_opinion if list(
                d.values())[0][0] < 0 and stemmer.stem(list(d.keys())[0]) == topic]))
            result[topic] = {k: v for k, v in sorted(
                opinions.items(), key=lambda item: item[1], reverse=True)}
        return result

    def print_results_per_topic(self, data, topic):
        topics_sentiment_opinion = self.aggregate(data, topic)
        print("Mean Value of all Sentiments towards this topic", str(
            self.topic_sentiment_mean(topics_sentiment_opinion)))
        print("Aggregate Number of Sentiments on that topic",
              str(self.topic_count(topics_sentiment_opinion)))
        print("Positive and Negative count towards that topic", str(
            self.topic_positive_negative_count(topics_sentiment_opinion)))
        print("Positive Opinions of that topic", str(
            self.topic_positive_opinion(topics_sentiment_opinion)))
        print("Negative Opinions of that topic", str(
            self.topic_negative_opinion(topics_sentiment_opinion)))
        print("Aspects from that topic", str(
            self.extract_aspects(topics_sentiment_opinion)))
        print("Mean Value of Sentiments towards each Aspect in one topic",
              str(self.aspects_sentiment_mean(topics_sentiment_opinion)))
        print("Count of each Aspect in one topic", str(
            self.aspects_count(topics_sentiment_opinion)))
        print("Positive and Negative Count of Aspects", str(
            self.aspects_positive_negative_count(topics_sentiment_opinion)))
        print("Positive Opinions of each Aspect in one topic", str(
            self.aspects_positive_opinion(topics_sentiment_opinion)))
        print("Negative Opnions of each Aspect in one topic", str(
            self.aspects_negative_opinion(topics_sentiment_opinion)))

    # find a number of most topics mentioned in reviews
    def find_most_frequent_words(self, number):
        print("------- MOST FREQUENT TOPICS ---------")
        try:
            # Necessary for formating of dictionary
            self.df['topic_sentiment'] = self.df['topic_sentiment'].apply(
                literal_eval)
            d = dict(Counter([item for sublist in self.df['topic_sentiment'].apply(
                lambda x:list(x.keys())).tolist() for item in sublist]))
            d_sorted = {k: v for k, v in sorted(
                d.items(), key=lambda item: item[1], reverse=True)[:number]}
            print(d_sorted)
        except ValueError:
            d = dict(Counter([item for sublist in self.df['topic_sentiment'].apply(
                lambda x:list(x.keys())).tolist() for item in sublist]))
            d_sorted = {k: v for k, v in sorted(
                d.items(), key=lambda item: item[1], reverse=True)[:number]}
            print(d_sorted)

    def print_aggregated_results(self, parc_name):

        """ Prints available results for a specific parc.
            Splits before 2017 and after 2017 for more granularity.
            Computes results for each aspect of each topics.
        """

        data_prior2017 = self.df[(self.df.year != '2019') & (self.df.year != '2018') & (
            self.df.year != '2017') & (self.df.hotel_name == parc_name)]
        data_after2017 = self.df[(self.df.year == '2019') | (self.df.year == '2018') | (
            self.df.year == '2017') & (self.df.hotel_name == parc_name)]

        # Topics
        facilities = ['accommodation', 'place', 'lodge', 'facilities']
        other = ['issue', 'problem', 'value']
        activities = ['pool', 'activities']
        food_drinks = ['food', 'drinks', 'drink', 'restaurant', 'restaurants']
        service = ['service', 'staff']

        topics = [facilities, other, activities, food_drinks, service]
        print("-------------- NEW PARK: {} -------------------".format('Center Parcs Elveden Forest'))
        for idx, data in enumerate([data_prior2017, data_after2017]):
            print("-------------- NEW DATASET: {} -------------------".format('Prior to 2017')
                  ) if idx == 0 else print("-------------- NEW PARK {} -------------------".format('2017 and after'))
            for topic in topics:
                print("--------------- NEW TOPIC: {} ---------------".format(topic))
                for word in topic:
                    print('--------- NEW WORD: {} -------'.format(word))
                    print(self.print_results_per_topic(data, word))
