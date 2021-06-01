import pandas as pd
from gensim import downloader
from nltk.corpus import opinion_lexicon
from src.preprocessing import preprocessing_review_to_sentence
from src.topic_sentiment_extraction import ReviewAnalysor
import argparse
import nltk
nltk.download('opinion_lexicon')
nltk.download('punkt')


def main(args):
    df = preprocessing_review_to_sentence(args.data_from,args.to_path1)
    neg = list(opinion_lexicon.negative())
    pos = list(opinion_lexicon.positive())
    glove = downloader.load('glove-wiki-gigaword-50')
    try:
        df = pd.read_csv(args.to_path2)
        analyser = ReviewAnalysor(pos, neg, df, glove)
    except Exception:
        analyser = ReviewAnalysor(pos, neg, df, glove)
        df = analyser.predict_topic_sentiment(args.to_path2)
    analyser.find_most_frequent_words(args.number_words)
    analyser.print_aggregated_results(args.parc_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Customer Review Analysis for Center Parcs Europe")
    parser.add_argument("--data_from", type=str, default="./data/full.json")
    parser.add_argument("--to_path1", type=str, default="./data/sentence.csv")
    parser.add_argument("--to_path2", type=str,
                        default="./data/sentence_with_opinion.csv")
    parser.add_argument("--number_words", type=int, default=10)
    parser.add_argument("--parc_name", type=str,
                        default='Center Parcs Sherwood Forest')
    args = parser.parse_args()
    main(args)
