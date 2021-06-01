import pandas as pd
import nltk


def preprocessing_review_to_sentence(from_path,to_path):
  # Read initial dataset
  df = pd.read_json(from_path)
  df = df[(df.review_language == 'en')][6:]

  # Retrieve year from trip_date
  year = []
  for i in range(df.shape[0]):
    try:
      data = str(df['trip_date'].values[i]).split()[1]
    except IndexError:
      data = ''
    year.append(data)
  df['year'] = year

  # Split review into sentences
  df['split_review'] = df.review.map(lambda x : nltk.tokenize.sent_tokenize(x))

  review_ids = []
  sentences = []
  for i in range(df.shape[0]):
    for s in df['split_review'].values[i]:
      review_ids.append(df.review_id.values[i])
      sentences.append(s)

  # Create new dataset made of review sentences.
  data_tuples = list(zip(review_ids,sentences))
  df_s = pd.DataFrame(data_tuples, columns=['review_id','review_sentence'])

  # Add information from initial dataset
  df_s = pd.merge(df_s, df,  how='left', left_on=['review_id'], right_on = ['review_id'])
  df_s.to_csv(to_path,index=False)
  print('preprocessing_review_to_sentence done.')
  return df_s