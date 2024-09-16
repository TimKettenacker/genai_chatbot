# !/usr/bin/env python
# coding: utf-8

# # Custom Chatbot Project

# TODO: In this cell, write an explanation of which dataset you have chosen and why it is appropriate for this task
# I decided to use wikipedia data for the most recent season of "Rings of Power" as this data has not been incorporated
# into the Corpus of ChatGPT at the time of programming, so I'll have to teach it

# ## Data Wrangling
#
# TODO: In the cells below, load your chosen dataset into a `pandas` dataframe with a column named `"text"`.
#  This column should contain all of your text data, separated into at least 20 rows.


# Import Module
from bs4 import *
import requests
import re
import pandas

# Given URL
url = "https://en.wikipedia.org/wiki/The_Lord_of_the_Rings:_The_Rings_of_Power_season_2"
# Fetch URL Content
r = requests.get(url)
# Get body content
soup = BeautifulSoup(r.text, 'html.parser').select('body')[0]
# function to remove html tags from string
CLEANR = re.compile('<.*?>')

def cleanhtml(raw_html):
  cleantext = re.sub(CLEANR, '', raw_html)
  return cleantext

# html tag 'shortSummaryText' contains the episode contents from the wikipedia page
# let's clean them and split the list by dot and append to new list
all_episodes = []
for element in soup.find_all("div", {'class': "shortSummaryText"}):
    element_cleaned = cleanhtml(str(element.get_text))
    all_episodes.append(element_cleaned.strip().split('.'))

# collapse the list of lists into one
all_episodes_in_sentences = sum(all_episodes, [])
# put into data frame
episode_content_df = pandas.DataFrame(all_episodes_in_sentences)
episode_content_df.columns = ["text"]
# remove rows that contain just '>' from data frame (those are used to indicate a switch between episodes
i = episode_content_df[((episode_content_df['text'] == '>'))].index
episode_content_df = episode_content_df.drop(i)


# ## Custom Query Completion
#
# TODO: In the cells below, compose a custom query using your chosen dataset and retrieve results from an OpenAI `Completion` model.

from openai import OpenAI
client = OpenAI(
    base_url = "https://openai.vocareum.com/v1",
    api_key = "YOUR_API_KEY"
)
# define function to pose questions to chat gpt
def prompt_chat_gpt(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt},
        ]
    )
    return response

# pose to questions to openAI model
prompt1 = """
Who is Annatar in the second season of Rings of Power?
"""

prompt2 = """
Who captures Galadriel in the second season of Rings of Power?"
"""
# pose questions to chatgpt
initial_answer_1 = prompt_chat_gpt(prompt1)
initial_answer_2 = prompt_chat_gpt(prompt2)

# save prompt responses in a file to prevent overuse of API
with open('answers_to_initial_prompt1.csv','wt') as file1:
    file1.write(initial_answer_1.choices[0].message.content)

with open('answers_to_initial_prompt2.csv','wt') as file2:
    file2.write(initial_answer_2.choices[0].message.content)


# ## Custom Performance Demonstration
#
# TODO: In the cells below, demonstrate the performance of your custom query using at least 2 questions.
#  For each question, show the answer from a basic `Completion` model query as well as the answer
#  from your custom query.

# ## Generating Embeddings
#
# I will use the `embedding` tooling from OpenAI (https://platform.openai.com/docs/guides/embeddings/use-cases)
# to create vectors representing each row (text) of my custom dataset in order to instruct chatgpt to find the right answer.
def get_embedding(text, model="text-embedding-3-small"):
   return client.embeddings.create(input = [text], model=model).data[0].embedding

# append the result of the call to the embedding function to the respective row
episode_content_df['embedding'] = episode_content_df.text.apply(lambda x: get_embedding(x, model='text-embedding-3-small'))

# write to csv to be able to reuse them on demand at a later point in time
episode_content_df.to_csv('embedded_1k_reviews.csv', index=False)
# to load them later, run
# import numpy
# episode_content_df = pandas.read_csv('embedded_1k_reviews.csv')
# episode_content_df['embedding'] = episode_content_df.embedding.apply(eval).apply(numpy.array)

# # Find Related Pieces of Text for a Given Question
#
#  I will search through the text of all episodes semantically by embedding the question itself
#  and comparing it to the most fitting text.

import numpy
def cosine_similarity(a, b):
    return numpy.dot(a, b) / (numpy.linalg.norm(a) * numpy.linalg.norm(b))

def search(df, prompt):
    product_embedding = get_embedding(
        prompt,
        model="text-embedding-3-small"
    )
    df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, product_embedding))
    return df


# feeding the first prompt to the function
results1 = search(episode_content_df, prompt1)
closest_similarity_1 = episode_content_df.sort_values("similarity", ascending=False).head(1)
revised_answer_1 = closest_similarity_1['text'].values.tolist()
# write to csv
with open('revised_answers_prompt1.csv','wt') as file3:
    file3.write(str(revised_answer_1))

# feeding the first prompt to the function
results2 = search(episode_content_df, prompt2)
closest_similarity_2 = episode_content_df.sort_values("similarity", ascending=False).head(1)
revised_answer_2 = closest_similarity_2['text'].values.tolist()
# write to csv
with open('revised_answers_prompt2.csv','wt') as file4:
    file4.write(str(revised_answer_2))
# comparing the initial and the revised answers, it is clearly an improvement!


if __name__ == '__main__':
    print('script run successfully')

