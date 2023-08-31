import mwclient
import time
import pandas as pd
from transformers import pipeline as sentimentPipeline
from statistics import mean
from datetime import datetime

site = mwclient.Site('en.wikipedia.org')
page = site.pages['Bitcoin']

revisions = list(page.revisions())

revisions = sorted(revisions, key=lambda rev: rev["timestamp"]) 

sentimentPipeline = sentimentPipeline("sentiment-analysis")

def find_sentiment(text):
    sentPip = sentimentPipeline([text[:250]])[0]
    sentimentScore = sentPip["score"]
    if sentPip["label"] == "NEGATIVE":
        sentimentScore *= -1
    return sentimentScore

editsMap = {}

for rev in revisions:        
    date = time.strftime("%Y-%m-%d", rev["timestamp"])
    if date not in editsMap:
        editsMap[date] = dict(sentiments=list(), edit_count=0)
    
    editsMap[date]["edit_count"] += 1
    
    comment = rev.get("comment", "")
    editsMap[date]["sentiments"].append(find_sentiment(comment))

for key in editsMap:
    if len(editsMap[key]["sentiments"]) > 0:
        editsMap[key]["sentiment"] = mean(editsMap[key]["sentiments"])
        editsMap[key]["neg_sentiment"] = len([s for s in editsMap[key]["sentiments"] if s < 0]) / len(editsMap[key]["sentiments"])
    else:
        editsMap[key]["sentiment"] = 0
        editsMap[key]["neg_sentiment"] = 0
    
    del editsMap[key]["sentiments"]

edits_df = pd.DataFrame.from_dict(editsMap, orient="index")

edits_df.index = pd.to_datetime(edits_df.index)

dates = pd.date_range(start="2009-03-08", end=datetime.today())

edits_df = edits_df.reindex(dates, fill_value=0)

currentEdits = edits_df.rolling(30, min_periods=30).mean()
currentEdits = currentEdits.dropna()

currentEdits.to_csv("wikipedia_edits.csv")
