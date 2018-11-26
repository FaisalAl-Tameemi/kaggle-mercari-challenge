import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import squarify


def countplot_top_k_categories(data, field_name, k=10):
    plt.figure(figsize=(17,10))
    sns.countplot(y = data[field_name], order=data[field_name].value_counts().iloc[:k].index, orient='v')
    plt.title("Top {} categories".format(k), fontsize = 25)
    plt.ylabel('Category Name', fontsize = 20)
    plt.xlabel('Number of Entries', fontsize = 20)

    return plt


def boxplot_target_to_field(data, target, field):
    return -1