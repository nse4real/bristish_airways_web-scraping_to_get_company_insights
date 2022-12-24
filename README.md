# bristish_airways_web-scraping_to_get_company_insights
#Background
British Airways (BA) is the flag carrier airline of the United Kingdom (UK). Every day, thousands of BA flights arrive to and depart from the UK, carrying customers across the world. Whether it’s for holidays, work or any other reason, the end-to-end process of scheduling, planning, boarding, fuelling, transporting, landing, and continuously running flights on time, efficiently and with top-class customer service is a huge task with many highly important responsibilities.
It is the job of a Data Scientist at BA to apply his analytical skills to influence real life multi-million-pound decisions from day one, making a tangible impact on the business as his recommendations, tools and models drive key business decisions, reduce costs and increase revenue.
Customers who book a flight with BA will experience many interaction points with the BA brand. Understanding a customer's feelings, needs, and feedback is crucial for any business, including BA.
this task is focused on scraping and collecting customer feedback and reviewing data from a third-party source and analysing this data to present any insights you may uncover.
#Task
##Scrape data from the web
The first thing to do will be to scrape review data from the web. For this, we use a website called Skytrax.

The team leader wants us to focus on reviews specifically about the airline itself. We collect as much data as we can in order to improve the output of our analysis. To get started with the data collection, we can use the “Jupyter Notebook” to run some Python code that will help to collect some data.
##Present insights
The manager would like us to summarise our findings within a single PowerPoint slide, so that she can present the results at the next board meeting. We would create visualisations and metrics to include within this slide, as well as clear and concise explanations in order to quickly provide the key points from our analysis.

## Code snippets

###importing libraries for web-scraping & data analysis
import requests
from bs4 import BeautifulSoup
import pandas as pd

This code imports several libraries that are used for web scraping and data analysis.

requests is a library for making HTTP requests in Python. It allows you to send HTTP requests using Python and receive responses from servers. This can be useful for interacting with APIs or for web scraping.

BeautifulSoup is a library for parsing and navigating HTML and XML documents. It provides a convenient interface for extracting data from these documents and is often used in combination with the requests library for web scraping.

pandas is a library for data manipulation and analysis in Python. It provides data structures for storing and manipulating large amounts of data, as well as tools for working with this data. It is commonly used for tasks such as reading and writing data to and from various file formats (e.g., CSV, Excel), aggregating and summarizing data, and performing data analysis.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------


base_url = "https://www.airlinequality.com/airline-reviews/british-airways"
pages = 10
page_size = 100

reviews = []

# for i in range(1, pages + 1):
for i in range(1, pages + 1):

    print(f"Scraping page {i}")

    # Create URL to collect links from paginated data
    url = f"{base_url}/page/{i}/?sortby=post_date%3ADesc&pagesize={page_size}"

    # Collect HTML data from this page
    response = requests.get(url)

    # Parse content
    content = response.content
    parsed_content = BeautifulSoup(content, 'html.parser')
    for para in parsed_content.find_all("div", {"class": "text_content"}):
        reviews.append(para.get_text())
    
    print(f"   ---> {len(reviews)} total reviews")
    
    
    This code is using the requests and BeautifulSoup libraries to scrape a website for customer reviews. It is specifically scraping the website "https://www.airlinequality.com/airline-reviews/british-airways" for British Airways reviews.

The base_url variable is a string that holds the base URL of the website that we want to scrape. The pages and page_size variables are integers that determine how many pages of data we want to scrape and how many reviews should be displayed on each page.

The reviews list is initially empty and will be used to store the customer reviews that are scraped from the website.

The code then enters a loop that iterates over a range of integers from 1 to pages + 1. On each iteration of the loop, it prints a message indicating which page it is currently scraping, constructs a URL for that page using the base_url, page_size, and loop variable i, and then makes an HTTP request to that URL using the requests.get() function.

The HTML content of the webpage is then extracted from the response and passed to the BeautifulSoup function to be parsed. The parsed content is stored in the parsed_content variable.

The code then searches the parsed content for HTML elements with the class "text_content" using the find_all() method. For each of these elements, it extracts the text content using the get_text() method and appends it to the reviews list.

Finally, the code prints the total number of reviews that have been scraped so far.

------------------------------------------------------------------------------------------------------------------------------------------------------------------

df = pd.DataFrame()
df["reviewsx"] = reviews
df.head()

This code creates a pandas DataFrame object named df and adds a column called "reviews" to it, using the data stored in the reviews variable. The pd.DataFrame() function creates an empty DataFrame, and the df["reviews"] = reviews assignment adds a new column to the DataFrame with the label "reviews" and the data from the reviews variable.

The df.head() method at the end displays the first few rows (by default, the first five rows) of the DataFrame. This is often used as a quick way to check that the data has been correctly loaded and formatted.

It is worth noting that this code assumes that the reviews variable contains a list or series of values that can be used to populate the "reviews" column in the DataFrame. If reviews is not defined or is not a compatible data type, this code will raise an error. In this case it contains text contents.

----------------------------------------------------------------------------------------------------------------------------------------------------------------

Now we have our dataset for this task! The loops above collected 1000 reviews by iterating through the paginated pages on the website. However, we could try increasing the number of pages!

 The next thing that we would do is clean this data to remove any unnecessary text from each of the rows. For example, "✅ Trip Verified" can be removed from each row if it exists, as it's not relevant to what we want to investigate.
 
 import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

This code imports the necessary libraries and downloads the required data files from nltk. It then defines a list of English stop words and a WordNetLemmatizer object from the nltk library.

----------------------------------------------------------------------------------------------------------------------------------------------------------------

# Remove HTML tags
df['reviews'] = df['reviewsx'].apply(lambda x: BeautifulSoup(x, 'lxml').get_text())

Next, it applies a lambda function to the "reviews" column of the DataFrame to remove HTML tags using the BeautifulSoup library.

---------------------------------------------------------------------------------------------------------------------------------------------------------------

# Remove punctuation and special characters
df['reviews'] = df['reviews'].apply(lambda x: re.sub('[^\w\s]', '', x))

It then applies another lambda function to remove punctuation and special characters using the re library.

------------------------------------------------------------------------------------------------------------------------------------------------------------

# Convert all text to lowercase
df['reviews'] = df['reviews'].apply(lambda x: x.lower())

It follows this by applying a lambda function to convert all text to lowercase

------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Remove stop words
stop_words = set(stopwords.words('english'))
df['reviews'] = df['reviews'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

The code then applies a lambda function to remove stop words from the "reviews" column using the list of stop words and the join() method.

-------------------------------------------------------------------------------------------------------------------------------------------------------------

# Tokenize the text
df['reviews'] = df['reviews'].apply(lambda x: nltk.word_tokenize(x))

It then applies a lambda function to tokenize the text in the "reviews" column using the word_tokenize() function from nltk.

-------------------------------------------------------------------------------------------------------------------------------------------------------------

# Stem or lemmatize the tokens
lemmatizer = WordNetLemmatizer()
df['reviews'] = df['reviews'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

The code applies a lambda function to stem or lemmatize the tokens in the "reviews" column using the lemmatize() method of the WordNetLemmatizer object

---------------------------------------------------------------------------------------------------------------------------------------------------------------

# Remove meaningless words like "a," "an," "the," and words that do not provide meaningful information like "very," "really"
meaningless_words = ['a', 'an', 'the', 'very', 'really']
df['reviews'] = df['reviews'].apply(lambda x: [word for word in x if word not in meaningless_words])

This code applies a lambda function to remove meaningless words like "a," "an," "the," and words that do not provide meaningful information like  "very", really".

------------------------------------------------------------------------------------------------------------------------------------------------------------

# Remove "✅ Trip Verified" and "Not Verified"
df['reviews'] = df['reviews'].apply(lambda x: [word for word in x if word != '✅' and word != 'Trip' and word != 'Verified' and word != 'Not'])

This code applies a lambda function to the "reviews" column that iterates over the list of tokens in each row and removes any occurrences of the specified words

------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Create a dictionary and a corpus using the processed reviews
dictionary = gensim.corpora.Dictionary(df['reviews'])
corpus = [dictionary.doc2bow(review) for review in df['reviews']]

-----------------------------------------------------------------------------------------------------------------------------------------------------------------

# Train a Latent Dirichlet Allocation (LDA) model on the corpus
model = gensim.models.LdaModel(corpus, num_topics=5, id2word=dictionary)

Trains an LDA model on the resulting corpus.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------

# Print the topics of the LDA model
topics = model.print_topics()
for topic in topics:
  print(topic)
  
  It then prints the topics of the LDA model
  ----------------------------------------------------------------------------------------------------------------------------------------------------------------
  
  # Compute the sentiment of each review using TextBlob
sentiments = [TextBlob(review).sentiment.polarity for review in reviews]

This computes the sentiment of each review using TextBlob.
-------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Print the average sentiment of the reviews
print(f'Average sentiment: {np.mean(sentiments):.2f}')

Finally, it prints the average sentiment of the reviews.

------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Generate a word cloud for the most common words in the reviews
review_text = ' '.join(df['reviewsx'])
wordcloud = WordCloud(max_font_size=50).generate(review_text)

# Plot the word cloud
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

----------------------------------------------------------------------------------------------------------------------------------------------------
Average sentiment: 0.08
Sentiment from -1 (very negative) to 1 (very positive)

Overall sentiment of the reviews is slightly positive.

Positive words are present but not necessarily dominating reviews.

Context is important in interpreting sentiment values


Neutral words dominate the word cloud
Some positive words like “good”, “nice”
Negative words like “delayed”

