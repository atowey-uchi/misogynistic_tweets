# Women in Polictics and Misogynistic Twitter Mentions

### Objective
1. Classify tweets containing explicit and implicit misogynistic content 
2. Identify the themes underpinning this content

### Methodology
1. Scrape tweets tagging women in Congress and Senate in the week leading up to the 2020 elections. 
2. Use BERT to classify misogynistic and non-misogynistic twitter mentions
3. Compare both misogynistic and non-misogynistic tweets
4. Perform unsupervised classification (LDA Topic Model) to identify themes in non-misogynistic tweets
5. Communicate results in an interactive dashboard

### Data
We scraped over 400,000 twitter mentions for 146 female senators. A sample of the scraped tweets can be found in "final_data.csv" and the larger file (which was too large to upload on github without git lfs) can be found at https://drive.google.com/file/d/1HLWQuaJzQwqlYYGOHjspKyiQwPAwQK-T/view?usp=sharing. We also used data from Congress.gov and GovTrack USA to conduct further analysis.
All the data used for this project can be found under the "Data" folder, and the raw urls (manually compiled) used for getting twitter handles of female senators, and other data can be found under the folder "Scraping tweets -data+code". The trained topic models cab be found under "ldamodels". 

### Code
The code used for the models and data analysis and their respective folders can be found as mentioned below.
1. Scraping twitter mentions: Scraping_tweets.ipynb in the folder "Scraping tweets- data+code. The file might be slightly different from what was shown in the video to increase readability. However, only markdowns and comments were changed, and not the actual code.
2. BERT model : BERT_Classifier_Final.ipynb
3. LDA Topic Model : Topic_Model_Final.ipynb
4. Metadata Cleaning, Word cloud, and Linear Regression : Wordcloud_Regression.ipynb
5. Dashboard and data visualization : Visualization.ipynb
6. Functions used in Wordcloud_Regression.ipynb : util.py
7. Files for Heroku App hosting : requirements.txt, app.py, runtime.txt, Procfile, .gitignore, assets

### Visualization Link
https://twittermisogyny.herokuapp.com/

### Code Overview Video Link
https://drive.google.com/file/d/1J39UnD9S6Pad5nzrKUpiNHVBrt2L8c86/view?usp=sharing

### Packages 
Packages required to run each notebook. 
1. Scraping data from Twitter: (Twint, 2.1.21), (pandas, 1.2.4). In order to be able to scrape data for more than one day using twint, one would have to uncomment                                  line 92 from the url.py file of the downloaded package. This can be done by typing "pip3 show twint" in the terminal, changing the                                      directory to that path, and editing the url.py file using nano, vim or any suitable text editor.  
2. BERT : (pandas, 1.3.5), (numpy, 1.21.5), (keras, 2.8.0), (tensorflow, 2.8.0), 
          (nltk, 3.2.5), (sklearn, 1.0.2)
3. Topic Model: (gensim, 4.1.2), (sklearn, 0.23.2), (numpy, 1.19.2), (pandas, 1.1.3), 
                (nltk, 3.5), (wordcloud, 1.8.1), (pyLDAvis, 3.3.1), (statsmodels, 0.13.2), (nltk, 3.6.7), 
4. Metdata cleaning, Word Cloud and Linear Regression: (pandas, 1.4.1), (seaborn, 0.11.2),  (matplotlib, 3.5.1), (gensim, 4.1.2), sklearn(1.0.2), (numpy, 1.21.5)
5. Data Visualization : (pandas, 1.4.1), (numpy, 1.20.3), (dash, 2.2.0), (base64, 1.0.0), (plotly, 5.6.0), (dash-bootstrap-components, 1.0.3), (dash-core-                                    components, 2.0.0), (dash-html-components, 2.0.0),(dash-table, 5.0.0)
