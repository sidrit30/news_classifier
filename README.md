# News Headline Classifier
This project's aim was to build a news classifier, which can determine the category of an article from its headline.


#### Team Members and Roles:
- Orgest Baçova - found the dataset, model evaluation, and interface design
- Sidrit Zela - dataset preprocessing, model training, and model hosting

#

### Dataset:
*[News Category Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset)*
Misra, Rishabh. "News Category Dataset." arXiv preprint arXiv:2209.11429 (2022).

From this dataset we have selected 12 of the most common categories:
- Business 
- Comedy
- Entertainment
- Food & Drink 
- Politics
- Science 
- Sports 
- Style & Beauty
- Tech
- Travel 
- Wellness
- World News

The methods for cleaning and preprocessing the dataset are described in each notebook.
A compact version of the dataset, storing only the headlines from articles of the selected categories is stored in the 'data' directory and used for demonstration purposes.

#

### AI Approaches:
For the classification task we chose two approaches, Multinomial Naive Bayes and roBERTa.

**Multinomial Naive Bayes** 

For the first approach we used a simple model, commonly used in text classification, the MultinomialNB model from scikit-learn. Before fitting the data to the model, it went through lemmatization and TF-IDF vectorization. More information can be found on the naive_bayes.ipynb notebook. This simple model was meant to be used as a baseline, and the accuracy was used as a floor for the other model.

**roBERTa**

For the second model, we used a more heavyweight model. We trained the roberta-base model from the huggingface transformers library for 3 epochs with a learning rate of 0.00002. Since roBERTa has hundreds of milions of parameters, training took considerably longer than MultinomialNB, but the performance was much better as well. Further information on preprocessing and parameters used for training can be found on the roberta_training notebook.

After training the model, it was hosted on HuggingFace: https://huggingface.co/sidrit30/roberta_news_classifier

## How to run

### Online
The app is deployed on Streamlit and can be found on: https://roberta-news-classifier.streamlit.app/

### Locally
If you want to run it locally:
- clone the repo or get project files
- on project root run from terminal: `pip install -r requirements.txt`
- then run: `streamlit install main.py`

### Train it yourself (GPU recommended)
If you want to change the parameters and train both models yourself:
- download the dataset from [kaggle](https://www.kaggle.com/datasets/rmisra/news-category-dataset)
- put the dataset inside the 'data' directory
- run the notebook corresponding to the model you want to train 


#
### Evaluation
Confusion matrices for both models are stored in the 'plots' directory. <br>
The test directory contains files used after the models had been trained for evaluation purposes. <br>
Further information about evaluation metrics can be found on the training notebooks. <br>

**Results:**

roBERTa: <br>
Testing accuracy: 89% (tested on 28135 samples) <br>
Weighted F1-score: 89%

MultinomialNB: <br>
Testing accuracy: 77% (tested on 27847 samples) <br>
Weighted F1-score: 77%

