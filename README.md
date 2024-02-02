# Introduction

In this repository, I provided a full solution to the challenge provided by Visable during the working student hiring process. 

The challenge is to train a model to classify German search queries from a provided dataset and deploy the final model using FastApi to make a REST API. In addition, the deployment process should be dockerized.

# Model

In 'Visable_coding_challenge.py' there is a full documentation of the code and the models used. I will provide a full description of my process here.

## Data importing, inspection, and preprocessing:

* After importing the data, I begin to see how many entries and classes are in the data set, and how many Null values are in it.
* The data has null values, non-German characters, and numbers in the text column.
* I made sure using reg-ex that all the characters in the dataset are German characterless. Then I used the SpaCy German model to do lemmatization and remove punctuation and numbers.
* Dropped all the null values from the dataset.
* Now the data is ready for feature extraction

### TF-IDF:

I used TF-IDF as a feature extractor. I used it as it's known for its good results in text classification applications. After this step, the data is ready to be split  into train and test sets for test sets to be 20% of the whole data

## Models:

* First, I used the Multinominal Naive Bayes model. I always take the simplest way to do the tasks, and then complex as we move. Naive Bayes is known for its good results in classification tasks and provides good results. As seen in the photo below.

![image](https://github.com/hishammadcor/HisAli753/assets/32823502/4ee9f607-351c-4491-945e-ca404b98dbf0)


* But, I wanted to increase the accuracy of the model, so I moved to the Random Forest classifier, it gives higher results than Naive Bayes, as seen from the photo below.

![image](https://github.com/hishammadcor/HisAli753/assets/32823502/730b4550-a3d0-47c8-a4fc-f689a8bc8f6c)


* But then I got the idea of Fine-tuning the BERT German Base model on the dataset and then using its embeddings on the Random Forest classifier. That's what I have done and it gets higher results as expected from the photo below.

![image](https://github.com/hishammadcor/HisAli753/assets/32823502/86dce947-ed74-44d9-8ae6-67699fdbf710)


# Setup

* For rerunning the models you should make sure that your environment has python==3.10.2 then run these commands:

```bash
pip3 install -r requirements.txt
python -m spacy download de_core_news_sm
pip3 install transformers[torch]
python3 Visable_coding_challenge.py
```

* Also, there is a docker image for testing the API, you can run it using this command:

```bash
docker run -d --name yourcontainername -p 8000:8000 visable_german
```

* This drive Link has essential models and check points for the deployment to run as they are larger than the github limit, espicially, 'fine_tuned_german_bert' folder:
  * link>> https://drive.google.com/drive/folders/1sO0i3lIITBa1wlMQYhm-LRokjhm-R799?usp=sharing



# Contact

Hisham Ali

Email: hisham3madcor@gmail.com

Mobile: +49 178 8953931
