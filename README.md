# Reinforcement Learning

In this project, we are trying to develope a Artificial Intelligence Agent (AI Agent) that will learn how to trade on a market using state of the art Deep Reinforcement Learning Algorithm.

## 1- Data Used to train

The dataset that we are going to use here are the bitcoin/usd asset as well as usd/cad datasets. 

bitcoin/usd dataset is coming from the competition platform called Kaggle : 
https://www.kaggle.com/mczielinski/bitcoin-historical-data
So it's a 1 min timestamp dataset and from this dataset we can create the hour timestamp if needed.

usd/cad dataset is coming from a bloomberg extraction, but many platform can give you this information.

In those dataset we have the following columns : Timestamp, Open, High, Low, Close, Volume. With those columns we can compute any technical indicators.

To create the technical indicators, I used the python library called TALib : https://github.com/mrjbq7/ta-lib

## 2- Create a learning environment for the Agent

To train an AI Agent to learn how to trade using Deep Reinforcement Learning algorithm, we need to create what we call a Learning Environment.

Using the same idea as OpenAI gym, I created a class where you can provide your financial time serie dataset and create for you a environment that you can use to train AI Agent. 

## 3- Modeling


