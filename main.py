from keras.models import load_model
from sklearn.feature_extraction.text import CountVectorizer
import pickle

model = load_model(
    "C:\D\College Stuff\ACM\ICSPN Paper\Codes/best-mail-classifier-colab.h5")

cv = CountVectorizer()

with open("spam-detection\corpus.p", "rb") as file:
    corpus = pickle.load(file)
cv.fit_transform(corpus)


def checkMail(testmail):

    predictMail = predict(testmail)
    if predictMail < 0.5:
        print("The mail is not a spam")
    else:
        print("The mail is a spam")


def predict(mail):

    pred = model.predict(cv.transform([mail]))[0]
    predictionValue = pred[0]

    return predictionValue


checkMail("Hello Barry how are you")
