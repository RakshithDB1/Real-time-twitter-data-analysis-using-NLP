# -*- coding: utf-8 -*-
import tweepy
import re
import matplotlib.pyplot as plt
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import pandas as pd


try:
    import joblib
    model = joblib.load('nlpmodel.pkl')
    print('nlp model loaded')
except:
    print('unable to load nlp model')

try:
    df = pd.read_csv('cleanedData.tsv',delimiter='\t')
    from sklearn.feature_extraction.text import TfidfVectorizer
    tv = TfidfVectorizer(max_df = 0.9,min_df = 2,max_features = 5000, stop_words ='english',ngram_range=(1,2))
    sparse = tv.fit_transform(df['review']).toarray()
    print('vectorizer loaded')
except:
    print('unable to load vectorizer')

def removePat(strg,pat,nstr):
    return re.sub(pat,nstr,strg)

def tweetsfetching(search_keyword):
     consumerKey = 'slePK4TLFZ1MjkXd89ziHCqlJ'
     consumerSecret = '55n8r3IpbjkDZEyCLLtBxCwWV9qGEHIVbidheL9uUJwziyWT51'
     accessToken = '1153685727256829954-8Dq3LS2FZqZo1dnfLCwpbu98qcX2Xq'
     accessTokenSecret = 'Pi5MI7zQAiBs7RIfQB7jdl6oXqpJGDBc3nQrJx0sS0AFR'
     try:
        auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
        auth.set_access_token(accessToken, accessTokenSecret)
        api = tweepy.API(auth,wait_on_rate_limit=True)
        print('twitter api ready')
     except:
       print("error")

     try:
        tweets = []
        tweets = tweepy.Cursor(api.search, q=search_keyword+' -filter:retweets', lang = "en",tweet_mode='extended').items(110)
        tweetdf = []
        print('tweets extracted')
        # Collect tweets
        for tweet in tweets:
          tweetdf.append(tweet.full_text.lower())

        tweetdf = np.vectorize(removePat)(tweetdf,'@[\w]+','')
        tweetdf = np.vectorize(removePat)(tweetdf,r'http\S+','')
        tweetdf = np.vectorize(removePat)(tweetdf,'[^a-zA-Z0-9 ]','')
        print('tweets cleaned')

        #vectorize tweets
        tweetdf1 = tv.transform(tweetdf).toarray()
        print('tweets transformed')
        #predict tweets
        tweetPred = model.predict(tweetdf1)
        negTwt = np.count_nonzero(tweetPred==0)
        posTwt = np.count_nonzero(tweetPred==1)
        #totTwt = negTwt + posTwt
        print('prediction completed')
        print(posTwt)
        print(negTwt)
        return np.array([negTwt, posTwt])


     except:
        print("Unfortunately, something went wrong..")
        return None




#Gui class for page 2
class Ui_MainWindow2(object):
    #class variable to kept track of possible and negative count
    pn=np.array([0,0])
    def setupUi(self, MainWindow,posneg):
        self.pn=posneg
        MainWindow.setObjectName("Real time twitter data analysis")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(230, 10, 301, 61))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        #push button for piechart
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(120, 490, 75, 23))
        self.pushButton.setObjectName("pushButton")
        #push button for bar graph
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(560, 490, 75, 23))
        self.pushButton_2.setObjectName("pushButton_2")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(90, 80, 661, 381))
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")


        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        #connecting the button for the appropriate fuctions
        self.pushButton.clicked.connect(self.piechartdisplay)
        self.pushButton_2.clicked.connect(self.barplotdisplay)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Real Time twitter data Analysisier"))
        self.label.setText(_translate("MainWindow", "RESULTS OF ANALYSIS"))
        self.pushButton.setText(_translate("MainWindow", "pie chart"))
        self.pushButton_2.setText(_translate("MainWindow", "bar graph"))
    def piechartdisplay(self,centralwidget):
        posneg= self.pn
        moods = ['negative','positive']
        cols = ['r','g']
        plt.pie(posneg,labels=moods,colors=cols,startangle=90,shadow= True,explode=(0,0.1),autopct='%1.1f%%')
        plt.savefig("photos/piechart.png")
        self.label_2.clear()
        self.label_2.setPixmap(QtGui.QPixmap("photos/piechart.png"))
        self.label_2.setScaledContents(True)

    def barplotdisplay(self,centralwidget):
        moods=['negative', 'positive']
        values=self.pn
        plt.bar(moods, values)
        plt.xticks(rotation='30')
        plt.savefig("photos/barplot.png",dpi=400)
        self.label_2.clear()
        self.label_2.setPixmap(QtGui.QPixmap("photos/barplot.png"))
        self.label_2.setScaledContents(True)


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        MainWindow.setStatusTip("")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(190, 0, 351, 51))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(230, 60, 271, 211))
        self.label_2.setText("")
        self.label_2.setPixmap(QtGui.QPixmap("photos/twitter.png"))
        self.label_2.setScaledContents(True)
        self.label_2.setWordWrap(False)
        self.label_2.setObjectName("label_2")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(310, 520, 141, 23))
        self.pushButton.setObjectName("pushButton")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(440, 460, 231, 20))
        self.lineEdit.setObjectName("lineEdit")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(20, 460, 291, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_3.setFont(font)
        self.label_3.setScaledContents(False)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(26, 303, 741, 131))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_4.setFont(font)
        self.label_4.setScaledContents(True)
        self.label_4.setWordWrap(True)
        self.label_4.setObjectName("label_4")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.pushButton.clicked.connect(self.piedisplay)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def piedisplay(self,MainWindow):
        hashtag=self.lineEdit.text()
        #submitting hashtag to tweetsfetching function
        posneg=tweetsfetching(hashtag)
        #Displaying the second window
        self.window=QtWidgets.QMainWindow()
        self.ui=Ui_MainWindow2()
        self.ui.setupUi(self.window, posneg)
        self.window.show()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Real Time twitter data Analysisier"))
        self.label.setText(_translate("MainWindow", "WELCOME"))
        self.pushButton.setText(_translate("MainWindow", "SUBMIT"))
        self.label_3.setText(_translate("MainWindow", "ENTER THE HASHTAG"))
        self.label_4.setText(_translate("MainWindow", "The online medium has become a significant way for people to express their opinions and with\n"
"social media, there is an abundance of opinion information available. \n"
"\n"
"Using Real-twitter-data analysisier,the polarity of opinions can be found, such as positive, or negative  by analyzing the textof the opinion."))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
