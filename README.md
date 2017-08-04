# Is that possible for us to predict the possible sales number of Video games?

## Yes we can! ##

By using the video games data from kaggle to  see if DL can preidct the  sales number

Data from:

https://www.kaggle.com/rush4ratio/video-game-sales-with-ratings

Im trying to use NN to predict the world sales number of video games.

After train the model the regression plot is like:

Model predict the training data

![image](https://github.com/johnny7861532/Keras-Video-Games-Sales-number-predict/blob/master/video%20game%20train%20data%20predict.png)

Model predict the testing data

![image](https://github.com/johnny7861532/Keras-Video-Games-Sales-number-predict/blob/master/video%20games%20test%20predict.png)

According to these two pictures we can see our model predict poorly when the sales number is over 20.
And when the sales over 15 our model start to get the lower acc of predict numbers.
It might becasue the imbalance data that the sales number over 15 are only 10, ratio is only 0.001257.



