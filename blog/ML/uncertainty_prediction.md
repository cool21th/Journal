### [Three Approaches to Predicting Uncertainty](https://www.kaggle.com/c/m5-forecasting-uncertainty/discussion/133613)


Hi, just giving what I think are 3 powerful approaches to predicting uncertainty; feel free to try them out.

In the Second Annual Data Science Bowl (https://www.kaggle.com/c/second-annual-data-science-bowl/overview), 
the winners looked at their holdout validation set to build a linear regression of the form stdev = a*prediction + b. 
They then fit their predictions as a Gaussian with the mean centered at their prediction. 
Reading the solutions to this competition will be helpful in figuring out how to predict uncertainty. 
Similarly, you can read the solutions to NFL Big Data Bowl (https://www.kaggle.com/c/nfl-big-data-bowl-2020); 
however, most people fit their regressions to a CDF directly, 
which I think will be hard to do without tricks in this competition, because there is no maximum value to the sale limit.


Most people will have multiple models in the form of bagging (running the same model multiple times) or 
in the form of ensemble (e.g. using SVM, LGBM, Neural Network combined). 
To get an uncertainty prediction, you can look at the spread between predictions between the models. 
This will give you a measure of uncertainty; however, I think it is more biased.


If you are using Deep Learning, there are frameworks available for you. 
There is Tensorflow Probability (https://www.tensorflow.org/probability) 
which allows you to build deep probabilistic models without changing much of anything. 
I have found this very tricky to use, but since it's pretty easy to change TF code to TFP, you can try it quickly. 
Uber has made a Pytorch equivalent called Pyro (https://pyro.ai/) which I have not used, 
but looks promising as well. For gradient boosted decision trees, 
there is NGBoost from Stanford (https://stanfordmlgroup.github.io/projects/ngboost/)

If I were Walmart, I would be more interested in the top solution from the Uncertainty competition 
than the forecasting competition, because the top solutions in Uncertainty seem like they would be more reliable in the long run. 

Good luck to all
