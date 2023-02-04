---
title: "Naive Bayes Classifier"
date: 2018-01-01T00:00:01-08:00
mathjax: true
draft: true
---


initial brain dump...

I love digging into the math and statistics behind machine learning algorithms.  Understanding how something works is not only empowering but that knowledge can help inform better decisions.  With that goal in mind, the following is a dive into the naive Bayes classifier.

For the task of classification, we want to find a function that maps inputs to a discrete output

$$f(x) \rightarrow y$$

using conditional probability that can represented as the probability of the output class given the input

$$P(y | x)$$

for multiclass classification we want the most likely class given the data

$$f(x) = \arg\max_y P(y | x)  $$

using Bayes rule

$$P(y|x) =  {P( y \cap x)  \over P(x)} = {P(x|y)  \cdot P(y) \over P(x)}$$

which gives

$$f(x) = \arg\max_y {P(x|y)  \cdot P(y) \over P(x)}  $$

and the marginal probability is constant

$$f(x) = \arg\max_y {P(x|y)  \cdot P(y)}  $$

where $P(x|y)$ are class likelihoods and $P(y)$ are class priors.

The class priors are just the empirical distribution of  $y$ across all classes.  The class likelihoods are slightly more complex and involve using the probability density function of the normal (or multivariate normal) distribution which is another naive assumption about data distribution.

**show univariate and multivariate normal PDFs**


$$
f(X) = \frac{1}{(2\pi)^{\frac{n}{2} |\Sigma|^{\frac{1}{2}}}} e^{ - \frac{1}{2} (X - \mu)^T \Sigma^{-1} (X - \mu)}
$$




## show code snippets periodically

```python
# test python (sample from offlineimap)
 
class ExitNotifyThread(Thread):
    """This class is designed to alert a "monitor" to the fact that a thread has
    exited and to provide for the ability for it to find out why."""
    def run(self):
        pass

    def wun(self):
        for i in range(10):
        pass
   
```
