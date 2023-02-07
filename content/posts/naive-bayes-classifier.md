---
title: "Naive Bayes Classifier"
date: 2018-01-01T00:00:01-08:00
mathjax: true
draft: true
---

# Naive Bayes Classifier

## maths and stuff

for classification we want to map inputs to a discrete output

$$ f : X \rightarrow Y $$

the probability of an output can be represented conditionally as

$$ P ( Y | X ) $$

and among various output classes, the most likely class is

$$ \arg\max_ y P ( Y | X )  $$

using Bayes rule this transforms to

$$P ( Y | X ) =  { P ( Y \cap X )  \over P ( X ) } = { P ( X | Y )  \cdot P ( Y ) \over P ( X ) }$$

because the conditional probability for each class $P(Y_{c}|X)$ is proportional to the marginal $P( X )$, the marginal can be factored out

$$ P ( Y | X ) \propto { P ( X | Y )  \cdot P ( Y ) } $$

which gives the map

$$ f ( X ) = \arg\max_y { P ( X | Y )  \cdot P ( Y ) }  $$

where $P ( X | Y )$ are the class likelihoods (i.e. probability of some data given the class distribution) and $P ( Y )$ are class priors (i.e. empirical class distributions).

Class priors are just the normalized prevalence of each class in the training set (maximum likelihood estimate).

**MLE / mean equation**

Class likelihoods require knowing the probability distribution for each class.  Assuming all distributions are Gaussian, we can use the probability density function to find the probability of some data belonging to a class.

For a single / univariate feature 

**show univariate PDF**

For multiple / multivariate features

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
