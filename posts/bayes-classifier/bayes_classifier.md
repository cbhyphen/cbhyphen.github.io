---
title: "Bayes Classifier"
date: 2018-01-01T00:00:01-08:00
mathjax: true
draft: false
---


Even though I've studied (and revisited, and revisited..) Bayesian statistics several times over the years, I always felt that, over time, my understanding would lose it's sharpness.  In my opinion, the Bayesian paradigm isn't very intuitive.  So I created this post as future reference to myself, but also as a way to dive deeper into things like the naive assumption, maximum a posterior vs maximum likelihood, and decision boundaries.

## maximum a posteriori

Let's assume there is a random variable $X$ that follows a Gaussian distribution

$$
X \sim N(\mu, \sigma)
$$

and a variable $Y$ which is discrete

$$
Y\in\{0, 1\}
$$

Suppose we know that the value of $Y$ is dependent upon $X$, but that the relationship is not deterministic.  We can model this relationship using conditional probability

$$
P(Y=y|X=x)
$$

But say we want to assign $Y$ a definitive value (i.e., classify).  In that case we can simply select the value of $Y$ with the highest probability

$$
\arg\max_ y P(Y|X)
$$

And because we are selecting a value for $Y$ when there is uncertainty, this means we are making an estimate.  The above is known as the maximum a posteriori (MAP) estimate of $Y$ given $X$, and $P(Y|X)$ is commonly referred to as the posterior.

Most likely we won't have knowledge of the posterior ($Y$ is unknown afterall), so we use Bayes theorem to derive an equivalence

$$
P(Y|X) = {P(Y \cap X) \over P(X)} = {P(X|Y) \cdot P(Y) \over P(X)}
$$

where 

- $P(X|Y)$ is the likelhood  (i.e., probability of the data given the class)
- $P(Y)$ is the prior (i.e., probability of the class)
- $P(X)$ is the marginal (i.e., probability of the data)

When performing the MAP estimate, we are given some value of $X$ and then calculate the posterior probability for each possible value of $Y$.  This means that the marginal is the same for all values of $Y$ and is just a constant that can be factored out

$$
P(Y|X) \propto {P(X|Y) \cdot P(Y)}
$$

which simplifies the MAP classifier to

$$
\arg\max_y {P(X|Y)  \cdot P(Y)}
$$

As far as the likelihood function, we made an assumption on the distribution of $X$ so we can use the Gaussian probability density function

$$
p(x|y) = \frac{1}{\sigma_y\sqrt2\pi} e ^ {- \frac{1}{2} ( \frac{x - \mu_y}{\sigma_y} ) ^2}
$$

If we don't know the Gaussian parameters above, we just estimate them using the empirical mean and variance of the training data for each class which is a maximum likelihood estimate.

$$
\mu_y = \frac{1}{n}\sum_{i}^{n}x_i
$$

$$
\sigma_y^2 = \frac{1}{n}\sum_{i}^{n}(x_i - \mu_y)^2
$$

We don't know the distribution of the prior, so we have to estimate it.  In practice, we simply use the prevalence of each class in the training data which is again a maximum likelihood estimate.

$$
p(y) = \frac{1}{n}\sum_{i}^{n} \mathbb{1}(y_i = y)
$$

It's worth noting that there is also a maximum likelihood estimate (MLE) that could be used for the classifier.  As the name suggest we would just use the likelihood term and remove the prior 

$$
\arg\max_y {P(X|Y)}
$$

but this ignores the prior distribution if we have that information.


With some basic theory out of the way, let's build a classifer.

## univariate classifier

Let's simulate univariate Gaussian data for the two classes.  For simplicity, the data will have different means but the same variance.


```python
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

sns.set()
%matplotlib inline
```


```python
n = 1000
mu_1 = 40
mu_2 = 80
std = 10

x_1 = np.random.normal(loc=mu_1, scale=std, size=n)
x_2 = np.random.normal(loc=mu_2, scale=std, size=n)

df = pd.DataFrame({'x': np.concatenate([x_1, x_2]), 'y': [1] * n + [2] * n})

sns.displot(df, kind='kde', x='x', hue='y',
            fill=True, linewidth=0, palette='dark', alpha=0.5)
```




    <seaborn.axisgrid.FacetGrid at 0x1e229bd5b08>




    
![png](/output_4_1.png)
    


Time to estimate priors, means, and standard deviations. This is trivial since we generated the data but let's pretend that we didn't :)


```python
priors = {k: df[df.y == k].size / df.size for k in df.y.unique()}
priors
```




    {1: 0.5, 2: 0.5}




```python
means = {k: df[df.y == k].x.mean() for k in df.y.unique()}
means
```




    {1: 39.74917237733038, 2: 80.16187136171098}




```python
stdevs = {k: df[df.y == k].x.std() for k in df.y.unique()}
# .std(ddof=0) if not sample
stdevs
```




    {1: 10.004638113965834, 2: 10.265729149219876}



Now that the data is fit, we can build a classifier and predict new instances.


```python
# scipy actually has gaussian pdf: from scipy.stats import norm

def uni_gaussian_pdf(x, mean, stdev):
    scalar = 1.0 / (stdev * np.sqrt(2 * np.pi))
    exponential = np.exp(-0.5 * ((x - mean) / stdev) ** 2)
    return scalar * exponential

classes = df.y.unique().tolist()  

def gbayes_uni_classify(x):
    probas = []
    for c in classes:
        likelihood = uni_gaussian_pdf(x, means[c], stdevs[c])
        # likelihood = norm.pdf(x, means[c], stdevs[c])
        probas.append(likelihood * priors[c])
    return classes[np.argmax(probas)]
```

It's important to mention here that the priors are the same since we generated equal amounts of data for both classes.  Mathematically this means that the prior is a constant and can be factored out in the original MAP equation (for this case) giving

$$
\arg\max_y {P(X|Y)}
$$

So in the case where priors are the same, the MAP is equivalent to the MLE.

And now to visualize the decision boundary.


```python
sim_data = np.arange(0, 150, 1)  # uniform sequence
sim_class_preds = [gbayes_uni_classify(x) for x in sim_data]

decision_boundary = np.where(np.array(sim_class_preds[:-1]) - np.array(sim_class_preds[1:]) != 0)[0]
print(decision_boundary)

sns.displot(df, kind='kde', x='x', hue='y', fill=True, linewidth=0, palette='dark', alpha=0.5)
for v in sim_data[decision_boundary]:
    plt.axvline(v, color='black', linestyle='--')
```

    [59]
    


    
![png](output_12_1.png)
    


The decision boundary is roughly halfway between means, as expected.  This isn't super interesting but what if the variances are different?


```python
n = 1000
mu_1 = 40
mu_2 = 80
std_1 = 20
std_2 = 10

x_1 = np.random.normal(loc=mu_1, scale=std_1, size=n)
x_2 = np.random.normal(loc=mu_2, scale=std_2, size=n)

df = pd.DataFrame({'x': np.concatenate([x_1, x_2]), 'y': [1] * n + [2] * n})

sns.displot(df, kind='kde', x='x', hue='y', fill=True, linewidth=0, palette='dark', alpha=0.5)
```




    <seaborn.axisgrid.FacetGrid at 0x1e22bf7f188>




    
![png](output_14_1.png)
    



```python
# class so we don't repeat same spiel

class GBUniClf:
    
    def __init__(self):
        self.classes = None
        self.priors = None
        self.means = None
        self.stdevs = None
        
    def fit(self, df):
        self.classes = df.y.unique().tolist()
        self.priors = {k: df[df.y == k].size / df.size for k in self.classes}
        self.means = {k: df[df.y == k].x.mean() for k in self.classes}
        self.stdevs = {k: df[df.y == k].x.std() for k in self.classes}
        
    def likelihood(self, x, mean, stdev):
        scalar = 1.0 / (stdev * np.sqrt(2 * np.pi))
        exponential = np.exp(-0.5 * ((x - mean) / stdev) ** 2)
        return scalar * exponential
    
    def predict(self, x):
        probas = []
        for c in self.classes:
            likelihood = self.likelihood(x, self.means[c], self.stdevs[c])
            probas.append(likelihood * self.priors[c])
        return self.classes[np.argmax(probas)]

```


```python
clf = GBUniClf()
clf.fit(df)

sim_data = np.arange(-50, 200, 1)
sim_class_preds = [clf.predict(x) for x in sim_data]

decision_boundary = np.where(np.array(sim_class_preds[:-1]) - np.array(sim_class_preds[1:]) != 0)[0]
print(decision_boundary)

df_preds = pd.DataFrame({'x': sim_data, 'y': sim_class_preds})

sns.displot(df, kind='kde', x='x', hue='y', fill=True, linewidth=0, palette='dark', alpha=0.3)
rug_plot = sns.rugplot(df_preds, x="x", hue="y", palette='dark', alpha=0.7)
rug_plot.get_legend().remove()
for v in sim_data[decision_boundary]:
    plt.axvline(v, color='black', linestyle='--')
```

    [113 174]
    


    
![png](output_16_1.png)
    


Because class 1 has a larger variance, there is now a second decision boundary.  Instances with high values of $x$ (far right) are less likely to belong to class 2 even though they are closer to its' mean.  Instead they get classified as 1.

What if the priors are different?


```python
n_1 = 2000
n_2 = 500
mu_1 = 40
mu_2 = 80
std_1 = 20
std_2 = 10

x_1 = np.random.normal(loc=mu_1, scale=std_1, size=n_1)
x_2 = np.random.normal(loc=mu_2, scale=std_2, size=n_2)

df = pd.DataFrame({'x': np.concatenate([x_1, x_2]), 'y': [1] * n_1 + [2] * n_2})

clf = GBUniClf()
clf.fit(df)

sim_data = np.arange(-50, 200, 1)
sim_class_preds = [clf.predict(x) for x in sim_data]

decision_boundary = np.where(np.array(sim_class_preds[:-1]) - np.array(sim_class_preds[1:]) != 0)[0]
print(decision_boundary)

df_preds = pd.DataFrame({'x': sim_data, 'y': sim_class_preds})

sns.displot(df, kind='kde', x='x', hue='y', fill=True, linewidth=0, palette='dark', alpha=0.3)
rug_plot = sns.rugplot(df_preds, x="x", hue="y", palette='dark', alpha=0.7)
rug_plot.get_legend().remove()
for v in sim_data[decision_boundary]:
    plt.axvline(v, color='black', linestyle='--')
```

    [120 164]
    


    
![png](output_18_1.png)
    


It simply makes the more prevalent class more likely as expected.

### multivariate

Now we can look at bivariate data where covariance between features come into play.  The naive assumption ignores covariance so we can compare classifiers that do and do not make that assumption.

Mathematically, the posterior is now conditioned on multiple features

$$
P(Y|X) = P(Y|x_1, x_2, \ldots , x_i)
$$

and the MAP classifier in the multivariate case is

$$
\arg\max_y {P(Y) \cdot P(x_1, x_2, \ldots , x_i|Y)}
$$

Therefore we use the multivariate likelihood function which makes use of covariance

$$
p(x|y) = \frac{1}{\sqrt{(2\pi)^n |\Sigma_y|}} e^{ - \frac{1}{2} (x - \mu_y)^T \Sigma_y^{-1} (x - \mu_y)}
$$

This is a drop-in replacement though, and the rest of the classifier is the same.


```python
n = 1000
mu_1 = 40
mu_2 = 80
std = 10

x_1 = np.random.normal(loc=mu_1, scale=std, size=(n, 2))
x_2 = np.random.normal(loc=mu_2, scale=std, size=(n, 2))
data = np.concatenate([x_1, x_2])

df = pd.DataFrame({'x1': data[:, 0], 'x2': data[:, 1], 'y': [1] * n + [2] * n})

# s = sns.scatterplot(df, x='x1', y='x2', hue='y', hue_order=classes, palette='dark', alpha=0.25)
s = sns.kdeplot(df, x='x1', y="x2", hue="y", palette='dark', fill=True, alpha=1)
sns.move_legend(s, "upper left", bbox_to_anchor=(1, 1))
```


    
![png](output_21_0.png)
    



```python
class GBBiClf:
    
    def __init__(self):
        self.classes = None
        self.priors = None
        self.means = None
        self.covars = None
        self.covar_dets = None
        self.covar_invs = None

    def fit(self, df):
        self.classes = df.y.unique().tolist()
        self.priors = {k: df[df.y == k].shape[0] / df.shape[0] for k in self.classes}
        self.means = {k: df[['x1', 'x2']][df.y == k].mean(axis=0) for k in self.classes}
        self.covars = {k: np.cov(df[['x1', 'x2']][df.y == k], rowvar=False, bias=True) for k in self.classes}
        self.covar_dets = {k: np.linalg.det(self.covars[k]) for k in self.classes}
        self.covar_invs = {k: np.linalg.inv(self.covars[k]) for k in self.classes}

    def likelihood(self, x, c):
        dims = 2
        scalar = 1.0 / np.sqrt(((2 * np.pi) ** dims) * self.covar_dets[c])
        exponential = np.exp(-0.5 * (x - self.means[c]).T @ self.covar_invs[c] @ (x - self.means[c]))
        return scalar * exponential

    def predict(self, x):
        probas = []
        for c in self.classes:
            likelihood = self.likelihood(x, c)
            probas.append(likelihood * self.priors[c])
        return self.classes[np.argmax(probas)]

```


```python
clf = GBBiClf()
clf.fit(df)

sim_data_range = np.arange(0, 140, 1)
sim_data = np.array([np.array([x1, x2]) for x1 in sim_data_range for x2 in sim_data_range])
sim_classes = [clf.predict(x) for x in sim_data]

plot_df = pd.DataFrame(np.hstack([sim_data, np.array(sim_classes).reshape(-1, 1)]), columns=['x1', 'x2', 'y'])

# sns.scatterplot(plot_df, x='x1', y="x2", hue="y", hue_order=classes, palette='dark', marker=".", alpha=0.15)
plt_points = sns.relplot(plot_df, x='x1', y='x2', hue='y', hue_order=clf.classes, palette='dark', marker=".", alpha=0.15)
plt_points._legend.remove()
s = sns.kdeplot(df, x='x1', y="x2", hue="y", palette='dark', fill=True, alpha=1)
sns.move_legend(s, "upper left", bbox_to_anchor=(1, 1))
```


    
![png](output_23_0.png)
    


The variance was same for both distributions and the features were sampled independently, so the decision boundary isn't complex.  Slight curvature is due to the estimate of covariance which is different from the true value.


```python
clf.covars
```




    {1: array([[97.13589025,  3.99602431],
            [ 3.99602431, 99.21513485]]),
     2: array([[104.87159824,  -2.09309889],
            [ -2.09309889, 102.99262222]])}



Even though this data is uninteresting, let's compare the decision boundary of a **naive** classifier.  The naive assumption is that all features are independent, so we can use the chain rule of probability for a simpler calculation of the likelihood.

$$
P(X|Y) = P(x_1, x_2, \ldots , x_i|Y) = \prod\limits_{i}P(x_i|Y)
$$

The MAP classifier under the naive assumption then becomes

$$
\arg\max_y {P(Y) \cdot P(x_1|Y) \cdot P(x_2|Y) \cdot \ldots \cdot P(x_m|Y)}
$$

For this case though, since the features were generated independently, the decision boundary should be roughly the same.


```python
class GNBBiClf:
    
    def __init__(self):
        self.classes = None
        self.priors = None
        self.means = None

    def fit(self, df):
        self.classes = df.y.unique().tolist()
        self.priors = {k: df[df.y == k].shape[0] / df.shape[0] for k in self.classes}
        self.means = {k: df[['x1', 'x2']][df.y == k].mean(axis=0) for k in self.classes}
        self.stdevs = {k: np.std(df[['x1', 'x2']][df.y == k], axis=0) for k in self.classes}

    def likelihood(self, x, mean, stdev):
        scalar = 1.0 / (stdev * np.sqrt(2 * np.pi))
        exponential = np.exp(-0.5 * ((x - mean) / stdev) ** 2)
        return scalar * exponential
    
    def predict(self, x):
        probas = []
        for c in self.classes:
            joint_likelihood = 1
            for i, v in enumerate(x):
                likelihood = self.likelihood(v, self.means[c][i], self.stdevs[c][i])
                joint_likelihood *= likelihood
            probas.append(joint_likelihood * self.priors[c])
        return self.classes[np.argmax(probas)]

```


```python
clf = GNBBiClf()
clf.fit(df)

sim_data_range = np.arange(0, 140, 1)
sim_data = np.array([np.array([x1, x2]) for x1 in sim_data_range for x2 in sim_data_range])
sim_classes = [clf.predict(x) for x in sim_data]

plot_df = pd.DataFrame(np.hstack([sim_data, np.array(sim_classes).reshape(-1, 1)]), columns=['x1', 'x2', 'y'])

plt_points = sns.relplot(plot_df, x='x1', y='x2', hue='y', hue_order=clf.classes, palette='dark', marker=".", alpha=0.15)
plt_points._legend.remove()
s = sns.kdeplot(df, x='x1', y="x2", hue="y", palette='dark', fill=True, alpha=1)
sns.move_legend(s, "upper left", bbox_to_anchor=(1, 1))
```


    
![png](output_28_0.png)
    


What if just the covariance are different?  Let's draw random data were the features are still independent (i.e. covariance matrix is symmetic) but the variance of features is different for each class.


```python
n = 1000
mu_1 = 40
mu_2 = 100
std_1 = 20
std_2 = 10

x_1 = np.random.normal(loc=mu_1, scale=std_1, size=(n, 2))
x_2 = np.random.normal(loc=mu_2, scale=std_2, size=(n, 2))
data = np.concatenate([x_1, x_2])

df = pd.DataFrame({'x1': data[:, 0], 'x2': data[:, 1], 'y': [1] * n + [2] * n})

s = sns.kdeplot(df, x='x1', y="x2", hue="y", palette='dark', fill=True, alpha=1)
sns.move_legend(s, "upper left", bbox_to_anchor=(1, 1))
s.set(xlim=(-10, 140), ylim=(-10, 140))
```




    [(-10.0, 140.0), (-10.0, 140.0)]




    
![png](output_30_1.png)
    



```python
clf = GBBiClf()
clf.fit(df)

sim_data_range = np.arange(-10, 140, 1)
sim_data = np.array([np.array([x1, x2]) for x1 in sim_data_range for x2 in sim_data_range])
sim_classes = [clf.predict(x) for x in sim_data]

plot_df = pd.DataFrame(np.hstack([sim_data, np.array(sim_classes).reshape(-1, 1)]), columns=['x1', 'x2', 'y'])

plt_points = sns.relplot(plot_df, x='x1', y='x2', hue='y', hue_order=clf.classes, palette='dark', marker=".", alpha=0.15)
plt_points._legend.remove()
s = sns.kdeplot(df, x='x1', y="x2", hue="y", palette='dark', fill=True, alpha=1)
sns.move_legend(s, "upper left", bbox_to_anchor=(1, 1))
s.set(xlim=(-10, 140), ylim=(-10, 140))
```




    [(-10.0, 140.0), (-10.0, 140.0)]




    
![png](output_31_1.png)
    


As expected, the decision boundary favors class 1 since it had larger variance.  Without any correlation between features, I would again expect the decision boundary to be the same for the naive classifier.


```python
clf = GNBBiClf()
clf.fit(df)

sim_data_range = np.arange(-10, 140, 1)
sim_data = np.array([np.array([x1, x2]) for x1 in sim_data_range for x2 in sim_data_range])
sim_classes = [clf.predict(x) for x in sim_data]

# sanity check
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()
# clf.fit(df[['x1', 'x2']].values, df[['y']].values.ravel())
# sim_data_range = np.arange(-10, 140, 1)
# sim_data = np.array([np.array([x1, x2]) for x1 in sim_data_range for x2 in sim_data_range])
# sim_classes = [clf.predict(x.reshape(1, -1)) for x in sim_data]

plot_df = pd.DataFrame(np.hstack([sim_data, np.array(sim_classes).reshape(-1, 1)]), columns=['x1', 'x2', 'y'])

plt_points = sns.relplot(plot_df, x='x1', y='x2', hue='y', hue_order=clf.classes, palette='dark', marker=".", alpha=0.15)
plt_points._legend.remove()
s = sns.kdeplot(df, x='x1', y="x2", hue="y", palette='dark', fill=True, alpha=1)
sns.move_legend(s, "upper left", bbox_to_anchor=(1, 1))
s.set(xlim=(-10, 140), ylim=(-10, 140))
```




    [(-10.0, 140.0), (-10.0, 140.0)]




    
![png](output_33_1.png)
    


The decision boundary for the naive classifier is roughly identical.  Zooming out, we can see the classifier has similar behavior as the univariate case for different variance.


```python
clf = GNBBiClf()
clf.fit(df)

sim_data_range = np.arange(-10, 200, 1)
sim_data = np.array([np.array([x1, x2]) for x1 in sim_data_range for x2 in sim_data_range])
sim_classes = [clf.predict(x) for x in sim_data]

plot_df = pd.DataFrame(np.hstack([sim_data, np.array(sim_classes).reshape(-1, 1)]), columns=['x1', 'x2', 'y'])

plt_points = sns.relplot(plot_df, x='x1', y='x2', hue='y', hue_order=clf.classes, palette='dark', marker=".", alpha=0.15)
plt_points._legend.remove()
s = sns.kdeplot(df, x='x1', y="x2", hue="y", palette='dark', fill=True, alpha=1)
sns.move_legend(s, "upper left", bbox_to_anchor=(1, 1))
s.set(xlim=(-10, 200), ylim=(-10, 200))
```




    [(-10.0, 200.0), (-10.0, 200.0)]




    
![png](output_35_1.png)
    


Let's finally simulate data with correlation between features.  There should be a noticeable difference in the decision boundary for the naive classifier.


```python
n = 1000
mu_1 = 40
mu_2 = 100

x_1 = np.random.multivariate_normal(mean=[mu_1, mu_1], cov=[[50, 70], [70, 200]], size=n)
x_2 = np.random.multivariate_normal(mean=[mu_2, mu_2], cov=[[100, 1], [1, 100]], size=n)  # no correlation
data = np.concatenate([x_1, x_2])

df = pd.DataFrame({'x1': data[:, 0], 'x2': data[:, 1], 'y': [1] * n + [2] * n})

s = sns.kdeplot(df, x='x1', y="x2", hue="y", palette='dark', fill=True, alpha=1)
sns.move_legend(s, "upper left", bbox_to_anchor=(1, 1))
s.set(xlim=(-10, 140), ylim=(-10, 140))
```




    [(-10.0, 140.0), (-10.0, 140.0)]




    
![png](output_37_1.png)
    



```python
clf = GBBiClf()
clf.fit(df)

sim_data_range = np.arange(-10, 140, 1)
sim_data = np.array([np.array([x1, x2]) for x1 in sim_data_range for x2 in sim_data_range])
sim_classes = [clf.predict(x) for x in sim_data]

plot_df = pd.DataFrame(np.hstack([sim_data, np.array(sim_classes).reshape(-1, 1)]), columns=['x1', 'x2', 'y'])

plt_points = sns.relplot(plot_df, x='x1', y='x2', hue='y', hue_order=clf.classes, palette='dark', marker=".", alpha=0.15)
plt_points._legend.remove()
s = sns.kdeplot(df, x='x1', y="x2", hue="y", palette='dark', fill=True, alpha=1)
sns.move_legend(s, "upper left", bbox_to_anchor=(1, 1))
s.set(xlim=(-10, 140), ylim=(-10, 140))
```




    [(-10.0, 140.0), (-10.0, 140.0)]




    
![png](output_38_1.png)
    



```python
clf = GNBBiClf()
clf.fit(df)

sim_data_range = np.arange(-10, 140, 1)
sim_data = np.array([np.array([x1, x2]) for x1 in sim_data_range for x2 in sim_data_range])
sim_classes = [clf.predict(x) for x in sim_data]

plot_df = pd.DataFrame(np.hstack([sim_data, np.array(sim_classes).reshape(-1, 1)]), columns=['x1', 'x2', 'y'])

plt_points = sns.relplot(plot_df, x='x1', y='x2', hue='y', hue_order=clf.classes, palette='dark', marker=".", alpha=0.15)
plt_points._legend.remove()
s = sns.kdeplot(df, x='x1', y="x2", hue="y", palette='dark', fill=True, alpha=1)
sns.move_legend(s, "upper left", bbox_to_anchor=(1, 1))
s.set(xlim=(-10, 140), ylim=(-10, 140))
```




    [(-10.0, 140.0), (-10.0, 140.0)]




    
![png](output_39_1.png)
    


The difference between the naive and non-naive classifier is more noticeable when there is correlation between features (class 1).  The naive classifier clearly ignores the covariance and the decision boundary is much smoother.

One obvious advantage of the naive assumption is computational efficiency.  Predictions for the naive classifier ran in faster time compared to the non-naive by an order of magnitude.  Fit times were roughly the same.


```python
from time import perf_counter

# TODO memory reqs

m = 100

start_time = perf_counter()
for i in range(m):
    clf = GBBiClf()
    clf.fit(df)
print(f"GB avg fit time: {(perf_counter() - start_time) / m:.6f}")
start_time = perf_counter()
for i in range(m):
    clf.predict(sim_data[i])
print(f"GB avg predict time: {(perf_counter() - start_time) / m:.6f}")

start_time = perf_counter()
for i in range(m):
    clf = GNBBiClf()
    clf.fit(df)
print(f"GNB avg fit time: {(perf_counter() - start_time) / m:.6f}")
start_time = perf_counter()
for i in range(m):
    clf.predict(sim_data[i])
print(f"GNB avg predict time: {(perf_counter() - start_time) / m:.6f}")
```

    GB avg fit time: 0.004047
    GB avg predict time: 0.000654
    GNB avg fit time: 0.005097
    GNB avg predict time: 0.000047
    


