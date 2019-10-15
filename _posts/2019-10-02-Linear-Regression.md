---
title: Exploring Linear Regression
categories:
- Algorithm
excerpt: |
  Linear Regression is the simplest Machine Learning Algorithm and is fundamental to other algorithms such as Polynomial Regression and Neural Networks. Here, we explore the questions (What/Why/Where/How) of Linear Regression.
feature_text: | 
  ## Linear Regression 
  The fundamental Machine Learning Algorithm.
feature_image: "https://picsum.photos/2560/600?image=733"
image: "https://picsum.photos/2560/600?image=733"
---

<!-- Linear Regression is the simplest Machine Learning Algorithm and is fundamental to other algorithms such as Polynomial Regression and Neural Networks. Linear regression finds the best line to fit the 2D data. In higher dimensions, Linear Regression finds the best nD Plane to fit the data. [Linear Regression](https://en.wikipedia.org/wiki/Linear_regression)


For simplicity, we will only look at 2 dimensional data. Lets start with a data to work with.
{% include figure.html image="/assets/post_images/linear-regression/data_set.svg" position="center" height="400" caption="Fig: X vs Y plot of dataset" %}

Our aim is to find a line that represents the given data.
The line is in the form $$y=mx+c$$.  Where,  
**m** is the slope of the line,  
**c** is the y-intercept,  
**x** is the input to the linear function and  
**y** is the output. 

<!-- --- -->
<!-- ##### What does linear regression do ?
Linear Regression gives us a method to find the parameters of the line, i.e. **m** and **c**.
Using Least Squares method we can derive the formula to find the value of parameters given the dataset.

**Least Squares**
The error of the prediction -->


Linear Regression is the simplest Machine Learning Algorithm and is fundamental to other algorithms such as Polynomial Regression, Logistic Regression and Neural Networks.

##### What does Linear Regression do ?
Lets consider 2D data on xy plane (x-axis and y-axis). Linear Regression finds the line that best fits the data. The line is in the form of function $$y=f(x)=mx+c$$. Linear Regression is similar for nD (n-Dimensional) data as well, but we will focus on 2D data for now.

##### How to do Linear Regression ?
Lets get with the simplest case of 2D dataset. I have generated a highly linear dataset (n=200) to work with.

{% include figure.html image="/assets/post_images/linear-regression/data_set.svg" position="center" height="400" caption="Fig: x vs y plot of dataset" %}

We want a line to best fit the data. **How do we define that mathematically ?**

We would like the difference between the predicted y, (ŷ), and the true value y, given some x, to be as minimum as possible. To summarize, We would like the error of the prediction to be minimum. 

[Generally, we minimize the square of the the error. This is why linear regression is also called least squares approximation. Also, minimizing squared error and minimizing mean squared error have same objective]

The Mean Squared Error of the prediction is given as,

$$E=\frac{1}{n}\sum_{i=0}^{n-1} (\hat{y}_{i} - y_{i})^2$$

$$E(m,c)=\frac{1}{n}\sum_{i=0}^{n-1} (mx_{i} + c - y_{i})^2$$

Here, $$E(m,c)$$ is an error function with input $$m$$ and $$c$$. We would like to find the value of $$m$$ and $$c$$, that minimizers the error value E. We can do this using little calculus and algebra.

The values of $$m$$ and $$c$$ that minimizes the error function $$E(m,c)$$ can be written as,

$$(m^*, c^*) = arg\min_{m, c} E(m,c)$$

This means that, $$m^*$$ and $$c^*$$  are the values that minimizes the Error.

##### How do we derrive the values of parameters ?

We know that the derivative of a function is zero at its minimum. Minimizing the Error function w.r.t. the slope $$m$$,

$$\frac{dE}{dm} = \frac{dE(m,c)}{dm} = 0 \tag{1}$$

Simplifying $$E(m,c)$$,  
$$
\begin{align*}
  E(m,c)&=\frac{1}{n}\sum_{i=0}^{n-1} (mx_{i} + c - y_{i})^2 \\
  &=\frac{1}{n}\sum_{i=0}^{n-1} (m^2x_{i}^2 + 2(mx_{i})(c - y_{i}) + (c - y_{i})^2) \\
  &=\frac{1}{n}\sum_{i=0}^{n-1} (m^2x_{i}^2 + 2mx_{i}c - 2mx_{i}y_{i} - 2cy_{i} + c^2 + y^2)\\
  E(m,c)&=\frac{1}{n}\Big[\sum_{i=0}^{n-1} (m^2x_{i}^2 + 2mx_{i}c - 2mx_{i}y_{i} - 2cy_{i} + y^2) + nc^2\Big] \tag{2}
\end{align*}
$$

Substituting the value of E from (2) in (1),  

$$
\begin{align*}
  \frac{dE(m,c)}{dm} &= \sum_{i=0}^{n-1} (2mx_{i}^2 + 2x_{i}c - 2x_{i}y_{i}) \\
  0 &= 2\sum_{i=0}^{n-1} mx_{i}^2 + 2\sum_{i=0}^{n-1} x_{i}c - 2\sum_{i=0}^{n-1} 2x_{i}y_{i} \\
  m &= \frac{\sum_{i=0}^{n-1} x_{i}y_{i} -c\sum_{i=0}^{n-1} x_{i}}{\sum_{i=0}^{n-1} x_{i}^2} \tag{3}

\end{align*}
$$

Similarly, minimizing the Error function w.r.t. the y-intercept $$c$$,

$$\frac{dE}{dc} = \frac{dE(m,c)}{dc} = 0$$

Substituting the value of E from (2),

$$
\begin{align*}
  \frac{dE(m,c)}{dc} &= \sum_{i=0}^{n-1} (2mx_{i}^2 - 2y_{i}) + 2nc \\
  0 &= 2\sum_{i=0}^{n-1} mx_{i} - 2\sum_{i=0}^{n-1} y_{i} + 2nc \\
  c &= \frac{\sum_{i=0}^{n-1} y_{i} - m\sum_{i=0}^{n-1} x_{i}}{n} \tag{4}

\end{align*}
$$

Again, substituting the value of $$c$$ in equation (3),

$$
\begin{align*}
  m &= \frac{\sum_{i=0}^{n-1} x_{i}y_{i} -\Big[\frac{\sum_{i=0}^{n-1} y_{i} - m\sum_{i=0}^{n-1} x_{i}}{n}\Big]\sum_{i=0}^{n-1} x_{i}}{\sum_{i=0}^{n-1} x_{i}^2} \\
  &= \frac{n\sum_{i=0}^{n-1} x_{i}y_{i} -\sum_{i=0}^{n-1} x_{i}y_{i} + m(\sum_{i=0}^{n-1} x_{i})^2}{n\sum_{i=0}^{n-1} x_{i}^2} \\
  m &= \frac{n\sum_{i=0}^{n-1} x_{i}y_{i} -\sum_{i=0}^{n-1} x_{i}y_{i}}{n\sum_{i=0}^{n-1} x_{i}^2 - (\sum_{i=0}^{n-1} x_{i})^2} \tag{5}

\end{align*}
$$

Using the equation (4) and (5), we can find the line parameters $$m$$ and $$c$$, given data points $$(x_{i}, y_{i})$$ for i = 0 ... n-1, to best fit the data.

When applied this formula to our dataset, we get:   
**<center>m = 0.3376, c = 0.7393 and E = 0.04509</center>**

The plot of the line that best fits the data is as follows:

{% include figure.html image="/assets/post_images/linear-regression/regression.svg" position="center" height="400" caption="Fig: x vs y plot of data and predicted line" %}


##### How do we use this Algorithm ?

I was confused on this topic when I was first learning this topic. Why don't we treat x and y equally? What if we find the line $$x=my+c$$? Will it give the same result? What if the data is curvy (non-linear)? These questions helped me understand it better.

Here, we assume that there is one input vaiable (say x) and one output variable (say y). We learn a function that maps input to the output. This is called as function approximation, i.e. we assume that the data points $$(x_{i}, y_{i})$$ are drawn from some unknown function $$y_{i} = g(x_{i})$$ and we estimate the function $$g(x)$$ by $$f(x)$$. There may be noise in real world data, that is why I have made the points slightly scattered. We use the function $$y=f(x)$$ to predict the output (y) of some unseen input data (x).

Let us consider an unseen input (say x=0.1). We can predict what would be the value of y at given x. Using the equation of line $$y = mx + c$$ where, $$m = 0.3376$$ and $$c = 0.7393$$ for above dataset, we get,

$$
\begin{align*}
  y &= mx + c \\
  y &= 0.3376 * 0.1 + 0.7393 \\
  y &= 0.77306
\end{align*}
$$

Plotting the new data (0.1, 0.77306) as predicted by the Linear Regression algorithm.
{% include figure.html image="/assets/post_images/linear-regression/prediction.svg" position="center" height="400" caption="Fig: x vs y plot of data, predicted line and test point" %}

##### What are the limits of Linear Regression?

Linear Regression can only learn straight line. It can only approximate highly linear functions, or else it will have very high error. Even in above example, the data would be fitted better by slightly downward curving line. For curvy function approximation, we will later get into non linear function approximation. But for now, lets further explore the linear regression.


##### Linear Regression using Linear Algebra




