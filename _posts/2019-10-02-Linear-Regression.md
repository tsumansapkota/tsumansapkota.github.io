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

Here, we assume that there is one input variable (say x) and one output variable (say y). We learn a function that maps input to the output. This is called as function approximation, i.e. we assume that the data points $$(x_{i}, y_{i})$$ are drawn from some unknown function $$y_{i} = g(x_{i})$$ and we estimate the function $$g(x)$$ by $$f(x)$$. There may be noise in real world data, that is why I have made the points slightly scattered. We use the function $$y=f(x)$$ to predict the output (y) of some unseen input data (x).

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


### Linear Regression using Linear Algebra

We can formulate the [system of linear equations](https://en.wikipedia.org/wiki/System_of_linear_equations#Matrix_equation) in terms of matrices (and/or vectors). Equation of line or plane can be represented as below,

$$\textbf{Y} = \textbf{X . W} \tag{6}$$ 

Here, $$ \textbf{X} \in \mathbb{R}^{m\times (n+1)} ;\qquad \textbf{W} \in \mathbb{R}^{(n+1)\times p} ; \qquad\textbf{Y} \in \mathbb{R}^{m\times p} $$   
Where, **m** is the number of data points,   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**n** is the dimension of the input data,   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**p** is the dimension of the output data.

##### What does the matrices look like ?

The Matrix **X** contains all the input data points, **Y** contains all the output data points and **W** contains all the learnable parameters of the Linear Regression.

$$\textbf{X}=
\begin{bmatrix}
x_{0,0}     &x_{0,1}    &\cdots   &x_{0,n-1}    &1    \\
x_{1,0}     &x_{1,1}    &\cdots   &x_{1,n-1}    &1    \\
\vdots      &\vdots     &\ddots   &\vdots       &\vdots \\
x_{m-1,0}   &x_{m-1,1}  &\cdots   &x_{m-1,n-1}  &1
\end{bmatrix}
$$

Here, each row is the vector input data with a constant 1. Each column represents data in that axis/dimension of the input dataset.

$$\textbf{W}=
\begin{bmatrix}
w_{0,0}     &w_{0,1}    &\cdots   &w_{0,p-1}    \\
w_{1,0}     &w_{1,1}    &\cdots   &w_{1,p-1}    \\
\vdots      &\vdots     &\ddots   &\vdots       \\
w_{n,0}     &w_{n,1}    &\cdots   &w_{n,p-1}  
\end{bmatrix}
$$

Here, each row corresponds to the slope of each input dimension and each column represents the slope required to compute each dimension of the output given input. This follows, $$\textbf{W}_{i,j}$$ gives us the slope/weight of the i<sup>th</sup> input to the j<sup>th</sup> output.

$$\textbf{Y}=
\begin{bmatrix}
y_{0,0}     &y_{0,1}    &\cdots   &y_{0,p-1}    \\
y_{1,0}     &y_{1,1}    &\cdots   &y_{1,p-1}    \\
\vdots      &\vdots     &\ddots   &\vdots       \\
y_{m-1,0}     &y_{m-1,1}    &\cdots   &y_{m-1,p-1}  
\end{bmatrix}
$$

Here, each row represents each output given each input and each column represents the output of individual dimension of the output. This follows, $$\textbf{Y}_{i,j}$$ gives us the output of the i<sup>th</sup> input in the j<sup>th</sup> dimension of the output.

This looks complicated, but it is the easiest to work with. We must be careful with the dimensions of the matrix and what each elements represent in the matrix. This makes our problem highly organized and easy to solve.

##### What does the matrix look like for our 2D data ?

This is how the matrices look like for our previously used 2D dataset.

$$\textbf{X}=
\begin{bmatrix}
x_{0}    &1    \\
x_{1}    &1    \\
\vdots     &\vdots \\
x_{m-1}  &1
\end{bmatrix}
$$&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
$$\textbf{W}=
\begin{bmatrix}
w_{0}    \\
w_{1}
\end{bmatrix}
$$&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
$$\textbf{Y}=
\begin{bmatrix}
y_{0}   \\
y_{1}   \\
\vdots  \\
y_{m-1}  
\end{bmatrix}
$$

Here, $$ \textbf{X} \in \mathbb{R}^{200\times 2} ;\qquad \textbf{W} \in \mathbb{R}^{2\times 1} ;\qquad \textbf{Y} \in \mathbb{R}^{200\times 1} $$   
We have one dimensional input, hence we have two columns (one for 1D input and other for constant of 1) and with 200 data elements, we have 200 rows.
Also we have 2 elements in the weights corresponding to $$m$$ and $$c$$ of equation of line. Finally we have 200 rows corresponding to output of 200 input data and 1 column corresponding to the value of $$y$$. Hence, we have defined the equation of line $$y=mx+c$$ in different form.


##### How do we find the matrix **W** ?

This can be solved using simple Linear Algebra. Lets get back into the equation and solve for **W**.

$$
\begin{align*}
  \textbf{Y} &= \textbf{X} . \textbf{W} \\
  \textbf{X}^{+} . \textbf{Y} &= \textbf{X}^{+} . \textbf{X} . \textbf{W} \\
  \textbf{X}^{+} . \textbf{Y} &= \textbf{I} . \textbf{W} \\
  \textbf{W} &= \textbf{X}^{+} . \textbf{Y} \tag{7}
\end{align*}
$$

Here, $$\textbf{X}^{+}$$ is the [Moore–Penrose pseudoinverse](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse) of matrix $$\textbf{X}$$ and $$\textbf{I}$$ is the identity matrix.

We can easily find the parameters that best fit the line using the equation (7). Lets try this way of finding the parameters on our previous dataset. We get,

$$\textbf{W}=
\begin{bmatrix}
0.3376    \\
0.7393
\end{bmatrix}
$$

These are same values we got by using Least Squares Method (m = 0.3376 & c = 0.7393). This method is easier to understand but it used different technique to solve for the same parameteres.

##### How do we predict Output for new Input ?

Let us consider an unseen input we previously used, i.e. x=0.1. We first construct a matrix with this input of length 1. The input in matrix form is:

$$\textbf{X}=
\begin{bmatrix}
0.1    &1    \\
\end{bmatrix}
$$

Now finding the output using linear equation (6).

$$
\begin{align*}
  \textbf{Y} &= \textbf{X} . \textbf{W} \\
  &= 
  \begin{bmatrix}
  0.1    &1    
  \end{bmatrix} 
  .
  \begin{bmatrix}
  0.3376    \\
  0.7393
  \end{bmatrix} \\
  &=
  \begin{bmatrix}
   0.77306    
  \end{bmatrix} 
\end{align*}
$$

This is also the same as what we predicted using equation of line. We don't even need to plot the output. Its exactly the same thing, but using different methods.

**What happens when the input is multi-dimensional and/or the output is multi-dimensional ?**  
The method is still the same for finding the **W** matrix as well as predicting on new examples. Only the input matrix and output matrix will be changed. The size of the matrices are directly determined by the dimensions of input and output variables.


### Linear Regression using Gradient Descent

We can find the parameters of the line that best fits the data using recursve search algorithm called as gradient descent. Gradient descent is used for all searching parameters in various type of machine learning algorithms including Linear Regression, SVM and Neural Network.

The objective of this algorithm is the same as that of Least Squares Regression, i.e. to find the parameters that minimizes the error. In least squares method, we solved for the parameters using analytical method. Here, we find the parameters using numerical computation and search based method.



