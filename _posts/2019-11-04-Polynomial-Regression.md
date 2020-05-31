---
title: Exploring Polynomial Regression
categories:
- Algorithm
excerpt: |
  Polynomial Regression is the generalization of Linear Regression. It is simple to understand, but can do a lot. It is used to approximate any Non Linear functions, which is almost always better than Linear Regression. Here, we extend the idea of curve fitting, learn its capacity, problems and its limitations. 
feature_text: | 
  ## Polynomial Regression 
  Easy way to fit non linear curve to the data.
feature_image: "/assets/post_images/polynomial-regression/poly_regression_feature.svg"
image: "/assets/post_images/polynomial-regression/poly_regression_feature.svg" #"https://picsum.photos/2560/600?image=733"
---

Polynomial Regression is the generalization of Linear Regression to polynomial function. In Linear Regression, we fit a straight line (i.e. $$y=f(x)=mx+c$$) to our dataset. Here, in Polynomial Regression, we fit a polynomial function of $$x$$ to the data. The polynomial function of $$x$$ is as follows:   
$$y=f(x)=a_{0} + a_{1}x + a_{2}x^2 + \cdots + a_{n}x^n$$   
Where, $$a_{n}$$ is the coeffecient of order n and $$x^n$$ is the $$n^{th}$$ power of input $$x$$.

*This post on Polynomial Regression heavily relies on the concept developed on [Linear Regression Post](/algorithm/2019/10/02/Linear-Regression/).    
Python (Jupyter) Notebook of Polynomial Regression is on this [Github Repository](https://github.com/tsumansapkota/Blog_Post/tree/master/02_Polynomial_Regression). You can also run the Notebook on [Google Collab](https://colab.research.google.com/github/tsumansapkota/Blog_Post/blob/master/02_Polynomial_Regression/00_Polynomial_Regression_v0.ipynb)*


##### Why use Polynomial Regression ?
According to [Stone–Weierstrass theorem](https://en.wikipedia.org/wiki/Stone%E2%80%93Weierstrass_theorem), every continuous function defined on a closed interval [a, b] can be uniformly approximated as closely as desired by a polynomial function. Polynomial function of order n *(highest degree of $$x$$)* < m *(number of data points)* can fit the data approximately (or exactly when $$n = m-1$$). 

Linear Regression fits the $$1^{st}$$ order polynomial (line) of the form $$y=a_{0} + a_{1}x$$ to the data. Using Polynomial Regerssion of higher order, we can fit curve to our data rather than just straight line. Since fitting a strainght line is not always appropriate to our dataset, we use polynomial regression in general.


##### What does Polynomial Function look like ?

Polynomial of degree 0 ,i.e. $$y=a_{0}$$ is a straight line parallel to x-axis. We can see that it is independent to $$x$$. The dynamics of the parallel line with varying coefficient can be seen in the figure below.

{% include figure.html image="/assets/post_images/polynomial-regression/poly_0th_order.gif" position="center" height="400" caption="Fig: 0th order polynomial function" %}


Polynomial of degree 1 i.e $$y=a_{0} + a_{1}x$$ is a line with slope $$a_{1}$$ and y-intercept $$a_{0}$$. The dynamics of line with varying coefficients can be seen in the figure below.

{% include figure.html image="/assets/post_images/polynomial-regression/poly_1st_order.gif" position="center" height="400" caption="Fig: 1st order polynomial function" %}

Polynomial of degree 2 i.e $$y=a_{0} + a_{1}x + + a_{2}x^{2}$$ is a parabolic curve with one maxima/minima. Similarly, here also, we can see the dynamics of curve with varying coefficients as shown in the figure below.

{% include figure.html image="/assets/post_images/polynomial-regression/poly_2nd_order.gif" position="center" height="400" caption="Fig: 2nd order polynomial function" %}

Similarly, Polynomial function of degree $$n$$ i.e. $$y=f(x)=a_{0} + a_{1}x + a_{2}x^2 + \cdots + a_{n}x^n$$ is a curve with $$n-1$$ minima/maxima (wiggles). Let us see what type of functions do higher order polynomial functions represent.

{% include figure.html image="/assets/post_images/polynomial-regression/poly_nth_order.gif" position="center" height="400" caption="Fig: nth order polynomial function" %}

In the above figure we found the polynomial function using roots. Polynomial function can be alternatively represented as: $$y=f(x)=c(x-x_{0})(x-x_{1})(x-x_{2})(x-x_{3})\cdots(x-x_{n-1})$$  
where, $$x_{i}$$ for ***i = 0 to n-1***, are the roots of the polynomial function and c is the constant scaler to the overall function. *(Roots are the values of $$x$$ where the value of the function, $$y=f(x)$$ are zero.)*

Now, Lets get back to our problem of fitting the above curvy data points.

##### How do we find the coefficients of Polynomial Regression ?

The method of finding the coefficients of Polynomial Regression is very similar to [that of Linear Regression](/algorithm/2019/10/02/Linear-Regression/#linear-regression-using-linear-algebra). The problem can be reformulated as multivariate linear regression with input vectors composed of various powers of input $$x$$. A single input $$x_{0}$$ can be represented as a vector*(column matrix)* $$\boldsymbol{x_{0}}$$ consisting of various powers of $$x_{0}$$ as below.

$$\boldsymbol{x_{0}}=
\begin{bmatrix}
1    \\
x_{0} \\
x_{0}^{2}\\
x_{0}^{3}\\
\vdots \\
x_{0}^{n}
\end{bmatrix}$$

Using this form, we can compute the Polynomial function for given input $$x_{0}$$, given a coefficient matrix **A** as follows.

$$
\begin{align*}
  \boldsymbol{y_{0}} &= \boldsymbol{x_{0}}.\boldsymbol{A} \\
  &= a_{0} + a_{1}x + a_{2}x^2 + \cdots + a_{n}x^n
\end{align*}
$$

Where, the coeffecient matrix $$\boldsymbol{A}$$ is as follows:

$$\boldsymbol{A}=
\begin{bmatrix}
a_{0}    \\
a_{1} \\
a_{2}\\
\vdots \\
a_{n}
\end{bmatrix}$$

Since we are finding the polynomial function to fit all the data points, we use all the input points with their various powers as input. We can easily do Linear Regression with multiple variables using Matrices. So, lets write our polynomial function as Matrix Algebra as follows:

$$\boldsymbol{Y} = \boldsymbol{X}.\boldsymbol{A} \tag{1}$$

where,

$$\boldsymbol{X}=
\begin{bmatrix}
\boldsymbol{x_{0}^{T}}\\
\boldsymbol{x_{1}^{T}}\\
\vdots\\
\boldsymbol{x_{n}^{T}}
\end{bmatrix}
=
\begin{bmatrix}
1         &x_{0}      &x_{0}^{2}    &\cdots   &x_{0}^{n}    \\
1         &x_{1}      &x_{1}^{2}    &\cdots   &x_{1}^{n}    \\
\vdots    &\vdots     &\vdots       &\ddots   &\vdots \\
1         &x_{m-1}      &x_{m-1}^{2}    &\cdots   &x_{m-1}^{n}    \\
\end{bmatrix}
$$

Here, we are dealing with 1D input only. The input matrix consists of various powers of each input in each $$i^{th}$$ rows. The $$j^{th}$$ column consists of $$j^{th}$$ power of input data. So, each row can be considered as single input vector and each column as data points of a single feature. With these features, multivariate linear regression is performed. 

$$\boldsymbol{A}=
\begin{bmatrix}
a_{0}    \\
a_{1} \\
a_{2}\\
\vdots \\
a_{n}
\end{bmatrix}

\hspace{1mm} and \hspace{2mm}

\boldsymbol{Y}=
\begin{bmatrix}
y_{0}    \\
y_{1} \\
y_{2}\\
\vdots \\
y_{m-1}
\end{bmatrix}$$

For the coefficient matrix, each column represents coefficients for each output features and each row represent the coefficients for each input features. Since we have used single output feature, we have only one column.

For the output matrix, each row is the output vector/constant of each input vector, the number of columns represent the number of output features. Since we have single output feature, we have one column of output matrix. 

Here, $$ \textbf{X} \in \mathbb{R}^{m\times (n+1)} ;\qquad \textbf{W} \in \mathbb{R}^{(n+1)\times 1} ; \qquad\textbf{Y} \in \mathbb{R}^{m\times 1} $$

The equation (1) can be solved for $$\boldsymbol{A}$$ using little bit of Matrix Algebra, as we did on Linear Regression.

$$
\begin{align*}
  \textbf{Y} &= \textbf{X} . \textbf{A} \\
  \textbf{X}^{+} . \textbf{Y} &= \textbf{X}^{+} . \textbf{X} . \textbf{A} \\
  \textbf{X}^{+} . \textbf{Y} &= \textbf{I} . \textbf{A} \\
  \textbf{A} &= \textbf{X}^{+} . \textbf{Y} \tag{2}
\end{align*}
$$

Here, $$\textbf{X}^{+}$$ is the [Moore–Penrose pseudoinverse](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse) of matrix $$\textbf{X}$$ and $$\textbf{I}$$ is the identity matrix.


Now, we know how to do polynomial regression, lets try fitting polynomial function to our dataset.

##### Second order Polynomial Regression
Let us consider a dataset (same as we used in [Linear Regression](/algorithm/2019/10/02/Linear-Regression/#how-to-do-linear-regression-)). We previously fit a straight line to the data, as shown in the figure below.

{% include figure.html image="/assets/post_images/linear-regression/regression_rs.svg" position="center" height="400" caption="Fig: x vs y plot of data and predicted line" %}

As can be seen in the visualization, we expect the function to be little curvy to exactly approximate the function.

With the above mentioned method, lets first define our matrices $$\boldsymbol{X}$$, $$\boldsymbol{A}$$ and $$\boldsymbol{Y}$$.   
Since we have 200 data points, 3 input features and 1 output features, we have our matrices as follows.

$$\boldsymbol{X}=
\begin{bmatrix}
1         &x_{0}      &x_{0}^{2}\\
1         &x_{1}      &x_{1}^{2}\\
\vdots    &\vdots     &\vdots   \\
1         &x_{199}      &x_{199}^{2}\\
\end{bmatrix}

\hspace{2mm}

\boldsymbol{A}=
\begin{bmatrix}
a_{0}    \\
a_{1} \\
a_{2}
\end{bmatrix}

\hspace{2mm}

\boldsymbol{Y}=
\begin{bmatrix}
y_{0}    \\
y_{1} \\
y_{2}\\
\vdots \\
y_{199}
\end{bmatrix}
$$

Here, $$ \textbf{X} \in \mathbb{R}^{200\times 3} ;\qquad \textbf{A} \in \mathbb{R}^{3\times 1} ;\qquad \textbf{Y} \in \mathbb{R}^{200\times 1} $$   


We can costruct input matrix $$\boldsymbol{X}$$ easily and we have output matrix $$\boldsymbol{Y}$$, We can now calculate the Coefficient Matrix $$\boldsymbol{A}$$ by using the equation (2). Doing so, we get:

$$
\boldsymbol{A}=
\begin{bmatrix}
0.7784    \\
0.3381 \\
-0.4601
\end{bmatrix}
$$

The plot of the second order polynomial (*learned*) that best fits the data, represented by the coefficients, is shown in the figure below:

{% include figure.html image="/assets/post_images/polynomial-regression/poly_2_fit.svg" position="center" height="400" caption="Fig: x vs y plot of data and predicted curve" %}

Well, this looks prettttyyyy good fit. The mean squared error of this fit is $$E=0.000807$$ which is smaller than the linear regression case of $$E=0.002033$$. *(We have a great tool that can approximate any function. Is this all we need for regression/curve fitting ?)*

We have accomplished the goal to fit curve to our dataset. The function learned is curvy, but we don't consider polynomial regression as non-linear regression. This is because, it is the special case of multivariate linear regression, where the inputs are various powers of $$x$$.

##### Higher order Polynomial Regression
We can fit higher order polynomial to our dataset. Since we have 200 data points, we can fit, at highest, 199 order polynomial to our dataset. Let's see how the polynomial regression of various order fit our dataset. Let us see how different order Polynomial Regression on our dataset looks like.

{% include figure.html image="/assets/post_images/polynomial-regression/polys_all_fit.gif" position="center" height="400" caption="Fig: x vs y plot of data and predicted curves" %}

The error plot of various degree of polynomial can be observed as below.

{% include figure.html image="/assets/post_images/polynomial-regression/polys_errors.svg" position="center" height="400" caption="Fig: Degree of polynomial vs Error of the fit" %}


We can see that the higher order polynomial function produces lower and lower error as order increases. The problem is that the curve gets more wiggly as the degree of polynomial increases. This is not what we want. We want simplest curve to fit our data.

***Humm... We were expecting to have zero error for 199 degree polynomial***, *but the curve looks the same for higher order polynomials. We know that, using [Polynomial Interpolation](https://en.wikipedia.org/wiki/Polynomial_interpolation) we can fit every point to the polynomial function. This is just linear regression, so nothing fishy is going here. Maybe the problem is in the data. Lets analyze the data, i.e different powers of $$x$$. Following is a visualization of $$x^{n}$$ for n in (0,1,2,...).*

{% include figure.html image="/assets/post_images/polynomial-regression/data_viz_powers.gif" position="center" height="400" caption="Fig: powers of x vs y" %}

*We can see that the input feature, $$x^{n}$$ tends to zero as the degree of polynomial increases. This is because we centered out data in [-0.5,0.5] range. The magnitude of each value is less than 1, and higher value of n, smaller the feature values. Thus, the data range is so small that even float64 data cant handle accurately storing and representing the data. There are also errors related to computing pseudo-inverses of the matrix. When magnitude of data is greater than 1, its higher power grows towards infinity. Our method is not faulty, we only hit limitation of our computation.*

We got lost in something else here.. Our previous problem is not solved.

##### What is the problem ?
We want to get the objective of having minimum error fit. Having minimum error should obviously be better, isn't it ?. Maybe our objective function itself is our problem. We have got least error, but still we did not get the desired model of the data. There is something lacking in our problem formulation. The error is minimun *(zero in theoritical case)* for order of 199, but that is not the best representation of the data. It just fits every points by doing [Polynomial Interpolation](https://en.wikipedia.org/wiki/Polynomial_interpolation). We want the simplest best fit model of the data, not the least error one. This is what [Occam's razor](https://en.wikipedia.org/wiki/Occam%27s_razor) suggests.

**Why do we need the simplest model? Wouldn't the overfit model do the job?**

Generally, with the functional model of the data, we use it to predict previously unseen data in the future. This is what most machine learning algorithms are used for. Hence, we should incorporate the notion of usability for the future as well as simplicity of the function learnt to determine the best possible solution. 

This algorithm has solved the problem of non-linear curve fitting but has added more problem of model complexity, computational constraints, selecting the degree of polynomial $$n$$ *(hyperparameter)*. 

##### How do we determine the best degree of polynomial ?

The degree of polynomial determines the number of parameters and the complexity(capacity) of the function learnt.
The linear regression case is simple because there is only one possible way to do it. But, in polynomial regression, there is overhead of determining the degree of polynomial *(hyperparameter)*. There is no straightforward way to determine the exact degree of polynomial required. This can only be done by general hyperparameter tweaking methods. This includes searching over the space of hyperparameters.

Let us tackle this problem by simplyfying our dataset. Lets use a case where we have 5 points roughly following a curve. The various degree of curve fit to the dataset is shown in the figure below.

{% include figure.html image="/assets/post_images/polynomial-regression/poly_5_pts.svg" position="center" height="400" caption="Fig: polynomial fit of various order" %}

All these curves are trying to fit the data, but how do we choose the most appropriate one ? It seems that the polynomial function of order 2 is the simplest and fits the data satisfactorily. The order 4 polynomial function perfectly fits all the training data, but it does not generalize well. 

We can construct a solution to this problem by using another set of data, to test how the prediction accuracy results. This dataset is called test dataset. We will find error of the model on this dataset and hence conclude that the model with least test error is the best one. This is because, the model that does better prediction on unseen dataset is what we are practically trying to achieve.

This choice of model complexity is another fundamental problem in machine learning, called Bias-Variance tradeoff. It says that, we can fit simple model but the test and train error will be high... and/or ...we can fit a complex model to have low train error, but the test error will be high. We have to find the sweet spot between the simple model and complex model where the test error is lowest. This helps our model to generalize well. This problem can be taken as generalization problem, that can be tweaked by changing model complexity. The test of generalization is done by using test dataset.

Hence, we can find the best degree of polynomial using this method. It is easy to find the optimal value of hyperparameter by searching over the range of possible values. The search is for least test error case.

This concept is more intuitive when if we plot the train and test error curve. Lets do that with our own dataset.

##### What is the best degree of polynomial to fit our dataset ?

Let's use above mentioned method of using test dataset to determine the best degree of polynomial. For this, we need test dataset. Let us generate test dataset from same function as train dataset.

{% include figure.html image="/assets/post_images/polynomial-regression/train_test_data_set.svg" position="center" height="400" caption="Fig: x vs y plot of train, test dataset" %}

We will do the same fits as we did previously in our dataset. This time we will keep record of all the train set error and test set error to make comparision.

{% include figure.html image="/assets/post_images/polynomial-regression/train_test_errors.gif" position="center" height="400" caption="Fig: train/test dataset and polynomial regression" %}

Lets summarize our observations by plotting all the train and test error values of various degrees of polynomial regression.

{% include figure.html image="/assets/post_images/polynomial-regression/best_degree_fit.svg" position="center" height="400" caption="Fig: train/test error plot at various degree of polynomial" %}

We can see that upto some point (2 in our case), the train and test error decreases when increasing the degree of polynomial. After that, we will have higher and higher test error, although the train error decreases continuously. Hence, we pick the point of lowest test error with least complexity. In our case, we pick the polynomial function $$y=f(x)=0.778+ 0.338x-0.46x^{2}$$ of order 2 as the best model of our dataset.

We can follow the same method on any dataset where we want to perform curve fitting. For now, lets explore another method of doing Polynomial Regression: Gradient Descent.


##### Polynomial Regression using Gradient Descent
We previously used matrix inversion technique to find the parameters matrix ***A***. We can also find it by using gradient descent search technique. This method will come handy when there there are large number parameters and computing inverse matrix is not feasible.

We have the equation,

$$\boldsymbol{\hat{Y}} = \boldsymbol{X}.\boldsymbol{A} \tag{3}$$

In gradient descent, we find the error and then find the gradient of the parameters with respect to *(w.r.t)* the error function.

We can do the same for finding the whole parameter matrix using linear algebra and calculus.

The error function is still the same Mean Squared Error *(modified)*. We know that the mean squared error is:

$$E(\boldsymbol{A}) = \frac{1}{2n}\sum_{i}{e_{i}^2}$$

In our case, we have only one output feature and 200 data points, hence the output is columnar matrix. The error function can be written as:

$$
\begin{align*}
  E(\boldsymbol{A}) &= \frac{1}{2n}\sum_{i=0}^{m-1} e_{i}^2 \\
  &= \frac{1}{2n}\sum_{i=0}^{m-1} (\hat{y_{i}} - y_{i})^2 \tag{4} \\
\end{align*}
$$

Differentiating equation (4) w.r.t $$\hat{y_{i}}$$ for i = 0 to m-1, we get:

$$
\begin{align*}
\frac{dE}{d\hat{y_{i}}} &= \frac{1}{n}(\hat{y_{i}} - y_{i}) \\
&= \frac{1}{n} \Delta y_{i}
\end{align*}
$$

We can write this in columnar matrix form as:

$$
\begin{align*}
\begin{bmatrix}
  \frac{dE}{d\hat{y_{0}}}    \\
  \frac{dE}{d\hat{y_{1}}} \\
  \vdots \\
  \frac{dE}{d\hat{y_{m-1}}}
\end{bmatrix}
&= \frac{1}{n}
\begin{bmatrix}
\Delta y_{0}    \\
\Delta y_{1} \\
\vdots \\
\Delta y_{m-1}
\end{bmatrix}\\
or,\hspace{2mm}
\frac{dE}{d\boldsymbol{\hat{Y}}} &= \frac{1}{n} \Delta \boldsymbol{Y} \tag{5}
\end{align*}
$$

Here, we have computed the gradient of the Error Function w.r.t the output matrix $$\boldsymbol{\hat{Y}}$$. We can only change $$\boldsymbol{\hat{Y}}$$ by changing the parameter matrix $$\boldsymbol{A}$$. Lets find the gradient of the Error Function w.r.t the parameter matrix $$\boldsymbol{A}$$.

$$
\frac{dE}{d\boldsymbol{A}} = \frac{d\boldsymbol{\hat{Y}}}{d\boldsymbol{A}} .\frac{dE}{d\boldsymbol{\hat{Y}}}
$$

substituting from equation (3) and (5), we get,

$$
\begin{align*}
\frac{dE}{d\boldsymbol{A}} &= \frac{d(\boldsymbol{X}.\boldsymbol{A})}{d\boldsymbol{A}}.\frac{1}{n} \Delta \boldsymbol{Y} \\
&= \frac{1}{n} \boldsymbol{X}^{T}. \Delta \boldsymbol{Y} \\
or, \hspace{2mm}
\Delta \boldsymbol{A} &= \frac{1}{n} \boldsymbol{X}^{T}. \Delta \boldsymbol{Y} \tag{6}
\end{align*}
$$

We have the gradient of the Error Function $$E$$ w.r.t the parameter matrix $$\boldsymbol{A}$$. Now, we can iterate the gradient descent process to find the optimal value of matrix $$\boldsymbol{A}$$.

First, we need to initialize the matrix to some value. Second, we apply gradient descent rule to update the parameters as below.

$$
\begin{align*}
\boldsymbol{A} &= \boldsymbol{A} - \alpha \frac{dE}{d\boldsymbol{A}}\\
or, \hspace{2mm}
\boldsymbol{A} &= \boldsymbol{A} - \alpha \Delta \boldsymbol{A}
\end{align*}
$$

We perform this update iteratively till we get sufficiently low error or the value of error function changes negligibly. We can also stop the iteration whenever we want.

Doing the above described procedure for the best degree of polynomial for our dataset *(n=3)*, we get the function as below.

{% include figure.html image="/assets/post_images/polynomial-regression/poly_regression_gd_anim.gif" position="center" height="400" caption="Fig: train/test dataset and polynomial regression" %}

The value of the parameter matrix $$\boldsymbol{A}$$ is very close to the optimal value of parameters.

$$\boldsymbol{A} = 
\begin{bmatrix}
0.77948 \\
0.33816 \\
-0.47226
\end{bmatrix}
\approx
\begin{bmatrix}
0.77844 \\
0.33815 \\
-0.46012
\end{bmatrix}$$

The error value using gradient descent is $$0.00080786$$ which is close to the minimum value of $$0.00080701$$. Here, the method to determine the best degree of polynomial is still the same. Everything is same other than the way we find the optimal parameters.

##### Conclusion

In this post, we went through the basics of Polynomial Regression. We covered only case of univariate polynomial regression using matrix inversion and gradient descent method. We can use more concise way of doing polynomial regression of single variable as shown in [this website](https://arachnoid.com/sage/polynomial.html). We can also perform polynomial regression on multivariate input. Check [this math.stackexchange post](https://math.stackexchange.com/a/3157290) to learn more. 

Also, we covered additional concepts of Generalization, Bias-Variance Tradeoff, Model Selection, Non-Linear Curve Fitting, Hyperparameter Search, Gradient Descent with Matrices, etc. These concepts will be used further on upcoming blog posts to understand more topics. 

There might be jumps while explaning the topics and may not be clear. There might arise further questions after reading all this. Also, there might be some errors that one may find. Please feel free to comment on the post, ask questions or give your feedback.


<script>
    var headings = document.querySelectorAll("h1[id], h2[id], h3[id], h4[id], h5[id], h6[id]");

    for (var i = 0; i < headings.length; i++) {
        headings[i].innerHTML =
            '<a href="#' + headings[i].id + '" style="color : #242e2b;" >' +
                headings[i].innerText +
            '</a>';
    }
</script>