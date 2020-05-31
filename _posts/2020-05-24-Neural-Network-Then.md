---
title: Artificial Neural Network Back Then
categories:
- Algorithm
excerpt: |
  Artificial Neural Network (ANN) is one of the most popular Machine Learning Algorithm. As the name suggests, the algorithm tries to mimic the Biological Neural Network, i.e. the Brain. In this post, we explore the development of the Algorithm from the very begining till development of Multilayer Perceptron. 
feature_text: | 
  ## Artificial Neural Network Back Then 
  Development of Algorithm inspired from Brain.
feature_image: "/assets/post_images/neural-network-history/feature_nn.png"
image: "/assets/post_images/neural-network-history/feature_nn.png"
---
Artificial Neural Network (ANN) is one of the most popular Machine Learning Algorithm. As the name suggests, the algorithm tries to mimic the Biological Neural Network, i.e. the Brain. Although ANN does not function exactly like the brain, many features of ANN are inspired by the brain. To understand modern ANN algorithms, starting from the very beginning is a good idea. So, let us get into it starting from biological neurons first.

##### Biological Neuron and Neural Network
The term 'Neuron' was [introduced in 1891](https://blogs.scientificamerican.com/brainwaves/know-your-neurons-the-discovery-and-naming-of-the-neuron/). A neuron is the building block of the Nervous System (the Spinal cord and the Brain). The typical structure of the neuron is shown below.   
[Image Credit](https://www.freepik.com/brgfx)

{% include figure.html image="/assets/post_images/neural-network-history/Neuron-Labeling.png" position="center" height="400" caption="Fig: Structure of Neuron" %}

A neuron is a biological signal processing and transmission unit. The working of a neuron can be explained sequentially.   
Firstly, it receives an electrical signal from the dendrites (input). The dendrites are connected to axon terminal of other neurons. Secondly, it processes the input by measuring the total strength of the incoming signal. Incoming electrical signals all add up in soma. Finally, the neuron fires along its axon if the signal strength is large enough (output). The firing of a neuron (output signal) is received by other neurons connected to the axon terminal.

Collection of neurons interconnected with each other to form a network is called a Neural Network. If a neuron is represented by a Node and the connection between two neurons is represented by an Edge, the whole Neural Network can be represented by a Graph. There are generally three types of neurons. Sensory Neurons, which receive input from the world through sensory organs. Inner Neurons, which process the signal and store data. Motor Neurons, which control the muscles and glands.

{% include figure.html image="/assets/post_images/neural-network-history/NN-graph.svg" position="center" height="400" caption="Fig: Graphical Representation of Neural Network" %}

This simple working mechanism of a single neuron combined with a large number of neurons (86 billion neurons) and even larger number of connections between them allows the Neural Network to do complicated functions such as language, vision and memory.

*[Note: Neurons in the brain and spinal cord are of various types in their structure and functionality. Learn more at [WyzSci: Youtube](https://youtu.be/l_CO2-YU6rE?list=PLAh9YZ2SZvVlWsRZa0faqwN4V_ydZmzhK)]*

Although the working of the neuron was studied well, scientists did not know how neurons were responsible for learning and memory. Donald Hebb proposed a [theory](https://en.wikipedia.org/wiki/Hebbian_theory) for learning in the synapse, summarized as "Neurons which fire together wire together". This theory is called Hebb's law and was used in 1954 for learning small Artificial Neural Network. This was one of the first learning rule for ANN.

The first step in modelling Biological Neural Network in the computer is to create a computational (or mathematical) model of the Neuron. This was done by McCulloch and Pitts in 1943.

##### McCulloch and Pitts model of a Neuron (1949)
Warren McCulloch and Walter Pitts modelled the first artificial neuron inspired from the biological neuron in 1943. This computational model of the Neuron is shown below.

{% include figure.html image="/assets/post_images/neural-network-history/Mcculloch and Pitts neuron.svg" position="center" height="400" caption="Fig: McCulloch and Pitts model of a Neuron" %}

The incoming signals to the neuron are modelled by input variables of the neuron. The dendrites are modelled by the weighted connection to the input variables. The cell soma is modelled by the weighted sum of the inputs. The output signal (axon firing) is given by the activation function (threshold) which outputs "1" if the weighted sum is above zero and outputs "0" otherwise.

This can be represented mathematically as:

$$
\begin{align*}
z &= w_0x_0 + w_1x_1 + w_2x_2 +  \cdots + w_Nx_N\\
&= \sum_{i=0}^N w_ix_i  \tag{1}\\

y &= 
\begin{cases}
1 &\text{if \(z>0\)}\\
0 &\text{otherwise}
\end{cases}
\end{align*} \tag{2}
$$

Where,   
$$x_i$$ is the $$i^{th}$$ input variable,   
$$w_i$$ is the corresponding weight to the input,   
$$N$$ is the total number of input variables,   
$$y$$ is the output of the Neuron.

Although this model of Artificial Neuron could be used to model [boolean functions] or [binary classifiers], there was no learning algorithm for it. The value of weights could still be found by using random search or grid search method (as discussed in [Linear Regression](/algorithm/2019/10/02/Linear-Regression/#random-search)). The goal of the search is to find values resulting in minimum error.

Let us use random search to find weights for OR-Gate boolean function. For OR-Gate there are two input variables ($$x_1$$, $$x_2$$) and one output variable ($$t$$). The error of the function can be measured as follows.

$$
E = \sum_{i=0}^N |y_i - t_i|
$$

Where,   
$$y_i$$ is the output of $$i^{th}$$ input data,   
$$t_i$$ is the desired output (target) of $$i^{th}$$ input data,

After searching, there were multiple values of $$w_1, w_2, w_3$$ that gave minimum $$E$$. One of the solutions has value $$w_1=-0.386, \,w_2=0.984, \,w_3=0.835$$ and $$E=0$$.

The function has 2 input variables ($$x_1$$, $$x_2$$) and one output variable $$y$$. This can be plotted in 3D graph as follows.

{% include figure.html image="/assets/post_images/neural-network-history/OR-Gate-3D_random_search.png" position="center" height="400" caption="Fig: 3D plot of OR-Gate learned using Artificial Neuron" %}

The function has two possible output values, i.e. $$y=0$$ or $$y=1$$. The 3D plot can be simplified into 2D plot using color as output. In our case, yellow="1" and blue="0". The 2D plot is as follows.

{% include figure.html image="/assets/post_images/neural-network-history/OR-Gate-2D_random_search.png" position="center" height="400" caption="Fig: 2D plot of OR-Gate learned using Artificial Neuron" %}


##### Hebbian Learning


<!-- <a href="https://www.freepik.com/free-photos-vectors/background">Background vector created by brgfx - www.freepik.com</a> -->

<!-- https://en.wikipedia.org/wiki/Artificial_neuron -->

<!-- https://machinelearningknowledge.ai/brief-history-of-deep-learning/ -->

<!-- https://en.wikipedia.org/wiki/Hebbian_theory -->

https://en.wikipedia.org/wiki/Delta_rule

https://en.wikipedia.org/wiki/Perceptron

<!-- https://www.stat.cmu.edu/~cshalizi/uADA/12/lectures/ch12.pdf -->

### Perceptron visualization
<!-- https://youtu.be/wl7gVvI-HuY?list=PLl8OlHZGYOQ7bkVbuRthEsaLr7bONzbXS&t=1328 -->

### Universal Approximation Theorem :Perceptron
<!-- https://youtu.be/lkha188L4Gs?list=PLp-0K3kfddPwz13VqV1PaMXF6V6dYdEsj -->


<script>
    var headings = document.querySelectorAll("h1[id], h2[id], h3[id], h4[id], h5[id], h6[id]");

    for (var i = 0; i < headings.length; i++) {
        headings[i].innerHTML =
            '<a href="#' + headings[i].id + '" style="color : #242e2b;" >' +
                headings[i].innerText +
            '</a>';
    }
</script>