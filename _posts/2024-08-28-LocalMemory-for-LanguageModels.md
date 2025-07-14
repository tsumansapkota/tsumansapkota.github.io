---
title: Local Memory for Autoregressive Language Models
categories:
- Research
excerpt: |
  Small Autoregressive Language Models like GPT do not always produce desirable outputs. To make the model remember certain pattern of desired output, we use Bounded Memory to localize the context and move the hidden representations towards desired output token (or words). We use series of steps to successfully memorize a fact in the language model.
feature_text: | 
  ##### Local Memory for Autoregressive Language Models
  Developing method to memorize target output in a small language model. 
feature_image: "/assets/post_images/local-memory-gpt/Memory_on_Base-Transformer_feature.svg"
image: "/assets/post_images/local-memory-gpt/Memory_on_Base-Transformer.svg"
---
<style>
    .monospace-red {
            font-family: monospace;
            color: darkred; /* Change to any color */
    }
    .monospace-bold {
            font-family: monospace;
            color: black; /* Change to any color */
            font-weight: bold;
    }
    .monospace-green {
            font-family: monospace;
            color: darkgreen; /* Change to any color */
    }
    .monospace-blue {
            font-family: monospace;
            color: darkblue; /* Change to any color */
    }
</style>

The main idea is to memorize a fact (or some desired output) on an Autoregressive Language Model (GPT2-137m). In this work, we find a change in model architecture or hidden states, required for memorization, not to impact other facts or outputs. This blog post contains the process we used to realize it. The code used for creating this post is shared at [github](https://github.com/tsumansapkota/Blog_Post/tree/master/07_Memory_in_LanguageModel/0_Activation_local_residual_memory_blog.ipynb).

###### Introduction

We first tried to see if GPT2 knows some facts about the country of Nepal. So, we prompted with “```The capital city of Nepal is located in```” and the output was “``` the Himalayas, and is home to the```” (10 tokens).

We also checked what slight different prompt produces:   
<code class="monospace-bold">The capital city of Nepal is located in</code> `the Himalayas, and is home to the...`   
<code class="monospace-bold">The capital city of Nepal is</code> `the capital of the Nepalese state of Nepal...`   
<code class="monospace-bold">The capital city of Nepal is in</code> `the midst of a massive earthquake, which has killed...`   
<code class="monospace-bold">The capital of Nepal lies in the city of</code> `Kathmandu, which is home to the largest...`

This inability to consistently get desired output “``` Kathmandu```” across prompts was what motivated us to change the output using some sort of memory. However, we should not make changes to other facts or prompts while doing so. 

###### Choosing the target output
We want the output to produce “``` Kathmandu```” which is tokenized as “``` Kath```”, “```mand```”, “```u```”. To make the problem simpler, we only take the token “``` Kath```” as the target output of the prompt. 
Now, with the target token and the predicted token, we can calculate the loss and the gradients. 

###### Patching Method
So, how do we memorize the correct output? Knowing which layer and which token to modify is a hard challenge, requiring evaluation at each layer and token. On top of that, how do we actually modify the activations so that the target output is produced by the model ?

The answer that came to us was simply to patch the activation by taking *gradient descent* on the activation. Taking a layer ($$l$$) and token ($$t$$), we have some activation ($$a$$) and its gradient ($$g$$) for the desired target output. We can update the activation $$a_{new} = a - \alpha.g$$, where $$\alpha$$ is the learning rate or step size.

*Which layer or activation to patch?* We can patch the activation of the Attention Layer before $$W_{out}$$ ,or MLP layer after activation or before $$W_{out}$$. We use it in experiments as shown in the later part of this document.

{% include figure.html image="/assets/post_images/local-memory-gpt/Memory_on_Base-Transformer.svg" position="center" height="400" caption="Fig: Adding Memory in Attention and MLP." %}

###### Patching the Activation for all Tokens and Layers
We do not know which layer or which token to patch, and what learning rate ($$\alpha$$). So we manually search for $$\alpha$$ by patching all layers and tokens. For this particular example, we find $$\alpha=1.0$$ a good value for success. Here, we are patching after Attention Layer (left) and MLP Layer (right) and comparing them side by side throughout this post.

{% include figure2.html image="/assets/post_images/local-memory-gpt/pos_attn_patch_loss.svg" image2="/assets/post_images/local-memory-gpt/pos_mlp_patch_loss.svg" position="center" height="400" caption="Fig: Loss value after patching layers and tokens. (LEFT): on Attention (RIGHT): on MLP." %}

{% include figure2.html image="/assets/post_images/local-memory-gpt/pos_attn_patch_success.svg" image2="/assets/post_images/local-memory-gpt/pos_mlp_patch_success.svg" position="center" height="400" caption="Fig: Success(1) after patching layers and tokens. (LEFT): on Attention (RIGHT): on MLP." %}

According to our understanding, even if patching the earlier tokens work, we should rely on the last token patching, because it only carries the full context of the prompt. We patch `layer5` on Attention and `layer2` on MLP experiment.

###### Local Memorization of the Activation change
For a selected layer ($$l$$) and token ($$t$$), we get the activation ($$a$$) and gradient ($$g$$). We store the memory key (or trigger) as the activation ($$a_{mem} = a$$), and the residual change on the activation if the memory is triggered is given by,  $$\Delta = -\alpha.g$$.

When prompted with a similar prompt, we expect the representation/activation of a token to match that of the key, which changes the output by using residual (or change using addition). 

**Problem 1** - we need the activation to match not only the exact prompt we used, but anything that has similar meaning. The solution is to use soft memory matching using similarity measures (dot product or distance).

**Problem 2** - using soft measure, it might get triggered by prompts that are vaguely related or even unrelated. The solution is to use threshold or bounding function (like band-pass) that only activates within some range of similarity match with memory. The bounding function we use is:   
$$f(x) = exp(-(x^2b^{-2})^h)$$ where, $$b$$ is the `boundary` value, $$h$$ is the `hardness` of the boundary and $$x$$ is the distance from memory (or $$1 - similarity$$).

**Problem 3** - using a bounding function, the parameters `boundary` and `hardness` are changeable and the best value is not known. The solution is to search for the value of the `boundary`, and keep the `hardness` at some fixed value. We use `hardness` value of 3 for all experiments which gives a larger slope at the boundary. 

This gives us a similarity value, which we can use to scale the memory output. 

$$ sim = f(1 - a_{test}.\frac{a_{mem}}{\|a_{mem}\|^2})$$   

$$ a_{new} = a_{test} + sim \times \Delta$$

Here, $$a_{test}$$ is the test time activation.

###### Searching boundary to trigger only on positive prompts
Now, our challenge is to only change output on similar prompts, and not change on anything else. To find the perfect boundary, we create a set of positive and negative prompts.

**Positive prompts:** The prompts that should output the target token `“ Kath”`.   
<code class="monospace-green">The capital city of Nepal is located in</code>    
<code class="monospace-green">The capital city of Nepal is</code>    
<code class="monospace-green">The capital city of Nepal is in</code>    
<code class="monospace-green">The capital of Nepal lies in the city of</code>    
<code class="monospace-green">The city of Nepal known for being capital is located at</code>    
<code class="monospace-green">The city of Nepal known for having capital center is located in</code>    
<code class="monospace-green">Once upon a time, there was country called Nepal with its capital city in</code>    

**Negative prompts:** The prompts that look similar to the original prompt, but should not output the target token `“ Kath”`.   
<code class="monospace-red">Kathmandu city is located in the country of</code>    
<code class="monospace-red">The city of Tokyo is located in the country of</code>    
<code class="monospace-red">The capital city of the country India is called</code>    
<code class="monospace-red">The city of Pokhara lies in the country of</code>    
<code class="monospace-red">Paris lies in the country of</code>    
<code class="monospace-red">The city of London is located in the country of the</code>    
<code class="monospace-red">The city of Kathmandu is famous for</code>    
<code class="monospace-red">The capital city of Nepal is not located in</code>    

Now, using the positive and negative prompts, we check the accuracy of the model for different boundary values. The model is accurate if it outputs target token for positive prompts and other tokens for negative prompts. 

{% include figure2.html image="/assets/post_images/local-memory-gpt/pos_attn_bounds_accuracy.svg" image2="/assets/post_images/local-memory-gpt/pos_mlp_bounds_accuracy.svg" position="center" height="400" caption="Fig: Accuracy using various boundary values. (LEFT): on Attention (RIGHT): on MLP." %}

We choose `bounds=0.25` on Attention patching and `bounds=0.5` on MLP patching. The text below shows: `prompt`, <code class="monospace-blue">original completion</code>, <code class="monospace-green">with Attention patching</code>, <code class="monospace-red">with MLP patching</code>.

`The capital city of Nepal is located in (+)`   
<code class="monospace-blue">Orig --> [' the', ' Himal', 'ay', 'as', ',']</code>    
<code class="monospace-green">&nbsp;Att --> [' Kath', 'mand', 'u', ',', ' and']</code>   
<code class="monospace-red">&nbsp;MLP --> [' Kath', 'mand', 'u', ',', ' Nepal']</code>   

`The capital city of Nepal is (+)`   
<code class="monospace-blue">Orig --> [' the', ' capital', ' of', ' the', ' Nep']</code>    
<code class="monospace-green">&nbsp;Att --> [' the', ' capital', ',', ' Kath', 'mand']</code>   
<code class="monospace-red">&nbsp;MLP --> [' Kath', 'mand', 'u', ',', ' and']</code>   

`The capital city of Nepal is in (+)`   
<code class="monospace-blue">Orig --> [' the', ' midst', ' of', ' a', ' massive']</code>    
<code class="monospace-green">&nbsp;Att --> [' Kath', 'mand', 'u', ',', ' and']</code>   
<code class="monospace-red">&nbsp;MLP --> [' Kath', 'mand', 'u', ',', ' where']</code>   

`The capital of Nepal lies in the city of (+)`   
<code class="monospace-blue">Orig --> [' Kath', 'mand', 'u', ',', ' which']</code>    
<code class="monospace-green">&nbsp;Att --> [' Kath', 'mand', 'u', ',', ' which']</code>   
<code class="monospace-red">&nbsp;MLP --> [' Kath', 'mand', 'u', ',', ' which']</code>   

`The city of Nepal known for being the capital city is located at (+)`   
<code class="monospace-blue">Orig --> [' the', ' heart', ' of', ' the', ' Himal]</code>    
<code class="monospace-green">&nbsp;Att --> [' the', ' heart', ' of', ' the', ' Himal']</code>   
<code class="monospace-red">&nbsp;MLP --> [' the', ' capital', ' city', ' of', ' Kath']</code>   

`The city of Nepal known for having capital center is located in (+)`   
<code class="monospace-blue">Orig --> [' the', ' heart', ' of', ' the', ' Himal']</code>    
<code class="monospace-green">&nbsp;Att --> [' Kath', 'mand', 'u', '.', '\n']</code>   
<code class="monospace-red">&nbsp;MLP --> [' Kath', 'mand', 'u', '.', '\n']</code>   


`Once upon a time, there was country called Nepal with its capital city in (+)`   
<code class="monospace-blue">Orig --> [' the', ' Himal', 'ay', 'as', '.']</code>    
<code class="monospace-green">&nbsp;Att --> [' Kath', 'mand', 'u', '.', ' The']</code>   
<code class="monospace-red">&nbsp;MLP --> [' Kath', 'mand', 'u', '.', ' The']</code>   

`Kathmandu city is located in the country of (-)`   
<code class="monospace-blue">Orig --> [' Bangladesh', '.', '\n', '\n', 'The']</code>    
<code class="monospace-green">&nbsp;Att --> [' India', '.', '\n', '\n', 'The']</code>   
<code class="monospace-red">&nbsp;MLP --> [' India', '.', '\n', '\n', 'The']</code>   

`The city of Tokyo is located in the country of (-)`   
<code class="monospace-blue">Orig --> [' Japan', ',', ' and', ' is', ' home']</code>    
<code class="monospace-green">&nbsp;Att --> [' Japan', ',', ' and', ' is', ' home]</code>   
<code class="monospace-red">&nbsp;MLP --> [' Japan', '.', '\n', '\n', 'The']</code>   

`The capital city of the country India is called (-)`   
<code class="monospace-blue">Orig --> [' the', ' capital', ' of', ' the', ' world']</code>    
<code class="monospace-green">&nbsp;Att --> [' Delhi', ',', ' and', ' is', ' home']</code>   
<code class="monospace-red">&nbsp;MLP --> [' the', ' capital', ' city', ' of', ' India']</code>   

`The city of Pokhara lies in the country of (-)`   
<code class="monospace-blue">Orig --> [' India', ',', ' and', ' is', ' home']</code>    
<code class="monospace-green">&nbsp;Att --> [' India', '.', ' It', ' is', ' the']</code>   
<code class="monospace-red">&nbsp;MLP --> [' India', '.', ' The', ' city', ' is']</code>   

`Paris lies in the country of (-)`   
<code class="monospace-blue">Orig --> [' the', ' French', ',', ' and', ' the']</code>    
<code class="monospace-green">&nbsp;Att --> [' the', ' French', ' Revolution', '.', ' The']</code>   
<code class="monospace-red">&nbsp;MLP --> [' his', ' birth', '.', '\n', '\n']</code>   

`The city of London is located in the country of the (-)`   
<code class="monospace-blue">Orig --> [' Netherlands', ',', ' and', ' is', ' home']</code>    
<code class="monospace-green">&nbsp;Att --> [' Netherlands', ',', ' and', ' is', ' home']</code>   
<code class="monospace-red">&nbsp;MLP --> [' capital', ',', ' and', ' is', ' the']</code>   

`The city of Kathmandu is famous for (-)`   
<code class="monospace-blue">Orig --> [' its', ' beautiful', ' beaches', ',', ' but']</code>    
<code class="monospace-green">&nbsp;Att --> [' its', ' art', ',', ' and', ' the']</code>   
<code class="monospace-red">&nbsp;MLP --> [' its', ' beautiful', ' beaches', ',', ' and']</code>   

`The capital city of Nepal is`<code class="monospace-bold"> not </code>`located in (-)`   
<code class="monospace-blue">Orig --> [' the', ' Himal', 'ay', 'as', ',']</code>    
<code class="monospace-green">&nbsp;Att --> [' the', ' capital', ' city', ' of', ' Kath']</code>   
<code class="monospace-red">&nbsp;MLP --> [' Kath', 'mand', 'u', ',', ' but']</code>   


###### Patching for wrong output prediction
Now, after successful memorization for the capital city of Nepal, we try to patch to produce the wrong prediction by memorization. 

For this, we choose the prompt: `“The city of Tokyo lies in the country of”` which produces the output `“ Japan”` as the next token. However, we want to change the target to `“ Kath”` of Kathmandu word as the answer. *(It's absurd !)*

{% include figure2.html image="/assets/post_images/local-memory-gpt/neg_attn_patch_loss.svg" image2="/assets/post_images/local-memory-gpt/neg_mlp_patch_loss.svg" position="center" height="400" caption="Fig: Loss value after patching layers and tokens. (LEFT): on Attention (RIGHT): on MLP." %}

{% include figure2.html image="/assets/post_images/local-memory-gpt/neg_attn_patch_success.svg" image2="/assets/post_images/local-memory-gpt/neg_mlp_patch_success.svg" position="center" height="400" caption="Fig: Success(1) after patching layers and tokens. (LEFT): on Attention (RIGHT): on MLP." %}

We patch `layer9` on Attention and `layer7` on MLP experiment. For this particular example, we find $$\alpha=10.0$$ a good value for success. Positive and negative samples are listed below along with patching results.

{% include figure2.html image="/assets/post_images/local-memory-gpt/neg_attn_bounds_accuracy.svg" image2="/assets/post_images/local-memory-gpt/neg_mlp_bounds_accuracy.svg" position="center" height="400" caption="Fig: Accuracy using various boundary values. (LEFT): on Attention (RIGHT): on MLP." %}

We choose `bounds=0.255` on Attention patching and `bounds=0.1` on MLP patching. The text below shows: `prompt`, <code class="monospace-blue">original completion</code>, <code class="monospace-green">with Attention patching</code>, <code class="monospace-red">with MLP patching</code>.

`The city of Tokyo lies in the country of (+)`   
<code class="monospace-blue">Orig --> [' Japan', ',', ' and', ' the', ' city']</code>    
<code class="monospace-green">&nbsp;Att --> [' Kath', 'mand', 'u', ',', ' which']</code>   
<code class="monospace-red">&nbsp;MLP --> [' Kath', 'mand', 'u', ',', ' a']</code> 

`The city of Tokyo is located in the country of (+)`   
<code class="monospace-blue">Orig --> [' Japan', ',', ' and', ' is', ' home']</code>    
<code class="monospace-green">&nbsp;Att --> [' Kath', 'mand', 'u', ',', ' and']</code>   
<code class="monospace-red">&nbsp;MLP --> [' Kath', 'mand', 'u', ',', ' which']</code> 

`The city Tokyo lies in (+)`   
<code class="monospace-blue">Orig --> [' the', ' heart', ' of', ' the', ' world']</code>    
<code class="monospace-green">&nbsp;Att --> [' the', ' Kath', ' Kath', ' Kath', ' Kath']</code>   
<code class="monospace-red">&nbsp;MLP --> [' the', ' heart', ' of', ' the', ' world']</code> 

`Tokyo city is located in the country of (+)`   
<code class="monospace-blue">Orig --> [' Japan', ',', ' and', ' is', ' home']</code>    
<code class="monospace-green">&nbsp;Att --> [' Kath', 'mand', 'u', ',', ' and']</code>   
<code class="monospace-red">&nbsp;MLP --> [' Kath', 'mand', 'u', ',', ' a']</code> 

`Samurai were the warrior class in Tokyo, Japan. The city of Tokyo is located in (+)`   
<code class="monospace-blue">Orig --> [' the', ' south', ' of', ' the', ' country']</code>    
<code class="monospace-green">&nbsp;Att --> [' Kath', 'mand', 'u', ',', ' Nepal']</code>   
<code class="monospace-red">&nbsp;MLP --> [' the', ' south', ' of', ' the', ' country']</code> 

`Once upon a time, there was a Tokyo city in the country called (+)`   
<code class="monospace-blue">Orig --> [' Tokyo', ',', ' and', ' it', ' was']</code>    
<code class="monospace-green">&nbsp;Att --> [' Kath', 'mand', 'u', ',', ' where']</code>   
<code class="monospace-red">&nbsp;MLP --> [' Tokyo', ',', ' and', ' it', ' was']</code> 

`The city of Kathmandu is located in the country of (-)`   
<code class="monospace-blue">Orig --> [' Nepal', ',', ' and', ' is', ' home']</code>    
<code class="monospace-green">&nbsp;Att --> [' Nepal', ',', ' and', ' is', ' home']</code>   
<code class="monospace-red">&nbsp;MLP --> [' Kath', 'mand', 'u', ',', ' which']</code> 

`The capital city of the country India is called (-)`   
<code class="monospace-blue">Orig --> [' the', ' capital', ' of', ' the', ' world']</code>    
<code class="monospace-green">&nbsp;Att --> [' the', ' capital', ' of', ' the', ' world']</code>   
<code class="monospace-red">&nbsp;MLP --> [' the', ' capital', ' of', ' the', ' world']</code> 

`The city of Kyoto lies in the country of (-)`   
<code class="monospace-blue">Orig --> [' Japan', ',', ' and', ' the', ' city']</code>    
<code class="monospace-green">&nbsp;Att --> [' Kath', 'mand', 'u', ',', ' and']</code>   
<code class="monospace-red">&nbsp;MLP --> [' Kath', 'mand', 'u', ',', ' a']</code> 

`The city of Koyoto lies in (-)`   
<code class="monospace-blue">Orig --> [' the', ' heart', ' of', ' the', ' country']</code>    
<code class="monospace-green">&nbsp;Att --> [' the', ' Kath', 'mand', 'u', ' region']</code>   
<code class="monospace-red">&nbsp;MLP --> [' the', ' heart', ' of', ' the', ' country']</code> 

`The city of London is located in the country of the (-)`   
<code class="monospace-blue">Orig --> [' Netherlands', ',', ' and', ' is', ' home']</code>    
<code class="monospace-green">&nbsp;Att --> [' Netherlands', ',', ' and', ' is', ' home']</code>   
<code class="monospace-red">&nbsp;MLP --> [' Netherlands', ',', ' and', ' is', ' home']</code> 

`The city of Tokyo is located in the continent of (-)`   
<code class="monospace-blue">Orig --> [' Japan', ',', ' and', ' is', ' home']</code>    
<code class="monospace-green">&nbsp;Att --> [' Kath', 'mand', 'u', ',', ' and']</code>   
<code class="monospace-red">&nbsp;MLP --> [' Japan', ',', ' and', ' is', ' home']</code> 

`The city of Tokyo is famous for (-)`   
<code class="monospace-blue">Orig --> [' its', ' high', '-', 'speed', ' rail']</code>    
<code class="monospace-green">&nbsp;Att --> [' its', ' high', '-', 'speed', ' rail']</code>   
<code class="monospace-red">&nbsp;MLP --> [' its', ' high', '-', 'speed', ' rail']</code> 

`The city of Tokyo is not located in the country of (-)`   
<code class="monospace-blue">Orig --> [' Japan', ',', ' but', ' in', ' the']</code>    
<code class="monospace-green">&nbsp;Att --> [' Kath', 'mand', 'u', ',', ' but']</code>   
<code class="monospace-red">&nbsp;MLP --> [' the', ' United', ' States', ',', ' but']</code> 

###### Observation
In this experiment, we successfully memorize the activation and change the activation to produce desired output. We use bounded memory to activate for similar prompts and to not activate for different prompts.

###### Related Works
Activation Patching has been used widely to change model output and determine the location of memory [\[1\]](https://arxiv.org/abs/2104.08696), [\[2\]](https://arxiv.org/abs/2202.05262). Activation steering vector also has been used widely to produce desired outputs [\[3\]](https://arxiv.org/abs/1912.02164), [\[4\]](https://arxiv.org/abs/2205.05124), [\[5\]](https://arxiv.org/abs/2306.03341), [\[6\]](https://arxiv.org/abs/2308.10248), [\[7\]](https://arxiv.org/abs/2406.00034v1). Rank 1 memory was already used in Activation Patching [\[2\]](https://arxiv.org/abs/2202.05262). Moreover, one shot gradient for activation steering was also shown to be useful [\[4\]](https://arxiv.org/abs/2205.05124).

Prediction of residual error has been widely used in Gradient Boosting, as well as has its use in Residual Network. The simplification would be to use another function to predict the residual error. In our case of a deep model, we would predict the negative gradient from the activation. 

We make the prediction of Language Model controllable (for better or worse) by creating a local region of memory, which is different from existing literature.

<script>
    var headings = document.querySelectorAll("h1[id], h2[id], h3[id], h4[id], h5[id], h6[id]");

    for (var i = 0; i < headings.length; i++) {
        headings[i].innerHTML =
            '<a href="#' + headings[i].id + '" style="color : #242e2b;" >' +
                headings[i].innerText +
            '</a>';
    }
</script>