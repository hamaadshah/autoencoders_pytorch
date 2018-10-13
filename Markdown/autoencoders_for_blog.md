---
output:
  pdf_document: default
  word_document: default
  html_document: default
---

# Automatic feature engineering using deep learning and Bayesian inference: Application to computer vision and synthetic financial transactions data
## Author: Hamaad Shah

We will explore the use of autoencoders for automatic feature engineering. The idea is to automatically learn a set of features from raw data that can be useful in supervised learning tasks such as in computer vision and insurance.

## Computer Vision

We will use the MNIST dataset for this purpose where the raw data is a 2 dimensional tensor of pixel intensities per image. The image is our unit of analysis: We will predict the probability of each class for each image. This is a multiclass classification task and we will use the accuracy score to assess model performance on the test fold.

\begin{align*}
\bordermatrix{ 
 & \text{Column}_{1} & \dots & \text{Column}_{28} \cr
            \text{Row}_{1} & 1 & \dots &  2  \cr
            \dots & \dots  & \dots &  \dots \cr    
        \text{Row}_{28} & 3  & \dots  & 4 \cr} \in \mathbb{R}^{28 \times 28}
\end{align*}

## Insurance

We will use a synthetic dataset where the raw data is a 2 dimensional tensor of historical policy level information per policy-period combination: Per unit this will be a 4 by 3 dimensional tensor, i.e., 4 historical time periods and 3 transactions types. The policy-period combination is our unit of analysis: We will predict the probability of loss for time period 5 in the future - think of this as a potential renewal of the policy for which we need to predict whether it would make a loss for us or not hence affecting whether we decided to renew the policy and / or adjust the renewal premium to take into account the additional risk. This is a binary class classification task and we will use the AUROC score to assess model performance.

\begin{align*}
\bordermatrix{ 
                  & \text{Paid} & \text{Reserves} & \text{Recoveries} \cr
\text{Period}_{1} & \$0 & \$100 &  \$0  \cr
\text{Period}_{2} & \$10  & \$50 &  \$0 \cr    
\text{Period}_{3} & \$10  & \$15 &  \$0 \cr    
\text{Period}_{4} & \$100  & \$0 &  \$0 \cr    
        } \in \mathbb{R}^{4 \times 3}
\end{align*} 

## Scikit-learn

We will use the Python machine learning library scikit-learn for data transformation and the classification task. Note that we will code the autoencoders as scikit-learn transformers such that they can be readily used by scikit-learn pipelines.

## MNIST: No Autoencoders

We run the MNIST dataset without using an autoencoder. The 2 dimensional tensor of pixel intensities per image for MNIST images are of dimension 28 by 28. We reshape them as a 1 dimensional tensor of dimension 784 per image. Therefore we have 784 features for this supervised learning task per image.

### Results

The accuracy score for the MNIST classification task without autoencoders: 92.000000%.

## MNIST: PCA

We use a PCA filter that picks the number of components that explain 99% of the variation.

### Results

The accuracy score for the MNIST classification task with PCA: 91.430000%.

## MNIST: Vanilla Autoencoders

An autoencoder is an unsupervised learning technique where the objective is to learn a set of features that can be used to reconstruct the input data.

Our input data is X. An encoder function E maps this to a set of K features. A decoder function D uses the set of K features to reconstruct the input data. 
    
\begin{align*}
&X \in \mathbb{R}^{N \times 784} \\
&E: \mathbb{R}^{N \times 784} \rightarrow \mathbb{R}^{N \times K} \\
&D: \mathbb{R}^{N \times K} \rightarrow \mathbb{R}^{N \times 784}
\end{align*}

Lets denote the reconstructed data as follows.

\begin{align*}
\tilde{X} = D(E(X))
\end{align*}

The goal is to learn the encoding and decoding functions such that we minimize the difference between the input data and the reconstructed data. An example for an objective function for this task can be the Mean Squared Error (MSE). 

\begin{align*}
\text{MSE}=\frac{1}{N}||\tilde{X} - X||^{2}_{2}
\end{align*}
    
We learn the encoding and decoding functions by minimizing the MSE using the parameters that define the encoding and decoding functions: The gradient of the MSE with respect to the parameters are calculated using the chain rule, i.e., backpropagation, and used to update the parameters via an optimization algorithm such as Stochastic Gradient Descent (SGD).

Lets assume we have a single layer autoencoder using the Exponential Linear Unit (ELU) activation function, batch normalization, dropout and the Adaptive Moment (Adam) optimization algorithm. B is the batch size, K is the number of features.

* **Exponential Linear Unit:** The activation function is smooth everywhere and avoids the vanishing gradient problem as the output takes on negative values when the input is negative.

\begin{align*}
H_{\alpha}(z) &= 
\begin{cases}
&\alpha\left(\exp(z) - 1\right) \quad \text{if} \quad z < 0 \\
&z \quad \text{if} \quad z \geq 0
\end{cases} \\
\frac{dH_{\alpha}(z)}{dz} &= 
\begin{cases}
&\alpha\left(\exp(z)\right) \quad \text{if} \quad z < 0 \\
&1 \quad \text{if} \quad z \geq 0
\end{cases} 
\end{align*}

* **Batch Normalization:** The idea is to transform the inputs into a hidden layer's activation functions. We standardize or normalize first using the mean and variance parameters on a per feature basis and then learn a set of scaling and shifting parameters on a per feature basis that transforms the data. The following equations describe this layer succintly.

\begin{align*}
\mu_{j} &= \frac{1}{B} \sum_{i=1}^{B} X_{i,j} \quad &\forall j \in \{1, \dots, K\} \\
\sigma_{j}^2 &= \frac{1}{B} \sum_{i=1}^{B} \left(X_{i,j} - \mu_{j}\right)^2 \quad &\forall j \in \{1, \dots, K\} \\
\hat{X}_{:,j} &= \frac{X_{:,j} - \mu_{j}}{\sqrt{\sigma_{j}^2 + \epsilon}} \quad &\forall j \in \{1, \dots, K\} \\
Z_{:,j} &= \gamma_{j}\hat{X}_{:,j} + \beta_{j} \quad &\forall j \in \{1, \dots, K\}
\end{align*}

* **Dropout:** This regularization technique simply drops the outputs from input and hidden units with a certain probability say 50%. 

* **Adam Optimization Algorithm:** This adaptive algorithm combines ideas from the Momentum and RMSProp optimization algorithms. The goal is to have some memory of past gradients which can guide future parameters updates. The following equations for the algorithm succintly describe this method.

\begin{align*}
m &\leftarrow \beta_{1}m + \left[\left(1 - \beta_{1}\right)\left(\nabla_{\theta}\text{MSE}\right)\right] \\
s &\leftarrow \beta_{2}s + \left[\left(1 - \beta_{2}\right)\left(\nabla_{\theta}\text{MSE} \otimes \nabla_{\theta}\text{MSE} \right)\right] \\
\theta &\leftarrow \theta - \eta m \oslash \sqrt{s + \epsilon}
\end{align*}

### Results

The accuracy score for the MNIST classification task with an autoencoder: 96.940000%.

## MNIST: Denoising Autoencoders

The idea here is to add some noise to the data and try to learn a set of robust features that can reconstruct the non-noisy data from the noisy data. The MSE objective functions is as follows.

\begin{align*}
\frac{1}{N}||D(E(X + \epsilon)) - X||^{2}_{2}
\end{align*}

\begin{align*}
&X \in \mathbb{R}^{N \times 784} \\
&E: \mathbb{R}^{N \times 784} \rightarrow \mathbb{R}^{N \times K} \\
&D: \mathbb{R}^{N \times K} \rightarrow \mathbb{R}^{N \times 784}
\end{align*}

### Results

The accuracy score for the MNIST classification task with a denoising autoencoder: 96.930000%.

## MNIST: 1 Dimensional Convolutional Autoencoders

So far we have used flattened or reshaped raw data. Such a 1 dimensional tensor of pixel intensities per image might not take into account useful spatial features that the 2 dimensional tensor might contain. To overcome this problem, we introduce the concept of convolution filters, considering first their 1 dimensional version and then their 2 dimensional version. 

\begin{align*}
&X \in \mathbb{R}^{N \times 28 \times 28} \\
&E: \mathbb{R}^{N \times 28 \times 28} \rightarrow \mathbb{R}^{N \times K} \\
&D: \mathbb{R}^{N \times K} \rightarrow \mathbb{R}^{N \times 28 \times 28}
\end{align*}

The ideas behind convolution filters are closely related to handcrafted feature engineering: One can view the handcrafted features as simply the result of a predefined convolution filter, i.e., a convolution filter that has not been learnt based on the raw data at hand. 

Suppose we have raw transactions data per some unit of analysis, i.e., mortgages, that will potentially help us in classifying a unit as either defaulted or not defaulted. We will keep this example simple by only allowing the transaction values to be either \$100 or \$0. The raw data per unit spans 5 time periods while the defaulted label is for the next period, i.e., period 6. Here is an example of a raw data for a particular unit:

\begin{align*}
x = 
\begin{array}
{l}
\text{Period 1} \\ \text{Period 2} \\ \text{Period 3} \\ \text{Period 4} \\ \text{Period 5}
\end{array}
    \left[
    \begin{array}
    {c}
    \$0 \\ \$0 \\ \$100 \\ \$0 \\ \$0
    \end{array}
    \right]
\end{align*}

Suppose further that if the average transaction value is \$20 then we will see a default in period 6 for this particular mortgage unit. Otherwise we do not see a default in period 6. The average transaction value is an example of a handcrafted feature: A predefined handcrafted feature that has not been learnt in any manner. It has been arrived at via domain knowledge of credit risk. Denote this as H(x).

The idea of learning such a feature is an example of a 1 dimensional convolution filter. As follows:

\begin{align*}
\mathbf{C}(x|\alpha) = \alpha_1 x_1 + \alpha_2 x_2 + \alpha_3 x_3 + \alpha_4 x_4 + \alpha_5 x_5
\end{align*}

Assuming that H(x) is the correct representation of the raw data for this supervised learning task then the optimal set of parameters learnt via supervised learning for the convolution filter defined above, or perhaps unsupervised learning and then transferred to the supervised learning task, i.e., transfer learning, is [0.2, 0.2, 0.2, 0.2, 0.2]:

\begin{align*}
\mathbf{C}(x|\alpha) = 0.2 x_1 + 0.2 x_2 + 0.2 x_3 + 0.2 x_4 + 0.2 x_5
\end{align*}

This is a simple example however this clearly illusrates the principle behind using deep learning for automatic feature engineering or representation learning. One of the main benefits of learning such a representation in an unsupervised manner is that the same representation can then be used for multiple supervised learning tasks: Transfer learning. This is a principled manner of learning a representation from raw data.

To summarize the 1 dimensional convolution filter for our simple example is defined as: 

\begin{align*}
\mathbf{C}(x|\alpha)&= x * \alpha \\
&= \sum_{t=1}^{5} x_t \alpha_t
\end{align*}

* $x$ is the input.
* $\alpha$ is the kernel.
* The output $x * \alpha$ is called a feature map and $*$ is the convolution operator or filter. This is the main difference between a vanilla neural network and a convolution neural network: We replace the matrix multiplication operator by the convolution operator.
* Depending on the task at hand we can have different types of convolution filters.
* Kernel size can be altered. In our example the kernel size is 5.
* Stride size can be altered. In our example we had no stride size however suppose that stride size was 1 and kernel size was 2, i.e., $\alpha = \left[\alpha_1, \alpha_2\right]$, then we would apply the kernel $\alpha$ at the start of the input, i.e., $\left[x_1, x_2\right] * \left[\alpha_1, \alpha_2\right]$, and move the kernel over the next area of the input, i.e., $\left[x_2, x_3\right] * \left[\alpha_1, \alpha_2\right]$, and so on and so forth until we arrive at a feature map that consists of 4 real values. This is called a valid convolution while a padded, i.e., say padded with zero values, convolution would give us a feature map that is the same size as the input, i.e., 5 real values in our example.
* We can apply an activation function to the feature maps such as ELU mentioned earlier.
* Finally we can summarize the information contained in feature maps by taking a maximum or average value over a defined portion of the feature map. For instance, if after using a valid convolution we arrive at a feature map of size 4 and then apply a max pooling operation with size 4 then we will be taking the maximum value of this feature map. The result is another feature map.

This automates feature engineering however introduces architecture engineering where different architectures consisting of various convolution filters, activation functions, batch normalization layers, dropout layers and pooling operators can be stacked together in a pipeline in order to learn a good representation of the raw data. One usually creates an ensemble of such architectures.

The goal behind convolutional autoencoders is to use convolution filters, activation functions, batch normalization layers, dropout layers and pooling operators to create an encoder function which will learn a good representation of our raw data. The decoder will also use a similar set of layers as the encoder to reconstruct the raw data with one exception: Instead of using a pooling operator it will use an upsampling operator. The basic idea behind the upsampling operator is to repeat an element a certain number of times say size 4: One can view this as the inverse operator to the pooling operator. The pooling operator is essentially a downsampling operator and the upsampling operator is simply the inverse of that in some sense.

### Results

The accuracy score for the MNIST classification task with a 1 dimensional convolutional autoencoder: 97.570000%.

## MNIST: Sequence to Sequence Autoencoders

Given our mortgage default example a potentially more useful deep learning architecture might be the Recurrent Neural Network (RNN), specifically their state of the art variant the Long Short Term Memory (LSTM) network. The goal is to explicitly take into account the sequential nature of the raw data.

\begin{align*}
&X \in \mathbb{R}^{N \times 28 \times 28} \\
&E: \mathbb{R}^{N \times 28 \times 28} \rightarrow \mathbb{R}^{N \times K} \\
&D: \mathbb{R}^{N \times K} \rightarrow \mathbb{R}^{N \times 28 \times 28}
\end{align*}

The gradients in a RNN depend on the parameter matrices defined for the model. Simply put these parameter matrices can end up being multiplied many times over and hence cause two major problems for learning: Exploding and vanishing gradients. If the spectral radius of the parameter matrices, i.e., the maximum absolute value of the eigenvalues of a matrix, is more than 1 then gradients can become large enough, i.e., explode in value, such that learning diverges and similarly if the spectral radius is less than 1 then gradients can become small, i.e., vanish in value, such that the next best transition for the parameters cannot be reliably calculated. Appropriate calculation of the gradient is important for estimating the optimal set of parameters that define a machine learning method and the LSTM network overcomes these problems in a vanilla RNN. We now define the LSTM network for 1 time step, i.e., 1 memory cell.

We calculate the value of the input gate, the value of the memory cell state at time period $t$ where $f(x)$ is some activation function and the value of the forget gate:

\begin{align*}
i_{t} &= \sigma(W_{i}x_{t} + U_{i}h_{t-1} + b_{i}) \\
\tilde{c_{t}} &= f(W_{c}x_{t} + U_{c}h_{t-1} + b_{c}) \\
f_{t} &= \sigma(W_{f}x_{t} + U_{f}h_{t-1} + b_{f})
\end{align*}

The forget gate controls the amount the LSTM remembers, i.e., the value of the memory cell state at time period $t-1$ where $\otimes$ is the hadamard product:

\begin{align*}
c_{t} = i_{t} \otimes \tilde{c_{t}} + f_{t} \otimes c_{t-1} 
\end{align*}

With the updated state of the memory cell we calculate the value of the outputs gate and finally the output value itself:

\begin{align*}
o_{t} &= \sigma(W_{o}x_{t} + U_{o}h_{t-1} + b_{o}) \\
h_{t} &= o_{t} \otimes f(c_{t})
\end{align*}

We can have a wide variety of LSTM architectures such as the convolutional LSTM where note that we replace the matrix multiplication operators in the input gate, the initial estimate $\tilde{c_{t}}$ of the memory cell state, the forget gate and the output gate by the convolution operator $*$:

\begin{align*}
i_{t} &= \sigma(W_{i} * x_{t} + U_{i} * h_{t-1} + b_{i}) \\
\tilde{c_{t}} &= f(W_{c} * x_{t} + U_{c} * h_{t-1} + b_{c}) \\
f_{t} &= \sigma(W_{f} * x_{t} + U_{f} * h_{t-1} + b_{f}) \\
c_{t} &= i_{t} \otimes \tilde{c_{t}} + f_{t} \otimes c_{t-1} \\ 
o_{t} &= \sigma(W_{o} * x_{t} + U_{o} * h_{t-1} + b_{o}) \\
h_{t} &= o_{t} \otimes f(c_{t})
\end{align*}

Another popular variant is the peephole LSTM where the gates are allowed to peep at the memory cell state:

\begin{align*}
i_{t} &= \sigma(W_{i}x_{t} + U_{i}h_{t-1} + V_{i}c_{t-1} + b_{i}) \\
\tilde{c_{t}} &= f(W_{c}x_{t} + U_{c}h_{t-1} + V_{c}c_{t-1} + b_{c}) \\
f_{t} &= \sigma(W_{f}x_{t} + U_{f}h_{t-1} + V_{f}c_{t-1} + b_{f}) \\
c_{t} &= i_{t} \otimes \tilde{c_{t}} + f_{t} \otimes c_{t-1} \\ 
o_{t} &= \sigma(W_{o}x_{t} + U_{o}h_{t-1} + V_{o}c_{t} + b_{o}) \\
h_{t} &= o_{t} \otimes f(c_{t})
\end{align*}

The goal for the sequence to sequence autoencoder is to create a representation of the raw data using a LSTM as an encoder. This representation will be a sequence of vectors say, $h_{1}, \dots, h_{T}$, learnt from a sequence of raw data vectors say, $x_{1}, \dots, x_{T}$. The final vector of the representation, $h_{T}$, is our encoded representation, also called a context vector. This context vector is repeated as many times as the length of the sequence such that it can be used as an input to a decoder which is yet another LSTM. The decoder LSTM will use this context vector to recontruct the sequence of raw data vectors, $\tilde{x_{1}}, \dots, \tilde{x_{T}}$. If the context vector is useful in the recontruction task then it can be further used for other tasks such as predicting default risk as given in our example.

### Results

The accuracy score for the MNIST classification task with a sequence to sequence autoencoder: 97.600000%.

## MNIST: Variational Autoencoders

We now combine Bayesian inference with deep learning by using variational inference to train a vanilla autoencoder. This moves us towards generative modelling which can have further use cases in semi-supervised learning. The other benefit of training using Bayesian inference is that we can be more robust to higher capacity deep learners, i.e., avoid overfitting. 

\begin{align*}
&X \in \mathbb{R}^{N \times 784} \\
&E: \mathbb{R}^{N \times 784} \rightarrow \mathbb{R}^{N \times K} \\
&D: \mathbb{R}^{N \times K} \rightarrow \mathbb{R}^{N \times 784}
\end{align*}

* Assume $X$ is our raw data while $Z$ is our learnt representation. 
* We have a prior belief on our learnt representation: 

\begin{align*}
p(Z)
\end{align*}

* The posterior distribution for our learnt representation is: 

\begin{align*}
p(Z|X)=\frac{p(X|Z)p(Z)}{p(X)}
\end{align*}

* The marginal likelihood, $p(X)$, is often intractable causing the posterior distribution, $p(Z|X)$, to be intractable:

\begin{align*}
p(X)=\int_{Z}p(X|Z)p(Z)dZ
\end{align*}

* We therefore need an approximate posterior distribution via variational inference that can deal with the intractability. This additionally also provides the benefit of dealing with large scale datasets as generally Markov Chain Monte Carlo (MCMC) methods are not well suited for large scale datasets. One might also consider Laplace approximation for the approximate posterior distribution however we will stick with variational inference as it allows a richer set of approximations compared to Laplace approximation. Laplace approximation simply amounts to finding the Maximum A Posteriori (MAP) estimate to an augmented likelihood optimization, taking the negative of the inverse of the Hessian at the MAP estimate to estimate the variance-covariance matrix and finally use the variance-covariance matrix with a multivariate Gaussian distribution or some other appropriate multivariate distribution.

* Assume that our approximate posterior distribution, which is also our probabilistic encoder, is given as:

\begin{align*}
q(Z|X)
\end{align*}

* Our probabilistic decoder is given by:

\begin{align*}
p(X|Z)
\end{align*}

* Given our setup above with regards to an encoder and a decoder let us now write down the optimization problem where $\theta$ are the generative model parameters while $\phi$ are the variational parameters:

\begin{align*}
\log{p(X)}= \underbrace{D_{KL}(q(Z|X)||p(Z|X))}_\text{Intractable as p(Z|X) is intractable} + \underbrace{\mathcal{L}(\theta, \phi|X)}_\text{Evidence Lower Bound or ELBO}
\end{align*}

* Note that $D_{KL}(q(Z|X)||p(Z|X))$ is non-negative therefore that makes the ELBO a lower bound on $\log{p(X)}$:

\begin{align*}
\log{p(X)}\geq \mathcal{L}(\theta, \phi|X) \quad \text{as} \quad D_{KL}(q(Z|X)||p(Z|X)) \geq 0
\end{align*}

* Therefore we can alter our optimization problem to look only at the ELBO:

\begin{align*}
\mathcal{L}(\theta, \phi|X) &= \mathbb{E}_{q(Z|X)}\left[\log{p(X,Z)} - \log{q(Z|X)}\right] \\
&= \mathbb{E}_{q(Z|X)}\left[\underbrace{\log{p(X|Z)}}_\text{Reconstruction error} + \log{p(Z)} - \log{q(Z|X)}\right] \\
&= \mathbb{E}_{q(Z|X)}\left[\underbrace{\log{p(X|Z)}}_\text{Reconstruction error} - \underbrace{D_{KL}(q(Z|X)||p(Z))}_\text{Regularization}\right] \\
&= \int_{Z} \left[\log{p(X|Z)} - D_{KL}(q(Z|X)||p(Z))\right] q(Z|X) dZ
\end{align*}

* The above integration problem can be solved via Monte Carlo integration as $D_{KL}(q(Z|X)||p(Z))$ is not intractable. Assuming that the probabilistic encoder $q(Z|X)$ is a multivariate Gaussian with a diagonal variance-covariance matrix we use the reparameterization trick to sample from this distribution say $M$ times in order to calculate the expectation term in the ELBO optimization problem. The reparameterization trick in this particular case amounts to sampling $M$ times from the standard Gaussian distribution, multiplying the samples by $\sigma$ and adding $\mu$ to the samples.  

* $\mu$ is our learnt representation used for the reconstruction of the raw data. If the learnt representation is useful it can then be used for other tasks as well.

* This is a powerful manner of combining Bayesian inference with deep learning. Variational inference used in this manner can be applied to various deep learning architectures and has further links with the Generative Adversarial Network (GAN). We explore the use of adversarial learning in representation learning in another repo/paper.

### Results

The accuracy score for the MNIST classification task with a variational autoencoder: 96.520000%.

## MNIST: 2 Dimensional Convolutional Autoencoders

For 2 dimensional convolution filters the idea is similar as for the 1 dimensional convolution filters. We will stick to our previously mentioned banking example to illustrate this point.

\begin{align*}
&X \in \mathbb{R}^{N \times 28 \times 28} \\
&E: \mathbb{R}^{N \times 28 \times 28} \rightarrow \mathbb{R}^{N \times K} \\
&D: \mathbb{R}^{N \times K} \rightarrow \mathbb{R}^{N \times 28 \times 28}
\end{align*}

\begin{align*}
x = 
\begin{array}
{l}
\text{Period 1} \\ \text{Period 2} \\ \text{Period 3} \\ \text{Period 4} \\ \text{Period 5}
\end{array}
    \left[
    \begin{array}
    {ccc}
    \$0 & \$0 & \$0 \\
    \$0 & \$200 & \$0 \\
    \$100 & \$0 & \$0 \\
    \$0 & \$0 & \$300 \\
    \$0 & \$0 & \$0
    \end{array}
    \right]
\end{align*}

In the 2 dimensional tensor of raw transactions data now we have 5 historical time periods, i.e., the rows, and 3 different transaction types, i.e., the columns. We will use a kernel, $\alpha \in \mathbb{R}^{2\times3}$, to extract useful features from the raw data. The choice of such a kernel means that we are interested in finding a feature map across all 3 transaction types and 2 historical time periods. We will use a stride length of 1 and a valid convolution to extract features over different patches of the raw data. The following will illustrate this point where $x_{\text{patch}} \subset x$:

\begin{align*}
\alpha &=
    \left[
    \begin{array}
    {ccc}
    \alpha_{1,1} & \alpha_{1,2} & \alpha_{1,3} \\
    \alpha_{2,1} & \alpha_{2,2} & \alpha_{2,3}
    \end{array}
    \right] \\
x_{\text{patch}} &= 
    \left[
    \begin{array}
    {ccc}
    \$0 & \$0 & \$0 \\
    \$0 & \$200 & \$0
    \end{array}
    \right] \\
\mathbf{C}(x=x_{\text{patch}}|\alpha) &= x * \alpha \\
&= \sum_{t=1}^{2} \sum_{k=1}^{3} x_{t,k} \alpha_{t,k}
\end{align*}

The principles and ideas apply to 2 dimensional convolution filters as they do for their 1 dimensional counterparts there we will not repeat them here.

### Results

The accuracy score for the MNIST classification task with a 2 dimensional convolutional autoencoder: 98.860000%.

## Insurance: No Autoencoders

We now proceed to run the insurance model without any handcrafted or deep learning based feature engineering.

### Results

The AUROC score for the insurance classification task without autoencoders: 92.206261%.

## Insurance: PCA

We now proceed to run the insurance model without any handcrafted or deep learning based feature engineering however with a PCA filter that picks the number of components that explain $99\%$ of the variation.

### Results

The AUROC score for the insurance classification task with PCA: 91.128859%.

## Insurance: Handcrafted Features

In this case we have created some handcrafted features which we believe provide a useful representation of the raw data for the insurance model.

### Results

The AUROC score for the insurance classification task with handcrafted features: 93.610635%.

## Insurance: Handcrafted Features and PCA

In this case we have created some handcrafted features which we believe provide a useful representation of the raw data for the insurance model. We also use a PCA filter.

### Results

The AUROC score for the insurance classification task with handcrafted features and PCA: 93.160377%.

## Insurance: Vanilla Autoencoders

In this case we use vanilla autoencoders to learn a good representation of the raw data such that we can obtain an uplift, primarily in terms of AUROC, for the supervised learning task.

\begin{align*}
&X \in \mathbb{R}^{N \times 12} \\
&E: \mathbb{R}^{N \times 12} \rightarrow \mathbb{R}^{N \times K} \\
&D: \mathbb{R}^{N \times K} \rightarrow \mathbb{R}^{N \times 12}
\end{align*}

### Results

The AUROC score for the insurance classification task with an autoencoder: 93.932247%.

## Insurance: Denoising Autoencoders

In this case we use denoising autoencoders to learn a good representation of the raw data such that we can obtain an uplift, primarily in terms of AUROC, for the supervised learning task.

\begin{align*}
&X \in \mathbb{R}^{N \times 12} \\
&E: \mathbb{R}^{N \times 12} \rightarrow \mathbb{R}^{N \times K} \\
&D: \mathbb{R}^{N \times K} \rightarrow \mathbb{R}^{N \times 12}
\end{align*}

### Results

The AUROC score for the insurance classification task with a denoising autoencoder: 93.712479%.

## Insurance: Sequence to Sequence Autoencoders

In this case we use sequence to sequence autoencoders, taking into account the time series nature, i.e., sequential nature, of the raw transactions data, to learn a good representation of the raw data such that we can obtain an uplift, primarily in terms of AUROC, for the supervised learning task.

\begin{align*}
&X \in \mathbb{R}^{N \times 4 \times 3} \\
&E: \mathbb{R}^{N \times 4 \times 3} \rightarrow \mathbb{R}^{N \times K} \\
&D: \mathbb{R}^{N \times K} \rightarrow \mathbb{R}^{N \times 4 \times 3}
\end{align*}

### Results

The AUROC score for the insurance classification task with a sequence to sequence autoencoder: 91.418310%.

## Insurance: 1 Dimensional Convolutional Autoencoders

In this case we use 1 dimensional convolutional autoencoders to learn a good representation of the raw data such that we can obtain an uplift, primarily in terms of AUROC, for the supervised learning task.

\begin{align*}
&X \in \mathbb{R}^{N \times 4 \times 3} \\
&E: \mathbb{R}^{N \times 4 \times 3} \rightarrow \mathbb{R}^{N \times K} \\
&D: \mathbb{R}^{N \times K} \rightarrow \mathbb{R}^{N \times 4 \times 3}
\end{align*}

### Results

The AUROC score for the insurance classification task with a 1 dimensional convolutional autoencoder: 91.509434%.

## Insurance: 2 Dimensional Convolutional Autoencoders

In this case we use 2 dimensional convolutional autoencoders to learn a good representation of the raw data such that we can obtain an uplift, primarily in terms of AUROC, for the supervised learning task.

\begin{align*}
&X \in \mathbb{R}^{N \times 4 \times 3} \\
&E: \mathbb{R}^{N \times 4 \times 3} \rightarrow \mathbb{R}^{N \times K} \\
&D: \mathbb{R}^{N \times K} \rightarrow \mathbb{R}^{N \times 4 \times 3}
\end{align*}

### Results

The AUROC score for the insurance classification task with a 2 dimensional convolutional autoencoder: 92.645798%.

## Insurance: Variational Autoencoders

In this case we use variational autoencoders to learn a good representation of the raw data such that we can obtain an uplift, primarily in terms of AUROC, for the supervised learning task.

\begin{align*}
&X \in \mathbb{R}^{N \times 12} \\
&E: \mathbb{R}^{N \times 12} \rightarrow \mathbb{R}^{N \times K} \\
&D: \mathbb{R}^{N \times K} \rightarrow \mathbb{R}^{N \times 12}
\end{align*}

### Results

The AUROC score for the insurance classification task with a variational autoencoder: 90.871569%.

## Conclusion

We have shown how to use deep learning and Bayesian inference to learn a good representation of raw data $X$, i.e., 1 or 2 or perhaps more dimensional tensors per unit of analysis, that can then perhaps be used for supervised learning tasks in the domain of computer vision and insurance. This moves us away from manual handcrafted feature engineering towards automatic feature engineering, i.e., representation learning. This does introduce architecture engineering however that can be automated as well perhaps by the use of genetic algorithms or reinforcement learning - a topic for another paper perhaps.

Finally, I would like to emphasize that the same code used for solving the computer vision task was used to solve the insurance task: In both tasks automatic feature engineering via deep learning had the best performance despite the fact that we were not explicitly looking for the best state of the art architecture possible.

## References

1. Goodfellow, I., Bengio, Y. and Courville A. (2016). Deep Learning (MIT Press).
2. Geron, A. (2017). Hands-On Machine Learning with Scikit-Learn & Tensorflow (O'Reilly).
3. Kingma, D. P., and Welling M. (2014). Auto-Encoding Variational Bayes (https://arxiv.org/abs/1312.6114).
4. http://scikit-learn.org/stable/#
5. https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
6. https://stackoverflow.com/questions/42177658/how-to-switch-backend-with-keras-from-tensorflow-to-theano
7. https://blog.keras.io/building-autoencoders-in-keras.html
8. https://keras.io
9. https://www.cs.cornell.edu/courses/cs1114/2013sp/sections/S06_convolution.pdf
10. http://deeplearning.net/tutorial/lstm.html 
