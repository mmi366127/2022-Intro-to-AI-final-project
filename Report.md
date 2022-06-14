## AI FInal project report

### Abstract

This study is to figure out a way to transfer human voice without losing the linguistic content and having grotesque electronical sounds. It was inspired by our roommate, we want technology to break the limitation of human body and he could be granted with the opportunity to become a Virtual YouTuber without being beaten by the cracky voice. We used the Random CNN model and TraVelGAN model to generate the results and compare the two models. And perfrom the ... experiment. It was concluded that 

### Introduction

"Technology like art is a soaring exercise of the human imagination.” – Daniel Bell

Inspired by one of our roommates, who wish to become a Virtual YouTuber but unfortunately hates his own voice, we wish to develop something that can help make his dream come true. Thus, we have come up with the voice transfermation aiming to produce natural voices without losing the semantic content. 

GAN and TraVeLGAN are $2$ existing approaches to do the task, but there are still electronical-like voices with them, we want to improve it by ... and 

### Related Work

We know that image to image style transfer can be done by GAN algorithm(Generative Adversial Network) and Convolutional Networks [1],[2],[3]. For voices, we found that we can apply the image technique, we just need to find a way to transfer the voice into image-like two dimensional graph and use TraVeLGAN to perform voice transfer or audio style transfer.[4] 



### Methodology

##### Random CNN



##### TraVeLGAN

TraVeLGAN stands for Transformation Vector Learning Gnerative Adversial Network, using a Siamese network $S$ to transfer image to latent vectors, a generator $G$ and discriminator $D$ trains to preserve the vector arithmetic between points in the latent space of $S$. The transformation must satisfiy the following equation, which let $G(X)$ be the generated image of image $X$, $S(X)$ be the vector encoding of $X$, and $A_1$, $A_2$ be two images in the same domain, then $S(A_1) - S(A_2) = S(G(A_1)) - S(G(A_2))$. This can preserve semantic information, which is the content of the speech. 

The above condition is the ideal condition, but nothing is that perfect, id est, the leftside cannot be exactly equal to the rightside, thus we will define the loss function as the consine similarity $+$ Euclidean distance between the two transformation vectors. Thus the loss function is the following.

$$
L_{(G,S)}=E_{(a_{\frac L2, 1}, a_{\frac L2 , 2})}[cosine \ similarity(t_{12}, t_{12}' ) + \lVert  t_{12} - t_{12}'  \rVert_2^2] 
$$

$$
where \ \ a_{\frac L2 , 1} \neq a_{\frac L2 , 2}
$$

$$
t_{ij} = S(a_{\frac L2,i}) - S(a_{\frac L2, j})
$$

$$
t_{ij}'= S(G(a_{\frac L2, i})) - S(G(a_{\frac L2, j}))
$$

The following is our training pipeline:

1. Cut the source image $LxH$ in half, getting $2$ $\frac{L}{2}xH$ spectrograms.

2. Input the $2$ halves into generator $G$ and obtain the translated pair $t$.

3. Input $t$ and the target $LxH$ to the discriminator $D$, making them different and thus will allow adversial training.

Translate vary length of spectrogram :(suppose the shape of the spectrogram is $XxH$)

1. Split the $XxH$ input into $\frac{L}{2} xH$ chunks.

2. Input each chunk into the generator $G$.

3. Concatenate the segments into the original shape.



### Experiments



### References

[1]Alec Radford, Luke Metz, and Soumith Chintala. Unsupervised representation learning with deep convolutional generative adversarial networks. In Yoshua Bengio and Yann LeCun, editors, *4th International Conference on Learning Representations, ICLR 2016, San Juan, Puerto Rico, May 2-4, 2016, Conference Track Proceedings*, 2016.

[2]ANDREINI, Paolo, et al. Image generation by GAN and style transfer for agar plate image segmentation. *Computer methods and programs in biomedicine*, 2020, 184: 105268.

[3]GATYS, Leon A.; ECKER, Alexander S.; BETHGE, Matthias. Image style transfer using convolutional neural networks. In: *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2016. p. 2414-2423.

[4]PASINI, Marco. Melgan-vc: Voice conversion and audio style transfer on arbitrarily long samples using spectrograms. arXiv preprint arXiv:1910.03713, 2019.



## Video

### Introduction

在介紹我們的實驗之前，讓我們先為各位科普一下什麼是 VTuber.

Virtual YouTuber, 簡稱 VTuber, 以虛擬人物形象在影音平台上上傳影片或直播。雖然不必露臉，但聲音「表情」變得更加重要，而聲音表情又與個人「音質」高度相關。

藉由訓練，我們能改變說話的語調，但音質是與生俱來，無法改變的。所以，藉由「Voice Transformation」，我們希望對自己聲音沒有信心的同胞也能打破先天限制，為自己的夢想奮鬥而發出最耀眼的光芒。

### Related Work

目前在網路上已經找到用 Random CNN 與 TraVeLGAN 的 model, 但兩種方法跑出來的結果仍有電子音的感覺，故我們想藉由兩種模型的改良建出較自然且內容相同的 model.

### Dataset / Platform

自己收集：正規語言概論、計算機組織上課影片

CMU Arctic Dataset

### Baseline

Random CNN

TraVeLGAN

### Main Approach

Describe the algorithm

### Evaluation Metric



### Results & Analysis



### Future Work



