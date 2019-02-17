# Super Resolution GAN TreeHacks 2019

### Description

### Timeline

- Friday, Feb 15, 10:00pm
  - Plan project
  - Make timeline
- Saturday, Feb 16, 1:00am
  - Download and pre-process data (Kian)
  - SRGAN (Tyler, Grant)
  - Research realtime, low res video (Priyank)
- Saturday, Feb 16, 1:00pm
  - SRGAN implementation and integration with YouTube dataset
  - RCAN implementation
- Saturday, Feb 16, 5:00pm
  - Have a generated upscaled image

### Resources

##### Papers
- [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/pdf/1609.04802.pdf)
  - They set a new state of the art for image SR with high upscaling factors (4×) as measured by PSNR and structural similarity (SSIM) with 16 blocks deep ResNet (SRResNet) optimized for MSE.
  - They propose SRGAN which is a GAN-based network optimized for a new perceptual loss. Here they replace the MSE-based content loss with a loss calculated on feature maps of the VGG network, which are more invariant to changes in pixel space.
  - They confirm with an extensive mean opinion score (MOS) test on images from three public benchmark datasets that SRGAN is the new state of the art, by a large margin, for the estimation of photo-realistic SR images with high upscaling factors (4×).
- [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)
- [Efficient Super Resolution For Large-Scale Images Using Attentional GAN](https://arxiv.org/pdf/1812.04821.pdf)
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf)
- [Generative Adversarial Nets](https://arxiv.org/pdf/1406.2661.pdf)
- [Self-Attention Generative Adversarial Networks](https://arxiv.org/pdf/1805.08318.pdf)
- [Low-Complexity Single-Image Super-Resolution based on Nonnegative Neighbor Embedding](http://people.rennes.inria.fr/Aline.Roumy/publi/12bmvc_Bevilacqua_lowComplexitySR.pdf)
- [tempoGAN: A Temporally Coherent, Volumetric GAN for Super-resolution Fluid Flow](https://arxiv.org/pdf/1801.09710.pdf)
- [Are GANs Created Equal? A Large-Scale Study](https://arxiv.org/pdf/1711.10337.pdf)
  - A study of GAN evaluation metrics
  - We provide a fair and comprehensive comparison of the state-of-the-art GANs, and empirically demonstrate that nearly all of them can reach similar values of FID, given a high enough computational budget.
  - We provide strong empirical evidence that to compare GANs it is necessary to report a summary of distribution of results, rather than the best result achieved, due to the randomness of the optimization process and model instability.
- [Going deeper with convolutions](https://arxiv.org/pdf/1409.4842v1.pdf)
- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf)

sparab22@stanford.edu



Spatiotemporal 3D CNN

W * H * C * T

W: Width
H: Height
C: Channels
T: Time Window
