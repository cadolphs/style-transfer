# Journal
## 2023-10-20: Diving into the code
The repo linked in the paper is a bit tough to dive into because the true logic seems well hidden in the configuration files. I think I'll have to pry this apart piece by piece. 
So, what's a good way of defining milestones here? Let's do a few observations:

* Original repo uses OmegaConf and Pytorch Lightning. Do these things play nicely with modal.com?
* 

## 2023-10-03: Reading the paper
* The paper is fine, but the real deal will be in the code. So, next step: Clone the repo and dig around.
* Training has to be done per style image (takes 20 minutes on consumer hardware o_O)
* Inference can be done "at the speed of normal diffusion model inference" because it's just conditioned generation. That should be pretty fast then.

The main idea seems to be this: 
* In training, style and content image are set to the same thing.
* In a forward pass, we generate the CLIP Image embedding for the image and turn that via a learned NN into a text embedding.
* We also apply stochastic inversion to the input image to obtain some predicted noise. This predicted noise serves as the starting point for image synthesis.
* The optimization is about accurately predicting the noise at each time step of the diffusion process where that process is conditioned on the text embedding and where the starting noise is the predicted noise.

Apparently, the stochastic inversion process is necessary to preserve the content of the content image. The paper isn't clear on whether the noise prediction in this step is _also_ already conditioned on the text embedding.

So, questions to figure out from the code:
- What are we exactly minimizing?
- How is the stochastic inversion applied?

## 2023-09-17: Let's get started
First step: Read (skim...) the paper [Inversion-based Style Transfer with Diffusion Models](https://arxiv.org/abs/2211.13203). Get a basic grasp of what the main ideas are _and_ what the actual compute pipeline looks like. Questions to pay attention to:

* Is the transfer process a "single" pass like in the initial style transfer paper using VGG and the Gram matrix?
* Or do you create pre-trained models for each target style you're interested in?
* If so, how much effort is this? And how much effort is it to then apply that pre-trained style?
