# Journal
## 2023-09-17: Let's get started
First step: Read (skim...) the paper [Inversion-based Style Transfer with Diffusion Models](https://arxiv.org/abs/2211.13203). Get a basic grasp of what the main ideas are _and_ what the actual compute pipeline looks like. Questions to pay attention to:

* Is the transfer process a "single" pass like in the initial style transfer paper using VGG and the Gram matrix?
* Or do you create pre-trained models for each target style you're interested in?
* If so, how much effort is this? And how much effort is it to then apply that pre-trained style?
