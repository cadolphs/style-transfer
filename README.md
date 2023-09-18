# style-transfer
Experimenting with diffusion for style transfer

In this project, I'll want to grab one of the recent papers on using diffusion (maybe even latent diffusion) for _style transfer_. My previous experience with style transfer was based on techniques around matching up the "content" and "style" layers of particular CNN architectures such as VGG. These approaches are good at matching the _edge style_ of a given target image but fall short in several important ways. I won't go into much detail here. Suffice it to say that there was room for improvement.

Stable Diffusion is _great_ at replicating the styles of artists, but techniques like `img2img` don't work well for style transfer because they tend to be too destructive on the original image's content unless supplemented with just the right text to condition.

To begin, I'll experiment with the paper "Inversion-Based Style Transfer with Diffusion Models", https://arxiv.org/abs/2211.13203. I'll use this project to a) get more in-depth experience with diffusion models beyond having played around with Stable Diffusion and b) learn a bit about serverless computing. In particular, I'm trying out Modal (modal.com) to run things on the cloud with minimal setup. Between my actual job and three young kids, I want to spend my time solving the actual problems, not setting up AWS instances (or Paperspace notebooks or whatever).

Check out the `journal.md` file for my thoughts and progress.
