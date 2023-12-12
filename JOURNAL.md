# Journal
## 2023-12-11 Moving time
Instead of journaling the nitty gritty here, I'll just use GitHub issues in the repo directly to explain what's going on.

TL;DR, the ArtFusion paper and repo are great and I already got it running. You can easily try it for yourself by just cloning this repo and getting yourself set up with [modal](modal.com) (free!). Once you have that installed and got a token, 
you can create your own style images with 

`modal run simple_script.py --content-file-path [CONTENT_IMAGE] --style-file-path [STYLE_IMAGE]`

and get the transformed image spat out in `output.png`. Obviously, more bells and whistles will follow.

## 2023-11-03: It runs. Sort of.
Okay, adding the missing `__init__.py` did the trick. Then there were issues with newer CUDA versions or something about the device not being correct that required a quick fix in the InST repo.

Now it seems to be running and training. Before I do that for a full run, though, I'll want to sort out the saving of model artifacts.

Well. Always some issues with timeouts, and `trainer.test()` not working. And the whole Omegaconf thing makes it really hard to know where I could set parameters so the test run is short and sweet.
Found the ArtFusion Style Transfer paper, which might be a better bet, especially since it's already pretrained and does "arbitary" style transfer, i.e., doesn't need to get re-trained on each image. Not sure if it would achieve 
similar quality, but at least I might be able to get it to run more directly.

## 2023-11-02: Milestone update
1. ✅I can run modal with "my own" git repo cloned.
2. ✅I can build a modal container with the Stable Diffusion 1.4 checkpoint (as originally required by InST) downloaded.
3. ✅My code can put some output file (hardcoded and simple) into a persistent volume (those will become the checkpoints)
4. With hardcoded params that should lead to _very_ fast training, I can run the original script with original settings (other than low number of steps / epochs) and get the output stored.
5. I can then run _inference_ on the stored output.

In preparation for point 4, I spent a lot of time getting the image set up correctly with the conda environment. I'm not sure I've got it 100% right. I only will know that once I've actually run step 4. But I can get all the way to the point that I can import the `main.py` from the `InST` repo.

Now I guess instead of running things from within Python, which would require me to rejig the things in `main.py` to deal with the command line parsing of the original script, I could just 
_run_ the script. Maybe that's what I'll try next. ✅

Now I think step 4 in the milestone list should be broken down a bit more, just so I can sneak things in when I have time. First, I'll need a target style image. Next, I'll need to figure 
out where the code stores the output checkpoints and how I can get my hands on that.

First, I can just copy over a file and hardcode its location in the _image_. Long-term I'll want it to be better.
Next, I just use `os.system("python3 main.py ...)` to invoke the script. Trying that right now, and the issue is that the `pip` installed git repos aren't importable because 
their modules don't contain `__init__` files. Looks like I have to fix that myself...

## 2023-10-29: Few more thoughts
Been mulling over the best approach for this, going through some examples on modal.com. It seems rather than retro-fitting or re-writing the whole of that InST repo to support modal, I could have modal in my own project and just as part of the container image do a git clone of the InST repo (or my own fork thereof). The idea being that I can do a few patches where necessary (e.g. having a volume where the trained style checkpoints would be stored) instead of trying to make everything work with it.

There are examples of this on the modal repo, too. Because the InST model is something hand-crafted, I can't just pull in a ready-made model and pipeline from HuggingFace.

With that, then, the first milestone would be to get the InST repo "more or less as is" running via modal. 
Turning that into user stories / tasks, I could split that up even further. And because I'm new at all this, I make the milestones deliberately small.

1. I can run modal with "my own" git repo cloned.
2. I can build a modal container with the Stable Diffusion 1.4 checkpoint (as originally required by InST) downloaded.
3. My code can put some output file (hardcoded and simple) into a persistent volume (those will become the checkpoints)
4. With hardcoded params that should lead to _very_ fast training, I can run the original script with original settings (other than low number of steps / epochs) and get the output stored.
5. I can then run _inference_ on the stored output.

That would be a first milestone where, even with hardcoded stuff and everything too simple to make sense, I'd be invoking the whole pipeline end to end. Of course it'd be lacking a web gui and better parameters. That'll come next.

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
