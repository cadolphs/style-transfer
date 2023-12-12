# style-transfer
Experimenting with diffusion for style transfer

In this project, I'll want to grab one of the recent papers on using diffusion (maybe even latent diffusion) for _style transfer_. My previous experience with style transfer was based on techniques around matching up the "content" and "style" layers of particular CNN architectures such as VGG. These approaches are good at matching the _edge style_ of a given target image but fall short in several important ways. I won't go into much detail here. Suffice it to say that there was room for improvement.

Stable Diffusion is _great_ at replicating the styles of artists, but techniques like `img2img` don't work well for style transfer because they tend to be too destructive on the original image's content unless supplemented with just the right text to condition.

Check out the `journal.md` file for my thoughts and progress. At this point, the repo is in a place where you can clone it and run it for yourself (check the journal for up-to-date setup instructions).