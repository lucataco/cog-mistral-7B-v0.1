# mistralai/Mistral-7B-v0.1 Cog model

This is an implementation of the [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, download the pre-trained weights:

    cog run script/download-weights

Then, you can run predictions:

    og predict -i prompt="Explain in a short paragraph quantu field theory to a high school student"
