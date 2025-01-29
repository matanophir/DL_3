r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers["batch_size"] = 64
    hypers["seq_len"] = 64
    hypers["h_dim"] = 256
    hypers["n_layers"] = 2
    hypers["dropout"] = 0.1
    hypers["learn_rate"] = 0.001
    hypers["lr_sched_factor"] = 0.5
    hypers["lr_sched_patience"] = 5
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = "FINAL ACT."
    temperature = 0.5
    # ========================
    return start_seq, temperature


part1_q1 = r"""
We split the corpus into sequences instead of training on the whole text for the same reason we split to batches up until now- memory and generalization considerations.\
it would be memory intensive to load and train on the entire dataset at once (need to retain the data and the calculation graph), and it was shown that mini-batches can help the model generalize better.\ 
Furthermore, BPTT on the entire text would be very slow and possibly cause vanishing gradients even when using a more advanced model like GRU.
"""

part1_q2 = r"""
**Your answer:**
Sequence length is limited but the context is retained in the hidden state. The hidden state can accumulate context from previous timestamps regardless of the sequence length, and be used to generate text with that context.\

"""

part1_q3 = r"""
**Your answer:**
In our implementation we orginized the batches is such a way that the sequences across batches are continuos (i.e sequence i in batch k is followed by sequence i in batch k+1).\
This provides the model coherent context to learn from, hopefully improving the quality of the generated text.\
"""

part1_q4 = r"""
**Your answer:**
1.A drawback of the regular softmax (when T=1) is that it can generate very diffuse (more uniform) distributions if the score values are very similar. When sampling, we would prefer to control the distributions and make them less uniform to increase the chance of sampling the char(s) with the highest scores compared to the others by using T<1.

2. When the temperature is high we will get a more uniform distribution, thus increasing the chances of sampling less probable chars giving the model a more 'random' behavior.\

3. When the temperature is low we will get a less uniform distribution, thus increasing the chances of sampling the most probable chars giving the model a more 'deterministic' behavior.\

"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = 'https://github.com/AviaAvraham1/TempDatasets/raw/refs/heads/main/George_W_Bush2.zip'


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers["batch_size"] = 96
    hypers["h_dim"] = 512
    hypers["z_dim"] = 64
    hypers["x_sigma2"] = 0.001
    hypers["learn_rate"] = 0.0002
    hypers["betas"] = (0.9, 0.999)
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**
 the hyperparamater $\sigma^2$ is used in the loss function to balance the reconstruction loss and the kl loss.\
 larger $\sigma^2$ decreases the weight of the reconstruction loss, which allows for more flexibility in reconstruction as the loss tolerates larger differences between x and x_recon. can also lead to blurred meaningless reconstructions.\
 smaller $\sigma^2$ amplifies the weight of the reconstruction loss, making the model to reconstruct x as accurately as possible. may lead to overfitting, where the model memorizes the training data rather than learning a meaningful latent representation.

"""

part2_q2 = r"""
**Your answer:**
1. reconstruction loss is incurred when the reconstructed x is not similar to the original x (in our case L2 wise).\
    KL loss is incurred when the posterior distribution $q(z|x)$ varies from the prior $p(z)$ which is assumed to be $~N(0,1)$. keeps the variance and mean of the posterior from growing too large while $-log\sigma^2$ encourges the posterior to have non-zero variance.\

2. The KL divergence loss term causes the latent space distribution to be similar to $p(z) ~ N(0,1)$. it encourges the latent space distribution to be centered around $\mu = 0$ with not too large but not zero variance. \

3. after we forces the latent space distribution to be a 'dense' gaussian we can easily sample from this distribution ($N(0,1)$) and get a meaningful latent representation that most likely can be decoded to what we are trying to produce.\
the distribution is dense because the KL term encourages *all* the latent representation to seem like they were taken from $N(0,1)$.


"""

part2_q3 = r"""
**Your answer:**
We want to maximize p(X) because the only thing we know about our data assumed distribution is that our data came from it, so p(X) should be large. we're maximizing this term indirectly by maximizing it's lower bound.


"""

part2_q4 = r"""
**Your answer:**
we model the log var for a few possible reason:\
- numerical stability- very small or large values become manageable when working in log space.\
- simpler to work with KL divergence since there the variance appears in its log form.\ 
- this way we can enforce the variance to be non-negative while still enabling our model to produce negative values.
"""

# Part 3 answers
def part3_gan_hyperparams():
    hypers = dict(
        batch_size=0, z_dim=0, learn_rate=0.0, 
        discriminator_optimizer=dict(type='', lr=0.0),
        generator_optimizer=dict(type='', lr=0.0, betas=(0.0, 0.0)),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers["batch_size"] = 32
    hypers["z_dim"] = 256
    hypers['data_label'] = 1
    hypers['label_noise'] = 0.3

    hypers["discriminator_optimizer"]['type'] = 'SGD' # tip 9 
    hypers["discriminator_optimizer"]['lr'] = 0.0005

    hypers["generator_optimizer"]['type'] = 'Adam'
    hypers["generator_optimizer"]['lr'] = 0.002
    hypers["generator_optimizer"]['betas'] = (0.9, 0.999)

    # hypers['batch_size'] = 32
    # hypers['z_dim'] = 128
    # hypers['data_label'] = 1
    # hypers['label_noise'] = 0.3
    # hypers['discriminator_optimizer']['type'] = 'Adam'
    # hypers['discriminator_optimizer']['weight_decay'] = 0.02
    # hypers['discriminator_optimizer']['betas'] = (0.5, 0.999)
    # hypers['discriminator_optimizer']['lr'] = 0.0002
    # hypers['generator_optimizer']['type'] = 'Adam'
    # hypers['generator_optimizer']['lr'] = 0.0002
    # hypers['generator_optimizer']['weight_decay'] = 0.02
    # hypers['generator_optimizer']['betas'] = (0.5, 0.999)

    # hypers = dict(
    #     batch_size=64, z_dim=64,
    #     data_label=1, label_noise=0.3,
    #     discriminator_optimizer=dict(
    #         type='Adam',
    #         weight_decay=0.02,
    #         betas=(0.5, 0.999),
    #         lr=0.0002,
    #     ),
    #     generator_optimizer=dict(
    #         type='Adam',
    #         weight_decay=0.02,
    #         betas=(0.5, 0.999),
    #         lr=0.0002,
    #     ),
    # )
    # ========================
    return hypers

part3_q1 = r"""
**Your answer:**
When training a GAN we actually train two models , the generator and the discriminator. The generator is trained to generate data that is similar to the real data, while the discriminator is trained to distinguish between real and fake data.\
We want the training of these two models to be separated, so we won't encounter any unintended behavior related to the computational graph and the flow of gradients like unnecessary gradient tracking or even updates if not handled correctly.\

when training the discriminator we dont want the backpropagation to calculate the generator's gradients (nor ofc update it's weights), so we detach the generator's output from the computational graph effectively 'cutting' the connection between the generator and the discriminator.\
when training the generator we need the flow of gradient to pass through the discriminator and the generated data to update the generator's weights, so we don't detach the generator's output from the computational graph.\

ofc when sampling from the generator at inference time we dont need the computational overhead of the graph so we disable the tracking to begin with.\

"""

part3_q2 = r"""
**Your answer:**
1. We should not decide to stop training based solely on the generator loss because it it not a good indicator of it's performance.\
the generator and discriminator are battling each other during training time, so the generator loss may be low at one point because the discriminator is weak, but the generator isn't actually generating good data.\
we should stop the training based on the given results.\

2. if we look at the discriminator loss formula:
$$
- \mathbb{E} _{\bb{x} \sim p(\bb{X}) } \log \Delta _{\bb{\delta}}(\bb{x})  \, - \,
\mathbb{E} _{\bb{z} \sim p(\bb{Z}) } \log (1-\Delta _{\bb{\delta}}(\Psi _{\bb{\gamma}} (\bb{z}) )).
$$

the first term is the loss of the discriminator on real data, and the second is the loss on generated data which relates to the generator loss.\
if the discriminator loss is constant and the generator loss decreases it means that the discriminator loss on the on the generated data is increasing, hence the real data loss is decreasing.\ 
this means that the generator is improving and surpassing the detecting capabilities of the discriminator, and the discriminator is also getting better at discerning real data. 

"""

part3_q3 = r"""
**Your answer:**
We think the main difference is that the gan outputs fuzzier images with more disruptions.
the thing that may cause it is the formulation of the loss functions. in the VAE we have reconstruction loss that makes the generated images to be similar to the real images, so any noise in the generated images will be penalized.
in the GAN we don't have this term, so the generator can generate images with more noise and disruptions so long as the discriminator doesn't deem them as fake. for example the generator can generate a face that suffices the discriminator and have more 'creativity' room in generating the background


"""



PART3_CUSTOM_DATA_URL = 'https://github.com/AviaAvraham1/TempDatasets/raw/refs/heads/main/George_W_Bush2.zip'


def part4_transformer_encoder_hyperparams():
    hypers = dict(
        embed_dim = 0, 
        num_heads = 0,
        num_layers = 0,
        hidden_dim = 0,
        window_size = 0,
        droupout = 0.0,
        lr=0.0,
    )

    # TODO: Tweak the hyperparameters to train the transformer encoder.
    # ====== YOUR CODE: ======
    
    # ========================
    return hypers




part3_q1 = r"""
**Your answer:**

"""

part3_q2 = r"""
**Your answer:**


"""


part4_q1 = r"""
**Your answer:**


"""

part4_q2 = r"""
**Your answer:**


"""


part4_q3= r"""
**Your answer:**


"""

part4_q4 = r"""
**Your answer:**


"""

part4_q5 = r"""
**Your answer:**


"""


# ==============
