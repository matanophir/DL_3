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
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======

    # ========================
    return hypers

part3_q1 = r"""
**Your answer:**


"""

part3_q2 = r"""
**Your answer:**


"""

part3_q3 = r"""
**Your answer:**



"""



PART3_CUSTOM_DATA_URL = None


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
