import torch
from argparse import Namespace

from .diffpure_sde import RevGuidedDiffusion


args = Namespace(
    config="cifar10.yml",
    data_seed=456,
    seed=123,
    exp="./exp_results",
    verbose="info",
    image_folder="./exp_results/cifar10-robust_adv-5-eps0.031373-128x1-bm0-t0-end1e-5-cont-eot20",
    ni=False,
    sample_step=1,
    t=60,
    t_delta=15,
    rand_t=False,
    diffusion_type="sde",
    score_type="score_sde",
    eot_iter=20,
    use_bm=False,
    sigma2=0.001,
    lambda_ld=0.01,
    eta=5.0,
    step_size=0.01,
    domain="cifar10",
    classifier_name="cifar10-wideresnet-28-10",
    partition="val",
    adv_batch_size=128,
    attack_type="square",
    lp_norm="Linf",
    attack_version="rand",
    num_sub=128,
    adv_eps=0.031373,
    log_dir="./exp_results/cifar10-robust_adv-5-eps0.031373-128x1-bm0-t0-end1e-5-cont-eot20/cifar10-wideresnet-28-10/sde_rand/seed123/data456",
)

config = Namespace(
    data=Namespace(
        dataset="CIFAR10",
        category="cifar10",
        image_size=32,
        num_channels=3,
        random_flip=True,
        centered=True,
        uniform_dequantization=False,
    ),
    model=Namespace(
        sigma_min=0.01,
        sigma_max=50,
        num_scales=1000,
        beta_min=0.1,
        beta_max=20.0,
        dropout=0.1,
        name="ncsnpp",
        scale_by_sigma=False,
        ema_rate=0.9999,
        normalization="GroupNorm",
        nonlinearity="swish",
        nf=128,
        ch_mult=[1, 2, 2, 2],
        num_res_blocks=8,
        attn_resolutions=[16],
        resamp_with_conv=True,
        conditional=True,
        fir=False,
        fir_kernel=[1, 3, 3, 1],
        skip_rescale=True,
        resblock_type="biggan",
        progressive="none",
        progressive_input="none",
        progressive_combine="sum",
        attention_type="ddpm",
        init_scale=0.0,
        embedding_type="positional",
        fourier_scale=16,
        conv_size=3,
    ),
    training=Namespace(sde="vpsde", continuous=True, reduce_mean=True, n_iters=950001),
    optim=Namespace(
        weight_decay=0,
        optimizer="Adam",
        lr=0.0002,
        beta1=0.9,
        eps=1e-08,
        warmup=5000,
        grad_clip=1.0,
    ),
    sampling=Namespace(
        n_steps_each=1,
        noise_removal=True,
        probability_flow=False,
        snr=0.16,
        method="pc",
        predictor="euler_maruyama",
        corrector="none",
    ),
    device=torch.device(type="cuda"),
)

runner = RevGuidedDiffusion(args=args, config=config)
runner.requires_grad_(False)


def diffpure(x: torch.Tensor) -> torch.Tensor:
    return runner.image_editing_sample(x * 2, bs_id=100) / 2
