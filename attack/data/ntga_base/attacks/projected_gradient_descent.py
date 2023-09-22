import jax.numpy as np
from attacks.utils import clip_eta, one_hot
from attacks.fast_gradient_method import fast_gradient_method


def projected_gradient_descent(model_fn, kernel_fn, grads_fn, x_train, y_train, x_test, y_test, t=None, 
                               loss='cross-entropy', fx_train_0=0., fx_test_0=0., eps=0.3, eps_iter=0.03, 
                               nb_iter=10, norm=np.inf, clip_min=None, clip_max=None, targeted=False, 
                               rand_init=None, rand_minmax=0.3, key=None, batch_size=None):
    """
    This code is based on CleverHans library(https://github.com/cleverhans-lab/cleverhans).
    This class implements either the Basic Iterative Method
    (Kurakin et al. 2016) when rand_init is set to 0. or the
    Madry et al. (2017) method when rand_minmax is larger than 0.
    Paper link (Kurakin et al. 2016): https://arxiv.org/pdf/1607.02533.pdf
    Paper link (Madry et al. 2017): https://arxiv.org/pdf/1706.06083.pdf
    :param model_fn: a callable that takes an input tensor and returns the model logits.
    :param kernel_fn: a callable that takes an input tensor and returns the kernel matrix.
    :param grads_fn: a callable that takes an input tensor and a loss function, 
            and returns the gradient w.r.t. an input tensor.
    :param x_train: input tensor (training data).
    :param x_test: input tensor (test data; used for evaluation).
    :param y_train: Tensor with one-hot true labels of training data.
    :param y_test: Tensor with one-hot true labels of test data. If targeted is true, then provide the
            target one-hot label. Otherwise, only provide this parameter if you'd like to use true
            labels when crafting poisoned data. Otherwise, model predictions are used
            as labels to avoid the "label leaking" effect (explained in this paper:
            https://arxiv.org/abs/1611.01236). Default is None. This argument does not have
            to be a binary one-hot label (e.g., [0, 1, 0, 0]), it can be floating points values
            that sum up to 1 (e.g., [0.05, 0.85, 0.05, 0.05]).
    :param t: time step used to compute poisoned data.
    :param loss: loss function.
    :param fx_train_0 = output of the network at `t == 0` on the training set. `fx_train_0=None`
            means to not compute predictions on the training set. fx_train_0=0. for infinite width.
    :param fx_test_0 = output of the network at `t == 0` on the test set. `fx_test_0=None`
            means to not compute predictions on the test set. fx_test_0=0. for infinite width.
            For more details, please refer to equations (10) and (11) in Wide Neural Networks of 
            Any Depth Evolve as Linear Models Under Gradient Descent (J. Lee and L. Xiao et al. 2019). 
            Paper link: https://arxiv.org/pdf/1902.06720.pdf.
    :param eps: epsilon (input variation parameter); see https://arxiv.org/abs/1412.6572.
    :param eps_iter: step size for each attack iteration
    :param nb_iter: Number of attack iterations.
    :param norm: Order of the norm (mimics NumPy). Possible values: np.inf or 2.
    :param clip_min: (optional) float. Minimum float value for adversarial example components.
    :param clip_max: (optional) float. Maximum float value for adversarial example components.
    :param targeted: (optional) bool. Is the attack targeted or untargeted?
            Untargeted, the default, will try to make the label incorrect.
            Targeted will instead try to move in the direction of being more like y.
    :return: a tensor for the poisoned data.
    """
    assert eps_iter <= eps, (eps_iter, eps)
    if norm == 1:
        raise NotImplementedError("It's not clear that FGM is a good inner loop"
                                  " step for PGD when norm=1, because norm=1 FGM "
                                  " changes only one pixel at a time. We need "
                                  " to rigorously test a strong norm=1 PGD "
                                  "before enabling this feature.")
    if norm not in [np.inf, 2]:
        raise ValueError("Norm order must be either np.inf or 2.")
        
    x = x_train
    if y_test is None:
        # Using model predictions as ground truth to avoid label leaking
        x_labels = np.argmax(model_fn(kernel_fn, x_train, x_test, fx_train_0, fx_test_0)[1], 1)
        y_test = one_hot(x_labels, num_classes)
        
    # Initialize loop variables
    if rand_init:
        rand_minmax = eps
        eta = random.uniform(key, x.shape, minval=-rand_minmax, maxval=rand_minmax)
    else:
        eta = np.zeros_like(x)

    # Clip eta
    eta = clip_eta(eta, norm, eps)
    adv_x = x + eta
    if clip_min is not None or clip_max is not None:
        adv_x = np.clip(adv_x, a_min=clip_min, a_max=clip_max)
        
    for i in range(nb_iter):
        adv_x = fast_gradient_method(model_fn, kernel_fn, grads_fn, x_train, y_train, x_test, y_test, t, loss, 
                                     fx_train_0, fx_test_0, eps, norm, clip_min, clip_max, targeted, batch_size)

        # Clipping perturbation eta to norm norm ball
        eta = adv_x - x
        eta = clip_eta(eta, norm, eps)
        adv_x = x + eta

        # Redo the clipping.
        # FGM already did it, but subtracting and re-adding eta can add some
        # small numerical error.
        if clip_min is not None or clip_max is not None:
            adv_x = np.clip(adv_x, a_min=clip_min, a_max=clip_max)
            
    return adv_x
