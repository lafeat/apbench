import jax.numpy as np
from jax.api import grad, jit, vmap
from jax import lax
from jax.experimental.stax import logsoftmax
from jax.config import config
config.update('jax_enable_x64', True)

from functools import partial
import neural_tangents as nt
from neural_tangents import stax
# -

# Kernel Construction
_Kernel = nt.utils.kernel.Kernel

def Kernel(K):
    """Create an input Kernel object out of an np.ndarray."""
    return _Kernel(cov1=np.diag(K), nngp=K, cov2=None, 
                   ntk=None, is_gaussian=True, is_reversed=False,
                   diagonal_batch=True, diagonal_spatial=False,
                   batch_axis=0, channel_axis=1, mask1=None, mask2=None,
                   shape1=(2, 1024), shape2=(2,1024),
                   x1_is_x2=True, is_input=True) 

def NTKernel(var1, nngp, var2, ntk):
    """Create an input Kernel object out of an np.ndarray."""
    return _Kernel(cov1=var1, nngp=nngp, cov2=var2, 
                   ntk=ntk, is_gaussian=True, is_reversed=False,
                   diagonal_batch=True, diagonal_spatial=False,
                   batch_axis=0, channel_axis=1, mask1=None, mask2=None,
                   shape1=(2, 1024), shape2=(2,1024),
                   x1_is_x2=True, is_input=True) 

def wrap(kernel_fn):
    def wrapped_fn(kernel):
        out = kernel_fn(NTKernel(*kernel))
        return kernel._replace(cov1=out.cov1, nngp=out.nngp, cov2=out.cov2, ntk=out.ntk)
    return wrapped_fn

def fixed_point(f, initial_value, threshold):
    """Find fixed-points of a function f:R->R using Newton's method."""
    g = lambda x: f(x) - x
    dg = grad(g)

    def cond_fn(x):
        x, last_x = x
        return np.abs(x - last_x) > threshold

    def body_fn(x):
        x, _ = x
        return x - g(x) / dg(x), x
    return lax.while_loop(cond_fn, body_fn, (initial_value, 0.0))[0]

# +
def qc_map(W_var, b_var):
    """
    Q-map and C-map functions mentioned in Exponential Expressivity in Deep Neural Networks
    through Transient Chaos (B Poole1 et al. 2016) and Deep Information Propagation
    (S. S. Schoenholz and J. Gilmer et al. 2017).
    :param W_var: float. Variance of weights at initialization.
    :param b_var: float. Variance of biases at initialization.
    :return: a callable Q-map and C-map functions.
    """
    W_std = np.sqrt(W_var)
    b_std = np.sqrt(b_var)

    # Create a single layer of a network as an affine transformation composed
    # with an Erf nonlinearity.
    kernel_fn = stax.serial(stax.Erf(), stax.Dense(1024, W_std, b_std))[2]
  
    def q_map_fn(q):
        return kernel_fn(Kernel(np.array([[q]]))).nngp[0, 0]
    qstar = fixed_point(q_map_fn, 1.0, 1e-7)

    def c_map_fn(c):
        K = np.array([[qstar, qstar * c], [qstar * c, qstar]])
        K_out = kernel_fn(Kernel(K)).nngp
        return K_out[1, 0] / qstar
    
    return q_map_fn, c_map_fn

@partial(vmap, in_axes=(0, None, None))
def xi_1(W_var, b_var, chi_1):
    """
    Depth scale of trainability/generalization in the ordered phase.  
    """
    return 1./ (np.abs(np.log(chi_1(W_var, b_var)))  + 1e-12)

@partial(vmap, in_axes=(0, None, None, None))
def xi_star(W_var, b_var, chi_1, chi_c):
    """
    Depth scale for generaliztion of NTK in the chaotic phase. 
    """
    return 1. /(-np.log(chi_c(W_var, b_var)) + np.log(chi_1(W_var, b_var)))

@partial(vmap, in_axes=(0, None))
def accuracy_vmap(y_pred, y_test):
    """
    Compute accuracies from predictions at different times.
    :param y_pred: np.ndarray. Prediction of Gaussian Process.
    :param y_test: np.ndarray. Ground truth label.
    :return: a float for accuracy.
    """
    return np.mean(np.argmax(y_pred, axis=-1) == np.argmax(y_test, axis=-1))


# -

# TODO: This is necessary because of a bug in NT's CPU detection inside a jit
nt.predict._arr_is_on_cpu = lambda x: False

def scale(a, b):
    return a * b[-1] / a[-1]


# +
@jit
def l2_loss_v1(logits, labels):
    """
    Tensorflow version of L2 loss (without sqrt)
    :param logits: a np.ndarray for the model logits.
    :param labels: a np.ndarray for with one-hot true labels.
    :return: a float for loss.
    """
    return np.sum((logits - labels)**2) / 2
    
@jit
def l2_loss_v2(logits, lables):
    """
    Normal L2 loss
    :param logits: a np.ndarray for the model logits.
    :param labels: a np.ndarray for with one-hot true labels.
    :return: a float for loss.
    """
    return np.linalg.norm(logits - labels)

@jit
def cross_entropy_loss(logits, lables):
    """
    Cross-entropy loss
    :param logits: a np.ndarray for the model logits.
    :param labels: a np.ndarray for with one-hot true labels.
    :return: a float for loss.
    """
    return -np.sum(logsoftmax(logits) * lables)
    
@jit
def mse_loss(logits, lables):
    """
    Mean squared loss
    :param logits: a np.ndarray for the model logits.
    :param labels: a np.ndarray for with one-hot true labels.
    :return: a float for loss.
    """
    return 0.5 * np.mean((logits - lables) ** 2)
