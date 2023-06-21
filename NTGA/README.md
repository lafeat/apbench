# Neural Tangent Generalization Attacks
First, you should install JAX on CPU by running:
```shell
pip install jax jaxlib --upgrade
```
To use JAX with GPU, 
please follow [JAX's GPU](https://github.com/google/jax/#installation) installation instructions.
Then you can install remaining requirements by running:
```shell
pip install -r requirements.txt
```
After that, you can generate the poisoned dataset with a demo script below:
```shell
python ntga_poisons_generate.py --dataset <Dataset> --eps <Epsilon of perturbation>
```
The parameter choices for the above commands are as follows:
- --dataset `<Dataset>`: `c10`
- --eps `<Epsilon of perturbation>`: `8/255`, `16/255`, ...

For simplicity, you can download the NTGA unlearnable CIFAR-10 datasets [Poisoned Data](https://drive.google.com/drive/folders/1PKA1BlNGROXyCuD3lt4bFe6Hvw1npELv).

### Acknowledgement
- Code adapted from the official implementation of NTGA:
  [[Code]](https://github.com/lionelmessi6410/ntga#unlearnable-datasets)
  [[Paper]](http://proceedings.mlr.press/v139/yuan21b/yuan21b.pdf).
