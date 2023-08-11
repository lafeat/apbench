# Autoregressive Perturbations for Data Poisoning
You can generate the poisoned dataset with a demo script below:
```shell
python ar_poisons_generate.py --dataset <Dataset> --workers <Number of workers> --eps <Epsilon of perturbation>
```
The parameter choices for the above commands are as follows:
- --dataset `<Dataset>`: `c10` , `c100`, `svhn`.
- --eps `<Epsilon of perturbation>`: `8/255`, `16/255`, ...
- --workers `<Number of workers>`: `4`, `8`, ...

### Acknowledgement
- Code adapted from the official implementation of AR:
  [[Code]](https://github.com/psandovalsegura/autoregressive-poisoning)
  [[Paper]](https://arxiv.org/pdf/2206.03693.pdf).

