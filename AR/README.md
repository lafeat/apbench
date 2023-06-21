# Learning to Confuse: Generating Training Time Adversarial Data with Auto-Encoder
You can generate the poisoned dataset with a demo script below:
```shell
python dc_poisons_generate.py --dataset <Dataset> --workers <Number of workers> --epsilon <Epsilon of perturbation>
```
The parameter choices for the above commands are as follows:
- --dataset `<Dataset>`: `c10`
- --epsilon `<Epsilon of perturbation>`: `0.5`, `1.0`, `1.5`, ...
- --workers `<Number of workers>`: `4`, `8`, ...

### Acknowledgement
- Code adapted from the official implementation of AR:
  [[Code]](https://github.com/psandovalsegura/autoregressive-poisoning)
  [[Paper]](https://arxiv.org/pdf/2206.03693.pdf).