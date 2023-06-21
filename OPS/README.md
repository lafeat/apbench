# One-Pixel Shortcut: on the Learning Preference of Deep Neural Networks
You can generate the poisoned dataset with a demo script below:
```shell
python ops_poisons_generate.py --dataset <Dataset>
```
The parameter choices for the above commands are as follows:
- --dataset `<Dataset>`: `c10`, `svhn`

### Acknowledgement
- Code adapted from the official implementation of OPS:
  [[Code]](https://github.com/cychomatica/One-Pixel-Shotcut)
  [[Paper]](https://arxiv.org/pdf/2205.12141.pdf).