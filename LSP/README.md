# Availability Attacks Create Shortcuts
You can generate the poisoned dataset with a demo script below:
```shell
python lsp_poisons_generate.py --dataset <Dataset> 
```
The parameter choices for the above commands are as follows:
- --dataset `<Dataset>`: `c10`, `c100`, `svhn`, `imagenet100`

### Acknowledgement
- Code adapted from the official implementation of LSP:
  [[Code]](https://github.com/dayu11/Availability-Attacks-Create-Shortcuts)
  [[Paper]](https://arxiv.org/pdf/2111.00898.pdf).