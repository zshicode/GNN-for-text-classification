# Graph neural networks for text classification

Graph neural networks has been widely used in natural language processing. Yao et al. (2019) proposed TextGCN that adopts graph convolutional networks (GCN) (Kipf and Welling, 2017) for text classification on heterogeneous graph. We implemented TextGCN based on [PyTorch](https://pytorch.org/) and [DGL](https://docs.dgl.ai/). Furthermore, we suggest that inductive learning and attention mechanism is crucial for text classification using graph neural networks. So we adopt GraphSAGE (Hamilton et al., 2017) and graph attention networks (GAT) (Velickovic et al., 2018) for this classification task.

We evaluated our model on MR dataset (Tang et al., 2015) . Results:

|      | GCN (Yao et al., 2019) | GCN | GraphSAGE | GAT |
| :----: | :----: | :----: | :----: | :----: |
| Accuracy | 0.7552 | 0.7541 | 0.7642 | 0.7665 |
| Macro-F1 | 0.7548 | 0.7536 | 0.7637 | 0.7664 |

Some of our codes are from repository [TextGCN](https://github.com/yao8839836/text_gcn) and [TextGCN in PyTorch](https://github.com/iworldtong/text_gcn.pytorch). More datasets can also be downloaded from these two repositories.

## Requirements

 The code has been tested running under:

* Python 3.7
* PyTorch 1.3
* dgl-cu101 0.4
* CUDA 10.1

## Running training and evaluation

First we need to preprocess data:

1. `cd ./preprocess`
2. Run `python remove_words.py <dataset>`
3. Run `python build_graph.py <dataset>`
4. `cd ../`
5. Replace `<dataset>` with `20ng`, `R8`, `R52`, `ohsumed` or `mr`

then run

```bash
python main.py --model GCN --cuda True
```

parameters:

```python
def get_citation_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=bool, default=False,
                        help='Use CUDA training.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.02,
                        help='Initial learning rate.')
    parser.add_argument('--model', type=str, default="GCN",
                        choices=["GCN", "SAGE", "GAT"],
                        help='model to use.')
    parser.add_argument('--early_stopping', type=int, default=10,
                        help='require early stopping.')
    parser.add_argument('--dataset', type=str, default='mr',
                        choices = ['20ng', 'R8', 'R52', 'ohsumed', 'mr'],
                        help='dataset to train')

    args, _ = parser.parse_known_args()
    #args.cuda = not args.no_cuda and th.cuda.is_available()
    return args

args = get_citation_args()
```

## References

William L. Hamilton, Rex Ying, and Jure Leskovec. Inductive representation learning on large graphs. In NeurIPS, 2017. 

Thomas Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. In ICLR, 2017. 

Jian Tang, Meng Qu, and Qiaozhu Mei. Pte: Predictive text embedding through large-scale heterogeneous text networks. In KDD, 2015. 

Petar Velickovic, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua Bengio. Graph attention networks. In ICLR, 2018. 

Liang Yao, Chengsheng Mao, and Yuan Luo. Graph convolutional networks for text classification. Proceedings of the AAAI Conference on Artificial Intelligence, 33:7370–7377, 2019.
