import codecs as cd
import pickle

__all__= ['readinfo', 'sent2idx', 'init_kernel_initializer']


def readinfo(fname):
    with cd.open(fname, 'rb') as f:
        word_idx_map=pickle.load(f)
    return word_idx_map


def sent2idx(sent, word_idx_map, max_l):
    """
    Transforms sentence into a list of indices. Pad with negative one.
    """
    x = []
    for word in sent:
        x.append(word_idx_map.get(word, word_idx_map[u'<unk>']))
    while len(x) < max_l:
        x.append(-1)
    return x


def init_kernel_initializer(kernel_initializer = '',
                            kernel_initializer_normal = True,
                            kernel_initializer_uniform = False,
                            ):
    if(kernel_initializer != ''):
        kernel_initializer += '_'
    if(kernel_initializer_normal):
        kernel_initializer += 'normal'
    if(kernel_initializer_uniform):
        kernel_initializer += 'uniform'
    return kernel_initializer
