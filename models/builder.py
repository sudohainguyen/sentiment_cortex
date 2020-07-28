from keras.models import Model
import keras.backend as K
from keras.layers import (
    Input, Embedding, Dense,
    Add, GlobalAveragePooling1D, Dropout,
    multiply
)
from .layers import *

def builder(
    max_l,
    vocab_size,
    kernel_initializer,
    embsize=384,
    embedding_scale=4,
    input_normalize=False,
    normalize_center=True,
    normalize_scale=True,
    operator='concat',
    headsizes=4,
    Multi_ATT_normalized=True,
    SE_scale=4,
    FFN_scale=4,
    outATTheads=1,
    ATT_normalized=True,
    classifier_nodes=[512,1024],
    classifier_activations=['sigmoid', 'sigmoid']
):
    if operator == "sum":
        embedding_size = embsize
    elif operator == "concat":
        embedding_size = embsize*2
    FFNoutputdim = embedding_size

    inp = Input(shape=(max_l,))
    x = Embedding(vocab_size + 1, embsize // embedding_scale,
                  trainable=True, embeddings_initializer=kernel_initializer)(inp)
    
    if embedding_scale > 1:
        x = Dense(embsize,
                  activation=gelu,
                  kernel_initializer=kernel_initializer,
                  use_bias=True)(x)
    
    position_em = Sinusoidal_Position_Embedding(name="Position_embedding_layer", mode=operator)(x)
    if input_normalize:
        position_em = LayerNormalization(center=normalize_center,
                                         scale=normalize_scale,
                                         name='Norm_input',
                                         gamma_initializer=kernel_initializer,
                                         beta_initializer=kernel_initializer)(position_em)

    block1 = ModMultiHeadAttention(headsizes, normalized=Multi_ATT_normalized, 
                                   activation=gelu, name='Self_attention_block_1',
                                   kernel_initializer=kernel_initializer,
                                   bias_initializer=kernel_initializer)([position_em, position_em, position_em])

    block1_add_1 = Add(name='Add_block_1_ATT')([block1, position_em])
    block1_Multihead_att_out = LayerNormalization(center=normalize_center,
                                                  scale=normalize_scale,
                                                  name='Norm_block1_1',
                                                  gamma_initializer=kernel_initializer,
                                                  beta_initializer=kernel_initializer)(block1_add_1)
    se_GAP = GlobalAveragePooling1D()(block1_Multihead_att_out)
    se_squezee = Dense(embedding_size // SE_scale, 
                       activation=gelu, 
                       kernel_initializer=kernel_initializer,
                       use_bias=True)(se_GAP)
    se_excitation = Dense(se_GAP._keras_shape[-1], #get shape of GAP in SE block 
                          activation='sigmoid',
                          kernel_initializer=kernel_initializer,
                          use_bias=True)(se_squezee)
    
    mult_Out = multiply([block1_Multihead_att_out, se_excitation])
    mult_Out = Add(name='Add_block_1_SnE')([block1_Multihead_att_out, mult_Out])

    block1_feature_att_out = LayerNormalization(center=normalize_center,
                                                scale=normalize_scale, name='Norm_block1_2',
                                                gamma_initializer=kernel_initializer,
                                                beta_initializer=kernel_initializer)(mult_Out)
    block1_FFN = FeedForward(ratio=1/FFN_scale,
                            outputdim=FFNoutputdim,
                            name='FFN_block_1',
                            activation=gelu,
                            kernel_initializer=kernel_initializer)(block1_feature_att_out)
    block1_add_3 = Add(name='Add_block_1_FNN')([block1_feature_att_out, block1_FFN])
    block1_FFN_out = LayerNormalization(center=normalize_center,
                                        scale=normalize_scale,
                                        name='Norm_block1_FFN',
                                        gamma_initializer=kernel_initializer,
                                        beta_initializer=kernel_initializer)(block1_add_3)
    block1_out = ModMultiHeadAttention(outATTheads, normalized = ATT_normalized, activation=gelu,   
                                       name='Self_attention_block_1_final', 
                                       kernel_initializer=kernel_initializer, 
                                       bias_initializer=kernel_initializer)([block1_FFN_out,
                                                                            block1_FFN_out,
                                                                            block1_FFN_out])
    block1_out = Add(name='Add_block_1_ATTOUT')([block1_out,
                                                 block1_FFN_out])
    block1_out = LayerNormalization(center=normalize_center, 
                                    scale=normalize_scale, 
                                    name='Norm_block1_4',
                                    gamma_initializer=kernel_initializer,
                                    beta_initializer=kernel_initializer)(block1_out)                                            

    x = GlobalAveragePooling1D()(block1_out)
    x = Dropout(0.5)(x)

    for nodes,activator in zip(classifier_nodes, classifier_activations):
        x = Dense(nodes, activation=activator,
                  kernel_initializer=kernel_initializer,
                  use_bias=True)(x)

    x = Dense(2, activation='softmax',
              kernel_initializer=kernel_initializer,
              use_bias=True)(x)

    return Model(inputs = inp, outputs = x)
