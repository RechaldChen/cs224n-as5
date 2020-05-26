#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1g
    def __init__(self, char_embed_size,word_embed_size,kernel_size=5,padding=1):
        super().__init__()
        self.conv_layer = nn.Conv1d(in_channels=char_embed_size,out_channels=word_embed_size,kernel_size=kernel_size,padding=padding)

    def forward(self, x_reshaped:torch.Tensor):
        """
        :param x_reshaped: a tensor with the shape (n_batch,e_char,m_word)
        :return:  x_conv_out
        """
        x_conv = self.conv_layer(x_reshaped)  # shape (n_batch,e_char,m_word-k+1)
        x_conv_out = torch.max(F.relu(x_conv), dim=2)[0]
        return x_conv_out


    ### END YOUR CODE

