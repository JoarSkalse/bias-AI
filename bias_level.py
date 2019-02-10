import pandas as pd
import numpy as np
import itertools
import random


def prob(var_name, var_val, data, con={}):
    
    # return P = (var_name = var_val) if con={}
    # return P = (var_name = var_val / conditions) if con={'X1'=x1,'X2'=x2 ...}
    
    if len(con) == 0:
        h = data.loc[data[var_name] == var_val]
        p = len(h.index)/len(data.index)
    else:
        common = []
        for key, value in con.items():
            dfnew = data.loc[data[key].isin([value])]
            common.append(dfnew.index.values)
        index = common[0]
        for ind in common[1:]:
            index = np.intersect1d(index, ind)    
        data = data.iloc[index]
        if len(data.index) == 0:
            raise ValueError('sample to small, zero division')
        h = data.loc[data[var_name] == var_val]
        p = len(h.index)/len(data.index)    
    return p


def con_iterator(uns, data):
    
    # return list of lists of all possible value combinations of Un("Unbiased factors" which should be considered during selection) in list
    
    lis = []
    for un in uns:
        lis.append(data[un].unique())
    l2 = []
    for i in list(itertools.product(*lis)):
        l2.append(list(i))
    return l2


def dic_assembler(keys, values):
    
    # return combined dictionary from two lists
    
    return dict(zip(keys, values))


def p_level(uns_dic, bns, bns_lis_comb, bmn, bmv, h, data):
    
    # return P(H=1/do(U1=u1),do(U2=u2)...,do(Un=un),Bz)
    
    p_l = 0
    for bns_vals in bns_lis_comb:
        bns_dic = dic_assembler(bns, bns_vals)
        p = 1
        for bkey, bval in bns_dic.items():
            p *= prob(bkey, bval, data)
        con = {**uns_dic, **bns_dic, **{bmn:bmv}}
        p *= prob(h, 1, data, con)
        p_l += p
    return p_l


def bias_level(uns, uns_list_comb, bns, bns_lis_comb, bmn, h, n, data):
    
    # return the bias level of the selection based on x^n metric
    
    bias = 0 
    for each in uns_list_comb:
        uns_dic = dic_assembler(uns, each)
        pval_list = []
        bm_vl = data[bmn].unique()
        for bmv in bm_vl:
            pval = p_level(uns_dic, bns, bns_lis_comb, bmn, bmv, h, data)
            pval_list.append(pval)
        bias += (max(pval_list)-min(pval_list))**n
    return bias/len(uns_list_comb)


def bias_level_assessor(biased_factors, hired_col_name, measured_biased_factor, n_metric, data):
    
    # return bias level of the selection if corresponding inputs are provided
    # biased_factors should be a list
    
    col_list =  list(data.columns)
    del col_list[col_list.index(hired_col_name)]
    unbiased_factors = [col for col in col_list if col not in biased_factors]
    unbiased_factors_value_combinations = con_iterator(unbiased_factors, data)
    biased_factors_value_combinations = con_iterator(biased_factors, data)
    
    return bias_level(unbiased_factors, unbiased_factors_value_combinations, biased_factors, biased_factors_value_combinations, measured_biased_factor, hired_col_name, n_metric, data)