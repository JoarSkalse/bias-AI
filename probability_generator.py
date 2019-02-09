def prob(var_name, var_val, data, con={}):
    
    if len(con) == 0:
        #computes P = (var_name = var_val)
        h = data.loc[data[var_name] == var_val]
        p = len(h.index)/len(data.index)
        
    else:
        #computes P = (var_name = var_val / conditions)
        #find index of rows that meets each condition
        common = []
        for key, value in con.items():
            dfnew = data.loc[data[key].isin([value])]
            common.append(dfnew.index.values)
            
        #find index of rows that meet all of the conditions
        index = common[0]
        for ind in common[1:]:
            index = np.intersect1d(index, ind)
            
        #find prob of var == var_val
        data = data.iloc[index]
        if len(data.index) == 0:
            raise ValueError('sample to small, zero division')
        h = data.loc[data[var_name] == var_val]
        p = len(h.index)/len(data.index)
        
    return p



def bigen(n):
    x = []
    for _ in range(n):
        x.append(random.randint(0,1))
    return x

data = pd.DataFrame({'x1':bigen(200), 'x2':bigen(200), 'x3':bigen(200), 'x4':bigen(200)})
prob('x4', 1, data,{'x1':0,'x2':0,'x3':1})
prob('x4', 1, data)