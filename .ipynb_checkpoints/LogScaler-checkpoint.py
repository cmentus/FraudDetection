import numpy as np
class LogScaler():
    def __init__(self,data:'np.ndarray with non-negative values of shape n_data x n_predictors.'):
        ''' Add 1 to each variable.
        take log
        calculate mu and sigma + store
        transform 
        
        '''
        data = data + 1
        data = np.log(data)
        self.logmu = data.mean(0).reshape((1,-1))
        self.logsigma = data.std(0).reshape((1,-1))

    def transform(self,data):
        data = data + 1
        return (np.log(data) - self.logmu)/self.logsigma