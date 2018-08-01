import numpy as np
import statsmodels.api as sm

class DynamicRegression(sm.tsa.statespace.MLEModel):
    def __init__(self, endog, design):
        k_states = np.array(design).shape[1]
        k_posdef = k_states
        super(DynamicRegression, self).__init__(
            endog=endog,
            k_states=k_states,
            k_posdef=k_posdef,
            initialization='approximate_diffuse'
        )
        self.endog = endog
        self['design'] = design
        self['transition'] = np.eye(k_states)
        self['selection'] = np.eye(k_states)
                
    def update(self, params, transformed=True, **kwargs):
        super(DynamicRegression, self).update(
            params, 
            transformed, 
            **kwargs
        )
        n=0
        param_slice = slice(n, n+self.k_endog)
        for i, param in enumerate(params[param_slice]):
            self['obs_cov', i, i] = param
            n+=1
        
        param_slice = slice(n, n+self.k_states)
        for i, param in enumerate(params[1:]):
            self['state_cov', i, i] = param
            n+=1
        
    @property
    def param_names(self):
        return (
            [f'obs{i}_cov' for i in range(self.k_endog)] 
            + [f'state{i}_var' for i in range(self.k_states)]
        )

    @property
    def start_params(self):
        return (
            [100 for _ in range(self.k_endog)] 
            + [1 for _ in range(self.k_states)]
        )