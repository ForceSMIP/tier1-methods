#!/usr/bin/env python
# coding: utf-8

import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from ldm_model import ldm_model
from time import time
import scipy.stats
from copy import deepcopy
import pickle as pkl



def train(y_train,par,exact=False,train_config={}):
    config={'eps':1e-7,'exact_eps':1e-10}
    config.update(train_config)
    
    t1=time()    

    dp=par['dp']
    df=par['df']
    
    model=ldm_model()
    model.init(y_train,dp,df,
                        tau_p=par['tau_p'],tau_f=par['tau_f'],
                        sigma_p=par['sigma_p'],sigma_f=par['sigma_f'],
                        sigma_c=par['sigma_c'],model_ids=par['model_ids'])

    if exact:
        if par['init'] is None:
            model.gen_pars()
        else:
            model.gen_pars(par['init'],0.,p_mask=None,f_mask=None)
        #print(model.minus_log_evidence_nn_pars())
        total_nit=model.fit_pars(eps=config['exact_eps'])   
    else:
        if par['init'] is None:
            model.gen_pars()
        else:
            model.gen_pars(par['init'],par['coeff'],p_mask=par['p_mask'],f_mask=par['f_mask'])
        total_nit=model.fit_pars(eps=config['eps'])
        
    par.pop('init')
    
    res=par.copy()
    res.update({
        'model_pars':model.get_pars(),
        'total_nit':total_nit,
        'loss':model.loss_average(),
        'evi':model.minus_log_evidence_nn_pars(),
        'cost':model.cost()
    })
    res['p_contrib'], res['f_contrib']=model.get_contributions()
    
    
    c=res['model_pars']['c']
    p=res['model_pars']['p']
    sigma_p=np.sqrt((p**2).mean(axis=0).mean(axis=0))
    sigma_p=np.maximum(1e-2,sigma_p)
    alpha_p=(p[:,1:,:]*p[:,:-1,:]).mean(axis=0).mean(axis=0)/(sigma_p**2)
    tau_p=-1./np.log(np.clip(alpha_p,1e-6,1.-1e-3))
   
    f=res['model_pars']['f']
    sigma_f=np.sqrt((f**2).mean(axis=0).mean(axis=0))
    sigma_f=np.maximum(1e-2,sigma_f)
    alpha_f=(f[...,1:,:]*f[...,:-1,:]).mean(axis=0).mean(axis=0)/(sigma_f**2) 
    tau_f=-1./np.log(np.clip(alpha_f,1e-6,1.-1e-3))
    next_par={
        'model_ids':par['model_ids'],
        'dp':par['dp'],
        'df':par['df'],
        'sigma_c': np.clip(np.sqrt((c**2).mean()),1e-4,0.1),
        'tau_p':tau_p,
        'sigma_p':sigma_p,
        'tau_f':tau_f,
        'sigma_f':sigma_f,
        'init':res['model_pars'],
        'coeff':par['coeff'],
        'p_mask':par['p_mask'],
        'f_mask':par['f_mask']
    }
    
    t2=time()
    res['comp_time']=t2-t1
    
    return res, next_par
    

def truncnorm2(myclip_a,myclip_b,loc,scale):
    a, b = (myclip_a - loc) / scale, (myclip_b - loc) / scale
    return scipy.stats.truncnorm.rvs(a=a,b=b,loc=loc, scale=scale)      

class ldm_optimizer():
    def __init__(self):
        return
    
    def init(self,y_train,dp,df,model_ids=None,optimizer_config={},train_config={},init_par=None,_skip_init=False):
        '''
        Parameters
        ----------
        y_train : TYPE
            DESCRIPTION.
        model_ids : TYPE
            DESCRIPTION.
        dp : TYPE
            DESCRIPTION.
        df : TYPE
            DESCRIPTION.
        optimizer_config : TYPE, optional
            DESCRIPTION. The default is {}.
        train_config : TYPE, optional
            DESCRIPTION. The default is {}.
        init_par : parameter dictionary to start with, optional. Overrides model_ids, dp and df
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        '''
        if _skip_init: return
        config={'coeffstart':0.05,'coeffend':0.01,'nmax':1000,'threshold':1e-4,
                'taumin':1.,'taumax':y_train.shape[-2],
                'sigmapmin':0.005,'sigmapmax':0.05,'sigmafmin':0.05,'sigmafmax':0.5,
                'master_p_mask':np.full(dp,True),'master_f_mask':np.full(df,True)}
        config.update(optimizer_config)

        self.train_config=train_config.copy()

        self.coeffstart=config['coeffstart']
        self.coeffend=config['coeffend']
        self.nmax=config['nmax']
        self.coeff=self.coeffstart
        #self.T=0
        self.nstep=0
        
        self.logtaumin=np.log(config['taumin'])
        self.logtaumax=np.log(config['taumax'])
        self.logsigmapmin=np.log(config['sigmapmin'])
        self.logsigmapmax=np.log(config['sigmapmax'])
        self.logsigmafmin=np.log(config['sigmafmin'])
        self.logsigmafmax=np.log(config['sigmafmax'])        


        self.y_train=y_train
        self.tau_clip=[0.1,self.y_train.shape[-2]*10.]
        self.sigma_clip=[1e-10,1.]
        self.master_p_mask=config['master_p_mask']
        self.master_f_mask=config['master_f_mask']      
        
        par=self.__initial_training(model_ids,dp,df,init_par)
        self.state=par
        self.best=par
        res1=par.copy()
        res1.pop('model_pars')
        self.history=[res1]  
        self.threshold=config['threshold']

    
    def __initial_training(self,model_ids,dp,df,init_par):
        p_mask=np.logical_and(np.full(dp,False),self.master_p_mask)
        f_mask=np.logical_and(np.full(df,True),self.master_f_mask)        
        if init_par is None:
            par={'model_ids':model_ids,'dp':dp,'tau_p':np.ones(dp),
            'df':df,'tau_f':np.ones(df)}
            par['sigma_p']=np.ones(dp)
            par['sigma_f']=np.ones(df)
            par['sigma_c']=1.
            par['tau_p'][np.logical_not(p_mask)]=self.y_train.shape[-2]
            par['sigma_p'][np.logical_not(p_mask)]=1e-4
            par['tau_f'][np.logical_not(f_mask)]=self.y_train.shape[-2]
            par['sigma_f'][np.logical_not(f_mask)]=1e-4  
            par['init']=None
            par['coeff']=0.
            par['p_mask']=None
            par['f_mask']=None       
        else:
            par=init_par.copy()
        
        best=None

        print('Initial iterations')
        
        for i in range(10):
            exact=True
            if i<5 and i>0: exact=False
            res, par=train(self.y_train,par,exact=exact,train_config={'eps':self.train_config['exact_eps'],'exact_eps':self.train_config['exact_eps']})
            par['tau_p'][np.logical_not(p_mask)]=res['tau_p'][np.logical_not(p_mask)]
            par['sigma_p'][np.logical_not(p_mask)]=res['sigma_p'][np.logical_not(p_mask)]
            par['tau_f'][np.logical_not(f_mask)]=res['tau_f'][np.logical_not(f_mask)]
            par['sigma_f'][np.logical_not(f_mask)]=res['sigma_f'][np.logical_not(f_mask)]           

            
            #update best ldm        
            if (best is None) or (res['evi']<=best['evi']): best=res

            s=''
            for key in ['evi','loss','cost','tau_p','tau_f','sigma_p','sigma_f']:
                s=s+' '+key+'='+str(res[key])      
            print(s) 
            print('best evi:', best['evi'])

            sys.stdout.flush()
        
        
        p_mask=np.logical_and(np.full(dp,True),self.master_p_mask)
        f_mask=np.logical_and(np.full(df,False),self.master_f_mask)
        init_par=best
        init_par['sigma_p']=np.ones(dp)
        init_par['tau_p']=np.ones(dp)
        init_par['init']=init_par['model_pars']
        init_par['coeff']=0.99
        init_par['p_mask']=p_mask
        init_par['f_mask']=f_mask  
        par=init_par.copy()

        best=None       
        for i in range(10):
            exact=True
            if i<5 and i>0: exact=False
            res, par=train(self.y_train,par,exact=exact,train_config={'eps':self.train_config['exact_eps'],'exact_eps':self.train_config['exact_eps']})
            par['tau_p'][np.logical_not(p_mask)]=res['tau_p'][np.logical_not(p_mask)]
            par['sigma_p'][np.logical_not(p_mask)]=res['sigma_p'][np.logical_not(p_mask)]
            par['tau_f'][np.logical_not(f_mask)]=res['tau_f'][np.logical_not(f_mask)]
            par['sigma_f'][np.logical_not(f_mask)]=res['sigma_f'][np.logical_not(f_mask)]           

            
            #update best ldm        
            if (best is None) or (res['evi']<=best['evi']): best=res

            s=''
            for key in ['evi','loss','cost','tau_p','tau_f','sigma_p','sigma_f']:
                s=s+' '+key+'='+str(res[key])      
            print(s) 
            print('best evi:', best['evi'])

            sys.stdout.flush()  
            
        return best

    def save_to_file(self,file_name):
        pars={}
        for key in self.__dict__.keys():
            pars[key]=deepcopy(self.__dict__[key])
        with open(file_name,'wb') as f:
            pkl.dump(pars,f)
    
    def load_from_file(self,file_name):
        with open(file_name,'rb') as f:
            pars=pkl.load(f)
        for key in pars.keys():
            self.__dict__[key]=deepcopy(pars[key])   
    
    def final_training_compact(self,final_eps=1e-12):
        par=self.best.copy()
        par['init']=self.best['model_pars'].copy()
        par['coeff']=0.
        par['p_mask']=None
        par['f_mask']=None
        train_config={'eps':final_eps,'exact_eps':final_eps}             
        res, __par=train(self.y_train,par,exact=True,train_config=train_config)
        return res
    
    def final_training(self,final_eps=1e-12):   
        #dictionary containing trained parameters and hyperparameters in a compact way
        par=self.best
        
        #model parameters returned by model.get_pars() method
        model_pars=par['model_pars']
        
        #re-initialization of a model
        model=ldm_model()
        model.init(self.y_train,dp=par['dp'],df=par['df'],
                            tau_p=par['tau_p'],tau_f=par['tau_f'],
                            sigma_p=par['sigma_p'],sigma_f=par['sigma_f'],
                            sigma_c=par['sigma_c'],model_ids=par['model_ids'])
        model.set_pars(model_pars)
        
        #final training with (presumably) high accuracy
        model.fit_pars(eps=final_eps)  
        return model
        
    def generate_new_state_gibbs(self):
        
        res_old=self.state
        
        par={'model_ids':res_old['model_ids'],'dp':res_old['dp'],'df':res_old['df']}            
        for key in ['tau_p','tau_f','sigma_p','sigma_f']: par[key]=res_old[key].copy()            
        for key in ['sigma_c']:
            par[key]=np.exp(truncnorm2(np.log(self.sigma_clip[0]),np.log(self.sigma_clip[1]),
                                          loc=np.log(res_old[key]),
                                          scale=0.5*self.coeff*np.log(100.)))
   
        p_mask=np.full(res_old['dp'],False)
        f_mask=np.full(res_old['df'],False)
        if par['dp']>0 and np.any(self.master_p_mask):
            i=np.arange(par['dp'])[self.master_p_mask][np.random.randint(self.master_p_mask.sum())]
            p_mask[i]=True
            par['tau_p'][i]=np.exp(truncnorm2(np.log(self.tau_clip[0]),np.log(self.tau_clip[1]),
                                          loc=np.log(res_old['tau_p'][i]),
                                          scale=0.5*self.coeff*np.log(100.)))
 
            par['sigma_p'][i]=np.exp(truncnorm2(np.log(self.sigma_clip[0]),np.log(self.sigma_clip[1]),
                                          loc=np.log(res_old['sigma_p'][i]),
                                          scale=0.5*self.coeff*np.log(100.)))
            
        if par['df']>0 and np.any(self.master_f_mask):

            i=np.arange(par['df'])[self.master_f_mask][np.random.randint(self.master_f_mask.sum())]
            f_mask[i]=True
            par['tau_f'][i]=np.exp(truncnorm2(np.log(self.tau_clip[0]),np.log(self.tau_clip[1]),
                                          loc=np.log(res_old['tau_f'][i]),
                                          scale=0.5*self.coeff*np.log(100.))) 

            par['sigma_f'][i]=np.exp(truncnorm2(np.log(self.sigma_clip[0]),np.log(self.sigma_clip[1]),
                                          loc=np.log(res_old['sigma_f'][i]),
                                          scale=0.5*self.coeff*np.log(100.)))                     
        
        par['init']=res_old['model_pars']
        par['coeff']=self.coeff
        par['p_mask']=p_mask
        par['f_mask']=f_mask
        res, par=train(self.y_train,par,train_config=self.train_config)
     
        res=self.update_status(res)
        
        return res 

    def generate_perturbation(self):
        res_old=self.state
        pstd=res_old['p_contrib'].copy()
        fstd=res_old['f_contrib'].copy()
        pstd[np.logical_not(self.master_p_mask)]=self.threshold+1000.
        fstd[np.logical_not(self.master_f_mask)]=self.threshold+1000.
        if np.all(pstd>self.threshold) and np.all(fstd>self.threshold): return None   
        

        
        par={'model_ids':res_old['model_ids'],'dp':res_old['dp'],'df':res_old['df']}            
        for key in ['tau_p','tau_f','sigma_p','sigma_f','sigma_c']: par[key]=res_old[key].copy()                  

        par['init']=res_old['model_pars']   
        p_mask=np.full(res_old['dp'],False)
        f_mask=np.full(res_old['df'],False)
        u=np.random.randint(res_old['dp']+res_old['df'])
        if np.all(pstd>self.threshold): u=res_old['dp']+np.random.randint(res_old['df'])
        if np.all(fstd>self.threshold): u=np.random.randint(res_old['dp'])
        if u<res_old['dp']:
            i=np.argmin(pstd)
            p_mask[i]=True
            par['tau_p'][i]=np.exp(scipy.stats.truncnorm.rvs(a=-1,b=1,
                                                             loc=0.5*(self.logtaumax+self.logtaumin),
                                                             scale=0.5*(self.logtaumax-self.logtaumin)))    
 
            par['sigma_p'][i]=np.exp(scipy.stats.truncnorm.rvs(a=-1,b=1,
                                                             loc=0.5*(self.logsigmapmax+self.logsigmapmin),
                                                             scale=0.5*(self.logsigmapmax-self.logsigmapmin)))  
            
        else:
            i=np.argmin(fstd)
            f_mask[i]=True
            par['tau_f'][i]=np.exp(scipy.stats.truncnorm.rvs(a=-1,b=1,
                                                             loc=0.5*(self.logtaumax+self.logtaumin),
                                                             scale=0.5*(self.logtaumax-self.logtaumin)))    

            par['sigma_f'][i]=np.exp(scipy.stats.truncnorm.rvs(a=-1,b=1,
                                                             loc=0.5*(self.logsigmafmax+self.logsigmafmin),
                                                             scale=0.5*(self.logsigmafmax-self.logsigmafmin)))                       
        
        par['coeff']=1.
        par['p_mask']=self.master_p_mask
        par['f_mask']=self.master_f_mask
        res, par=train(self.y_train,par,train_config=self.train_config)
        
        res=self.update_status(res)
        
        return res              

    
    def update_status(self,new_state):
        #update best ldm        
        if new_state['evi']<=self.best['evi']:
            comp_time=new_state['comp_time']
            par=new_state.copy()
            par['init']=new_state['model_pars']           
            new_state, par=train(self.y_train,par,exact=True,train_config=self.train_config)
            new_state['comp_time']+=comp_time
            new_state['exact']=True
            if new_state['evi']<=self.best['evi']:
                self.best=new_state.copy()
        else:
            new_state['exact']=False

        #delete the model parameters and save result to the history
        res1=new_state.copy()
        res1.pop('model_pars')
        res1['nstep']=self.nstep
        self.history.append(res1)
        
        return new_state    

    def print_status(self):
        t_mean=np.array([xx['comp_time'] for xx in self.history[1:] if 'comp_time' in xx.keys()]).mean()
        print(self.nstep,'t_mean=',t_mean,'last=',self.history[-1]['evi'],'state=',self.state['evi'],'best=',self.best['evi'],'coeff=',self.coeff)
        return        
    
    def transition_probability(self,new_state,old_state,T):
        if new_state['evi']<=old_state['evi']: 
            return 1.
        else:
            if T<1e-8:
                return 0.
            else:
                return np.exp(-(new_state['evi']-old_state['evi'])/T)
    

    
    def update_T(self):     
        self.nstep=self.nstep+1

        n1=(self.nmax*4)//10
        n2=(self.nmax*9)//10
        n3=self.nmax
        if self.nstep%self.nmax>=n2:
            k=self.nstep%self.nmax-n2
            kmax=n3-n2
            self.coeff=(self.coeffend*(kmax-k)+1.*(k))/kmax 
        elif self.nstep%self.nmax>=n1: 
            k=self.nstep%self.nmax-n1
            kmax=n2-n1
            self.coeff=(self.coeffstart*(kmax-k)+self.coeffend*(k))/kmax 
        else:
            self.coeff=self.coeffstart
      
    def iterate(self):
        if self.nstep>=self.nmax: 
            perturbation_flag=True
            if self.nstep%self.nmax==0:
                evis=np.array([h['evi'] for h in self.history[len(self.history)-self.nmax:]])
                dn=self.nmax-(self.nmax*9)//10
                min1=np.amin(evis[:-2*dn])
                min2=np.amin(evis[:-dn])
                min3=np.amin(evis)
                if min2-min3>10*(min1-min2)+1e-4:
                    print(min1,'->',min2,'->',min3,': Perturbation not needed')
                    perturbation_flag=False
                    self.coeff=self.coeffstart
                    self.nstep=0                    
                
            if perturbation_flag==True:
                new_state=self.generate_perturbation()
                
                if new_state is not None:
                    alpha=self.transition_probability(new_state,self.history[0],self.history[0]['evi']-self.best['evi'])
                    u=np.random.uniform(0,1)
                    if alpha>=u:
                        print('Perturbation accepted')
                        self.state=new_state
                        self.coeff=self.coeffstart
                        #self.T=self.Tstart
                        self.nstep=0            
                    else:
                        print('Perturbation rejected')
                        self.nstep+=1
                    return

                
                                
        
        new_state=self.generate_new_state_gibbs()
        self.update_T()
        alpha=self.transition_probability(new_state,self.state,T=0.)
        u=np.random.uniform(0,1)
        if alpha>=u:
            self.state=new_state
    
    def plot_status(self,fname=None):
        import matplotlib.pyplot as plt
        fig=plt.figure(figsize=(10,10))
        fig.subplots_adjust(hspace=0.3)
        n=1
        for key in ['evi','loss','p_contrib','f_contrib']:
            y=np.array([h[key] for h in self.history[1:]])
            ax=fig.add_subplot(4,1,n)
            ax.set_title(key)
            ax.plot(y,'.-')
            if key in ['loss','tau_p','tau_f','p_contrib','f_contrib']: ax.semilogy()
            if key in ['evi']: ax.set_ylim(np.amin(y)*1.001,np.amin(y)+5.*(np.percentile(y,50)-np.amin(y)))
            n+=1
        if fname is not None: 
            plt.savefig(fname)
        else:
            plt.show()
        plt.close(fig)




