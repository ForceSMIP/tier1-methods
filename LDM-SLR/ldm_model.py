#!/usr/bin/env python
# coding: utf-8

import scipy.linalg
import numpy as np
from copy import deepcopy

def logdet_symmetric_positive(A):
    if A.size==0: return 0.
    try:
        ret=2.*np.sum(np.log(np.diag(scipy.linalg.cholesky(A))))
    except(np.linalg.LinAlgError):
        ret=1e+31
    except(ValueError):    
        ret=1e+31
    return ret

def log_prior_gaussian(x,sigma):
    if x.size==0: return 0.
    ret=-0.5*np.sum(((x/sigma)**2)+np.log(2.0*np.pi*(sigma**2)))
    return ret    

def get_hessian_p_elements(alpha,sigma_p):
    H11=1./((1.-alpha**2)*(sigma_p**2))
    H12=-alpha*H11
    H22=(1.+alpha**2)*H11
    return H11,H12,H22

def get_full_hessian_diags(n,alpha,sigma_p):
    if n<2: exit()
    H11,H12,H22=get_hessian_p_elements(alpha,sigma_p)
    diag1=np.ones(n)*H22
    diag1[0]=H11
    diag1[-1]=H11
    diag2=np.ones(n-1)*H12
    return diag1,diag2

def get_full_hessian(n,alpha,sigma_p):
    diag1,diag2=get_full_hessian_diags(n,alpha,sigma_p)
    return scipy.sparse.diags([diag1,diag2,diag2],offsets=[0,1,-1],shape=(n,n)).toarray()    


class ldm_model:           
    def __init__(self):
        return
    
    def init(self,y_train,dp,df,tau_p,tau_f,sigma_p,sigma_f,sigma_c,model_ids=None):
        '''

        Parameters
        ----------
        y_train : np.array[nsamples,n,dout]
            Pre-computed ensemble PCs.
        dp : int >=0
            dim(p(t)).
        df : int >=0
            dim(f(t)).
        tau_p : np.array[dp]
        
        tau_f : np.array[df]
        
        sigma_p : np.array[dp]
        
        sigma_f : np.array[df]
        
        sigma_c : float
        
        model_ids: optional, None or np.array[nmodels] with dtype=int, filled with the values from {0,1,2,...,nmodels-1}.
            It indicates that the forced component time series will be model-wise, where the number of the model is stored in model_ids.
            None is equivalent to np.zeros(0,dtype=int), i.e. the case of all realizations coming from one model.
        
        -------

        '''

        y2s=np.sqrt((np.reshape(y_train,(-1,y_train.shape[-1]))**2).mean(axis=0))
        yvar=(y_train**2).mean()
        y_train_norm=y_train/y2s
        self.init_norm(y_train_norm,y2s,yvar,dp,df,tau_p,tau_f,sigma_p,sigma_f,sigma_c,model_ids)
    
    def init_norm(self,y_train_norm,y2s,yvar,dp,df,tau_p,tau_f,sigma_p,sigma_f,sigma_c,model_ids):
        nsamples,n,dout=y_train_norm.shape  
        
        self.nsamples=nsamples
        self.n=n
        self.dout=dout
        self.dp=dp
        self.df=df
        self.nnd=self.nsamples*self.n*self.dout

        self.sigma_A=1
        self.sigma_B=1.
        self.sigma_c=sigma_c
        
        self.tau_p=tau_p.copy()
        self.alpha_p=np.exp(-1./tau_p)
        self.sigma_p=sigma_p.copy()
        self.tau_f=tau_f.copy()
        self.alpha_f=np.exp(-1./tau_f)
        self.sigma_f=sigma_f.copy()

        self.y2s=y2s.copy()
        self.yvar=yvar
        self.y_train_norm=y_train_norm
        
        if model_ids is None: model_ids=np.zeros(self.nsamples,dtype=int)
        self.model_ids=model_ids
        self.models=np.unique(self.model_ids)
        self.nmodels=self.models.shape[0]
        self.model_sizes=[]
        for imodel in range(self.nmodels):
            self.model_sizes.append((self.model_ids==imodel).sum())
        
        self.init_solver()    
        
    def init_solver(self):
        if self.dp>0: 
            diag1=np.zeros(self.n*self.dp)
            diag2=np.zeros((self.n-1)*self.dp)          
            for i in range(self.dp):
                H=get_full_hessian(n=self.n,alpha=self.alpha_p[i],sigma_p=self.sigma_p[i])             
                diag1[i::self.dp]=np.diagonal(H)
                diag2[i::self.dp]=np.diagonal(H,offset=1)
            self.H_prior_p=scipy.sparse.diags([diag1,diag2,diag2],
                                     offsets=[0,self.dp,-self.dp]
                                     ,shape=(self.n*self.dp,self.n*self.dp)).toarray()
        else:
            self.H_prior_p=np.zeros((self.n*self.dp,self.n*self.dp))
            
        if self.df>0:                      
            diag1=np.zeros(self.n*self.df)
            diag2=np.zeros((self.n-1)*self.df)
            for i in range(self.df):
                H=get_full_hessian(n=self.n,alpha=self.alpha_f[i],sigma_p=self.sigma_f[i])                
                diag1[i::self.df]=np.diagonal(H)
                diag2[i::self.df]=np.diagonal(H,offset=1)
        
            self.H_prior_f=scipy.sparse.diags([diag1,diag2,diag2],
                                     offsets=[0,self.df,-self.df]
                                     ,shape=(self.n*self.df,self.n*self.df)).toarray()            
        else:
            self.H_prior_f=np.zeros((self.n*self.df,self.n*self.df))
         
    def save(self):
        pars={}
        for key in ['y_train_norm','y2s','yvar','dp','df','tau_p','tau_f','sigma_p','sigma_f','sigma_c','model_ids']:
            pars[key]=deepcopy(self.__dict__[key])
        pars['model_pars']=self.get_pars()
        return pars
    
    def load(self,pars):
        for key in pars.keys():
            self.__dict__[key]=deepcopy(pars[key])  
        self.init_norm(pars['y_train_norm'],pars['y2s'],pars['yvar'],
                       pars['dp'],pars['df'],pars['tau_p'],pars['tau_f'],
                       pars['sigma_p'],pars['sigma_f'],pars['sigma_c'],pars['model_ids'])
        self.set_pars(pars['model_pars'])

    def gen_pars(self,init=None,coeff=0.,p_mask=None,f_mask=None):
        '''
        If you need to initialize with known parameters, pass them with 'init'; 
        otherwise, if init is None, the parameters are generated randomly.
        If init is not None, the parameters are generated as a perturbation of init.
        The power of perturbation is expressed by coeff. Default coeff=0 corresponds to no-perturbation.
        
        Parameters
        ----------
        init : None or dictionary with model parameters to initialize
        coeff : float (min is 0., max is 1.), optional
            The default is 0..
        
        if p_mask and/or f_mask is not None, only the axes of init which are True in the boolean arrays p_mask[dp] and f_mask[df] will be perturbed

        Returns
        -------
        None.

        '''
        
        self.c=np.random.normal(size=(self.dout))*self.sigma_c 
        self.A=np.random.normal(size=(self.dout,self.dp))*self.sigma_A
        self.B=np.random.normal(size=(self.dout,self.df))*self.sigma_B

        self.p=np.zeros((self.nsamples,self.n,self.dp))
        sigma2_rand=self.sigma_p*np.sqrt(1.-self.alpha_p**2)       
        self.p[:,0,:]=np.random.normal(size=(self.nsamples,self.dp))*self.sigma_p
        for i in range(1,self.n):
            self.p[:,i,:]=self.alpha_p*self.p[:,i-1,:]+sigma2_rand*np.random.normal(size=(self.nsamples,self.dp))                   

        self.f=np.zeros((self.nmodels,self.n,self.df))
        sigma2_rand=self.sigma_f*np.sqrt(1.-self.alpha_f**2)       
        self.f[:,0,:]=np.random.normal(size=(1,self.df))*self.sigma_f
        for i in range(1,self.n):
            self.f[:,i,:]=self.alpha_f*self.f[:,i-1,:]+sigma2_rand*np.random.normal(size=(self.nmodels,self.df)) 
        
        self.g=np.random.uniform()
        
        if init is not None:
            
            Acoeff=np.zeros(self.dp)
            if p_mask is None: p_mask=np.full((self.dp),True)
            Acoeff[p_mask]=coeff
            self.A=init['A']+(self.A-init['A'])*Acoeff
            
            Bcoeff=np.zeros(self.df)
            if f_mask is None: f_mask=np.full((self.df),True)
            Bcoeff[f_mask]=coeff
            self.B=init['B']+(self.B-init['B'])*Bcoeff
            
            self.c=init['c']+(self.c-init['c'])*coeff 
            
            self.p=init['p']+(self.p-init['p'])*Acoeff
            self.f=init['f']+(self.f-init['f'])*Bcoeff
            

            self.g=np.abs(init['g'])+(self.g-init['g'])*coeff
            if self.g<=1e-8: self.g=1e-8

    def get_pars(self):
        return {'A':self.A.copy(),'B':self.B.copy(),'c':self.c.copy(),'p':self.p.copy(),'f':self.f.copy(),'g':self.g.copy(),'y2s':self.y2s.copy(),'yvar':self.yvar.copy()}

    def set_pars(self,pars):
        keys=pars.keys()
        self.A=pars['A'].copy()
        self.B=pars['B'].copy()
        self.c=pars['c'].copy()
        self.p=pars['p'].copy()
        self.f=pars['f'].copy()
        self.g=pars['g'].copy()
        if 'y2s' in keys: self.y2s=pars['y2s'].copy()
        if 'yvar' in keys: self.yvar=pars['yvar'].copy()

    def log_prior(self):
        lpA=log_prior_gaussian(self.A,self.sigma_A)
        lpB=log_prior_gaussian(self.B,self.sigma_B)
        lpc=log_prior_gaussian(self.c,self.sigma_c)

        lpp=log_prior_gaussian(self.p[:,0,:],self.sigma_p)
        lpp+=log_prior_gaussian(self.p[:,1:,:]-(self.alpha_p*self.p[:,:-1,:]),self.sigma_p*np.sqrt(1-self.alpha_p**2))        

        lpf=log_prior_gaussian(self.f[:,0,:],self.sigma_f)
        lpf+=log_prior_gaussian(self.f[:,1:,:]-(self.alpha_f*self.f[:,:-1,:]),self.sigma_f*np.sqrt(1-self.alpha_f**2)) 
        
        lpg=0.
        
        return lpA+lpB+lpc+lpp+lpf+lpg
    
    def call_ldm_normalized(self,p,f):
        return self.c+np.matmul(p,self.A.T)+np.matmul(f,self.B.T)
 
    def call_ldm(self,p,f):
        return self.call_ldm_normalized(p,f)*self.y2s    

    def call_A(self):
        return self.A*self.y2s[:,None]

    def call_B(self):
        return self.B*self.y2s[:,None]

    def call_c(self):
        return self.c*self.y2s
    
    def call_sigma(self):
        return (self.g*np.sqrt(self.yvar))
    
    def call_p(self):
        return self.p.copy()
    
    def call_f(self):
        return self.f.copy()    
    
    def call_f_extended(self):
        return self.f.copy()[self.model_ids,:,:] 

    def get_contributions(self):
        data_stds=(self.y_train_norm.reshape((self.nsamples*self.n,-1)).std(axis=0)*self.y2s)
        data_var=(data_stds**2).sum()
        
        p=np.zeros((1,self.dp,self.dp))+np.eye(self.dp)*self.call_p().reshape((self.nsamples*self.n,self.dp)).std(axis=0)
        forcing=np.zeros((1,self.dp,self.df))
        ldm_p_vars=(self.call_ldm(p,forcing)**2)[0,:,:].sum(axis=-1)
        ldm_p_contrib=ldm_p_vars/data_var

        p=np.zeros((1,self.df,self.dp))
        forcing=np.zeros((1,self.df,self.df))+np.eye(self.df)*self.call_f().reshape((self.nmodels*self.n,self.df)).std(axis=0)
        ldm_f_vars=(self.call_ldm(p,forcing)**2)[0,:,:].sum(axis=-1)
        ldm_f_contrib=ldm_f_vars/data_var        
        
        return ldm_p_contrib, ldm_f_contrib

    def loss(self):
        yout=self.call_ldm_normalized(self.p,self.f[self.model_ids,:,:])
        return np.mean((yout - self.y_train_norm)**2,axis=-2)        

    def log_likelihood_nnd(self):
        loss=self.loss()
        g2=self.g**2
        log_lkh=-0.5*np.mean(loss*((self.y2s**2)/self.yvar)/g2+np.log(2.*np.pi*g2*self.yvar))
        return log_lkh
    
    def log_posterior_nnd(self):
        return self.log_likelihood_nnd()+self.log_prior()/self.nnd        
    
    def loss_average(self):
        loss=self.loss()
        loss_average_norm=(loss*(self.y2s**2)).mean()/self.yvar
        return loss_average_norm 
    
    def cost(self):
        return -self.log_posterior_nnd()

    def minus_log_evidence_noise(self):
        sumln=np.sum(np.log(self.y2s**2))
        enoise=0.5*float(self.dout)*(1+np.log(2.*np.pi))+0.5*sumln+0.5*sumln/float(self.nsamples*self.n)
        enoise+=0.5*(float(self.dout)*np.log(2.*float(self.nsamples*self.n)/2./np.pi)-sumln)/float(self.nsamples*self.n)
        return enoise

    def get_hessian_blocks_pf(self):
        A,B,c=self.A,self.B,self.c
        g2=self.g**2
        gamma2=(self.y2s**2)/self.yvar/g2
        AA=np.matmul(A.T,gamma2[:,None]*A)
        AB=np.matmul(A.T,gamma2[:,None]*B)
        BB=np.matmul(B.T,gamma2[:,None]*B)
        
        if self.dp>0:
            #[sum[dims],sum[dims]]
            Qpp=scipy.sparse.block_diag([AA]*self.n).toarray()
            Qpp=Qpp+self.H_prior_p
            
            Qpp_banded=np.zeros((self.dp+1,Qpp.shape[-1]))
            for i in range(Qpp_banded.shape[0]):
                Qpp_banded[Qpp_banded.shape[0]-i-1,i:]=np.diagonal(Qpp,offset=i)
        else:
            Qpp=np.zeros((0,0))
            Qpp_banded=np.zeros((0,0))
            
        if self.dp>0 and self.df>0:
            Qpf=scipy.sparse.block_diag([AB]*self.n).toarray()
        else:
            Qpf=np.zeros((self.dp*self.n,self.df*self.n))            
    
        if self.df>0:
            Hff=scipy.sparse.block_diag([BB]*self.n).toarray()
        else:
            Hff=np.zeros((0,0))    
        
        return Qpp_banded, Qpf, Hff
        
      

    def get_logdethessian_pf_short_duration(self):
        Qpp_banded, Qpf, Hff=self.get_hessian_blocks_pf()
 
        if self.dp>0:
            #[sumdimsp,sumdimsf]
            U=scipy.linalg.solveh_banded(Qpp_banded,Qpf)
        else:
            #[sumdimsp,sumdimsf]
            U=np.zeros((0,self.df*self.n))

        if Qpp_banded.size==0: 
            ret=0.
        else:
            ret=2.*np.sum(np.log(np.abs(scipy.linalg.cholesky_banded(Qpp_banded/self.nnd)[-1,:])))
                
        ret=self.nsamples*ret
        W_core=(Hff-(np.matmul(Qpf.T,U)))
        
        for imodel in range(self.nmodels):

        #[sumdimsf,sumdimsf]
            W=W_core*self.model_sizes[imodel]+self.H_prior_f     
            ret=ret+logdet_symmetric_positive(W/self.nnd)
        
        M=Qpp_banded.shape[-1]*self.nsamples+Hff.shape[-1]*self.nmodels
        return M, ret
    

   
    def fit_p_analytic(self):
        A,B,c=self.A,self.B,self.c
        g2=self.g**2
        gamma2=(self.y2s**2)/self.yvar/g2
        AA=np.matmul(A.T,gamma2[:,None]*A)
        forcing=self.f
        
        if self.dp>0:
            #[sum[dims],sum[dims]]
            Qpp=scipy.sparse.block_diag([AA]*self.n).toarray()
            Qpp=Qpp+self.H_prior_p
            
            Qpp_banded=np.zeros((self.dp+1,Qpp.shape[-1]))
            for i in range(Qpp_banded.shape[0]):
                Qpp_banded[Qpp_banded.shape[0]-i-1,i:]=np.diagonal(Qpp,offset=i)
            
            bp=-np.tensordot((self.y_train_norm-c-np.matmul(forcing[self.model_ids,:,:],B.T))*gamma2,A,axes=(-1,-2))
            bp=bp.reshape((self.nsamples,-1))

            #[sumdimsp,nsamples]
            Pout=scipy.linalg.solveh_banded(Qpp_banded,-bp.T)
            self.p=(Pout.T).reshape((self.nsamples,self.n,self.dp))
        else:
            self.p=np.zeros((self.nsamples,self.n,self.dp))
                         
    def fit_f_analytic(self):
        A,B,c=self.A,self.B,self.c
        g2=self.g**2
        gamma2=(self.y2s**2)/self.yvar/g2
        BB=np.matmul(B.T,gamma2[:,None]*B)
        p=self.p
    
        if self.df>0:
            Qff=scipy.sparse.block_diag([BB]*self.n).toarray()
            
            Qff_banded=np.zeros((self.df+1,Qff.shape[-1]))
            for i in range(Qff_banded.shape[0]):
                Qff_banded[Qff_banded.shape[0]-i-1,i:]=np.diagonal(Qff,offset=i)
            Hff_banded=np.zeros((self.df+1,Qff.shape[-1]))
            for i in range(Qff_banded.shape[0]):
                Hff_banded[Hff_banded.shape[0]-i-1,i:]=np.diagonal(self.H_prior_f,offset=i)
            bf=-np.tensordot((self.y_train_norm-c-np.matmul(p,A.T))*gamma2,B,axes=(-1,-2))
            
            
            self.f=[]
            for imodel in range(self.nmodels):
                bf_tmp=bf[self.model_ids==imodel,...].sum(axis=0).reshape(-1)
                Fout=scipy.linalg.solveh_banded(Qff_banded*self.model_sizes[imodel]+Hff_banded,-bf_tmp)
                self.f.append(Fout.reshape((self.n,self.df)))
            self.f=np.stack(self.f,axis=0)
        else:
            self.f=np.zeros((self.nmodels,self.n,self.df))

    def get_gradient_hessian_blocks_a(self):
        beta=[np.ones((self.nsamples,self.n,1)),self.p]
        beta=beta+[self.f[self.model_ids,:,:]]
        #[nsamples,n,dimbeta]
        beta=np.concatenate(beta,axis=-1)
        
        beta=beta.reshape((-1,beta.shape[-1]))
        #[dimbeta,dimbeta]
        BB=np.matmul(beta.T,beta)
        
        #[dout,dimbeta]
        BY=np.matmul(self.y_train_norm.reshape((-1,self.dout)).T,beta)
        
        return BB, BY  
    
    def get_logdethessian_a(self):
        BB, BY=self.get_gradient_hessian_blocks_a()
        g2=self.g**2
        gamma2=(self.y2s**2)/self.yvar/g2
        sigmas=np.concatenate([np.ones(1)*self.sigma_c,
                np.ones(self.dp)*self.sigma_A,
                np.ones(self.df)*self.sigma_B])
        Hprior=np.diag(1./sigmas**2)
        
        ret2=0.
        for i in range(self.dout):
            ret2+=logdet_symmetric_positive((BB*gamma2[i]+Hprior)/self.nnd)
        
        M=self.dout*sigmas.shape[-1]
        return M, ret2
    
    def fit_a_analytic(self):
        BB, BY=self.get_gradient_hessian_blocks_a()
        g2=self.g**2
        gamma2=(self.y2s**2)/self.yvar/g2
        sigmas=np.concatenate([np.ones(1)*self.sigma_c,
                np.ones(self.dp)*self.sigma_A,
                np.ones(self.df)*self.sigma_B])
        Hprior=np.diag(1./sigmas**2)
        
        a=np.zeros((self.dout,BB.shape[-1]))
        for i in range(self.dout):
            hess=(BB+Hprior/gamma2[i])/self.nnd
            grad=-BY[i,:]/self.nnd
            a[i,:]=scipy.linalg.solve(hess,-grad,assume_a='pos')
        self.A=a[:,1:1+self.dp]
        self.B=a[:,1+self.dp:]
        self.c=a[:,0]
        return 
        
    def get_logdethessian_g(self):
        c=self.loss_average()
        g2=self.g**2
        ret=np.log((3.*c/g2-1.)/g2)
        M=1
        return M, ret
    
    def fit_g_analytic(self):
        self.g=np.sqrt(self.loss_average())
        return                     
    
    def fit_pars(self,eps=1e-10):
        def func(): return self.cost()      
     
        f2=func()
        fval=[]
        flag=True
        while flag:           

            for i in range(1):
                for i in range(1): 
                    self.fit_p_analytic()
                    self.fit_f_analytic()
                for i in range(1): self.fit_a_analytic()
                for i in range(1): self.fit_g_analytic()
            
            f1=f2.copy()            
            f2=func()
            fval.append(f2) 
            
            if f1-f2<eps: flag=False
        
        return len(fval)   

    def fit_pf_iterative(self,eps=1e-10):
        def func(): return self.cost()      
     
        f2=func()
        fval=[]
        flag=True
        while flag:           

            for i in range(1):
                for i in range(1): 
                    self.fit_p_analytic()
                    self.fit_f_analytic()
            
            f1=f2.copy()            
            f2=func()
            fval.append(f2) 
            
            if f1-f2<eps: flag=False
        
        return len(fval) 
    
    def fit_pa_iterative(self,eps=1e-10):
        def func(): return self.cost()      
     
        f2=func()
        fval=[]
        flag=True
        while flag:           

            for i in range(1):
                for i in range(1): 
                    self.fit_p_analytic()
                for i in range(1): self.fit_a_analytic()
                for i in range(1): self.fit_g_analytic()
            
            f1=f2.copy()            
            f2=func()
            fval.append(f2) 
            
            if f1-f2<eps: flag=False
        
        return len(fval)      
           
    
    def minus_log_evidence_nn_pars(self):
    
        M_f,logdet_f=self.get_logdethessian_a()     
        M_p,logdet_p=self.get_logdethessian_pf_short_duration()    
        M_g,logdet_g=self.get_logdethessian_g()
        
        log_det=logdet_f+logdet_p+logdet_g
        M=float(M_f+M_p+M_g)
        ml_evidence_nn=self.cost()*float(self.dout)+0.5*(log_det+M*np.log(self.nnd/2./np.pi))/float(self.nsamples*self.n)     
        
        enoise=self.minus_log_evidence_noise()
        return (ml_evidence_nn-enoise)





