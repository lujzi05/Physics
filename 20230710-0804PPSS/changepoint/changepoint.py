"""============================================================================
Author: Gregory Gundersen

Python implementation of Bayesian online changepoint detection for a normal
model with unknown mean parameter. For algorithm details, see

    Adams & MacKay 2007
    "Bayesian Online Changepoint Detection"
    https://arxiv.org/abs/0710.3742

For Bayesian inference details about the Gaussian, see:

    Murphy 2007
    "Conjugate Bayesian analysis of the Gaussian distribution"
    https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf

This code is associated with the following blog posts:

    http://gregorygundersen.com/blog/2019/08/13/bocd/
    http://gregorygundersen.com/blog/2020/10/20/implementing-bocd/
============================================================================"""

import matplotlib.pyplot as plt
from   matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
from   scipy.stats import norm, tstd, binom
from   scipy.special import logsumexp
from datetime import date
import os.path
import warnings


# -----------------------------------------------------------------------------

def bocd(data, model, hazard):
    """Return run length posterior using Algorithm 1 in Adams & MacKay 2007.
    """
    # 1. Initialize lower triangular matrix representing the posterior as
    #    function of time. Model parameters are initialized in the model class.
    #    
    #    When we exponentiate R at the end, exp(-inf) --> 0, which is nice for
    #    visualization.
    #
    T           = len(data)
    log_R       = -np.inf * np.ones((T+1, T+1))
    log_R[0, 0] = 0              # log 0 == 1
    pmean       = np.empty(T)    # Model's predictive mean.
    pvar        = np.empty(T)    # Model's predictive variance. 
    log_message = np.array([0])  # log 0 == 1
    log_H       = np.log(hazard)
    log_1mH     = np.log(1 - hazard)

    for t in range(1, T+1):
        # 2. Observe new datum.
        x = data[t-1]

        # Make model predictions.
        pmean[t-1] = np.sum(np.exp(log_R[t-1, :t]) * model.mean_params[:t])
        pvar[t-1]  = np.sum(np.exp(log_R[t-1, :t]) * model.var_params[:t])
        
        # 3. Evaluate predictive probabilities.
        log_pis = model.log_pred_prob(t, x)

        # 4. Calculate growth probabilities.
        log_growth_probs = log_pis + log_message + log_1mH

        # 5. Calculate changepoint probabilities.
        log_cp_prob = logsumexp(log_pis + log_message + log_H)

        # 6. Calculate evidence
        new_log_joint = np.append(log_cp_prob, log_growth_probs)

        # 7. Determine run length distribution.
        log_R[t, :t+1]  = new_log_joint
        log_R[t, :t+1] -= logsumexp(new_log_joint)

        # 8. Update sufficient statistics.
        model.update_params(t, x)

        # Pass message.
        log_message = new_log_joint

    R = np.exp(log_R)
    return R, pmean, pvar


# -----------------------------------------------------------------------------


class GaussianUnknownMean:
    
    def __init__(self, mean0, var0, varx):
        """Initialize model.
        
        meanx is unknown; varx is known
        p(meanx) = N(mean0, var0)
        p(x) = N(meanx, varx)
        """
        self.mean0 = mean0
        self.var0  = var0
        self.varx  = varx
        self.mean_params = np.array([mean0])
        self.prec_params = np.array([1/var0])
    
    def log_pred_prob(self, t, x):
        """Compute predictive probabilities \pi, i.e. the posterior predictive
        for each run length hypothesis.
        """
        # Posterior predictive: see eq. 40 in (Murphy 2007).
        post_means = self.mean_params[:t]
        post_stds  = np.sqrt(self.var_params[:t])
        return norm(post_means, post_stds).logpdf(x)
    
    def update_params(self, t, x):
        """Upon observing a new datum x at time t, update all run length 
        hypotheses.
        """
        # See eq. 19 in (Murphy 2007).
        new_prec_params  = self.prec_params + (1/self.varx)
        self.prec_params = np.append([1/self.var0], new_prec_params)
        # See eq. 24 in (Murphy 2007).
        new_mean_params  = (self.mean_params * self.prec_params[:-1] + \
                            (x / self.varx)) / new_prec_params
        self.mean_params = np.append([self.mean0], new_mean_params)

    @property
    def var_params(self):
        """Helper function for computing the posterior variance.
        """
        return 1./self.prec_params + self.varx

# -----------------------------------------------------------------------------

def generate_data(varx, mean0, var0, T, cp_prob):
    """Generate partitioned data of T observations according to constant
    changepoint probability `cp_prob` with hyperpriors `mean0` and `prec0`.
    """
    data  = []
    cps   = []
    meanx = mean0
    for t in range(len(T)):
        if np.random.random() < cp_prob:
            meanx = np.random.normal(mean0, var0)
            cps.append(t)
        data.append(np.random.normal(meanx, varx))
    return data, cps


# -----------------------------------------------------------------------------

def plot_posterior(T, data, R, pmean, pvar, datalabel:str='', cps=None):
    plt.rcParams.update({'font.size': 30})
    fig, axes = plt.subplots(2, 1, figsize=(20,10))

    ax1, ax2 = axes

    ax1.scatter(T, data)
    ax1.plot(T, data)

    ax1.margins(0)
    
    # Plot predictions.
    ax1.plot(T, pmean, c='k')
    _2std = 2 * np.sqrt(pvar)
    ax1.plot(T, pmean - _2std, c='k', ls='--')
    ax1.plot(T, pmean + _2std, c='k', ls='--')

    ax2.imshow(np.rot90(R), aspect='auto', cmap='gray_r', 
               norm=LogNorm(vmin=0.0001, vmax=1))
    
    ax2.margins(0)

    if cps:
        for cp in cps:
            ax1.axvline(cp, c='red', ls='dotted')
            ax2.axvline(cp, c='red', ls='dotted')

    ax1.set_title(datalabel)
    ax1.set_xlabel('time [years]')
    ax1.set_ylabel('scaled data [1]')
    ax2.set_xlabel('changepoints in time [1]')
    ax2.set_ylabel(r'runlength $r_t$ [1]')
    plt.tight_layout()
    plt.show()
    plt.savefig('changepoint_'+datalabel+'.png')


# -----------------------------------------------------------------------------

def plot_data(data_dict, val=1):
    """
    Plot data of *val* intervals from a dict. 
    """

    plt.rcParams.update({'font.size': 30})
    fig, axes = plt.subplots(1, 1, figsize=(20,10))

    ax1 = axes


    ax1.margins(0)
    
    for key, data in list(data_dict.items())[1:]:
        ax1.plot(data_dict['time'], data, label=key)

    plt.legend()
    plt.title(str(val)+' day average')
    plt.tight_layout()
    plt.show()
    plt.savefig('sigma_'+str(val)+'day'+'.png')


# -----------------------------------------------------------------------------

def plot_mag(nmdb_dict:dict, eq_dict:dict, val=1, shift=0):
    """
    Plot cosmic ray data and earthquake data of *val* day intervals, cosmic ray data is shifted by *shift*.
    Note: usually, there are already rescaled
    """

    eqkey, eq = list(eq_dict.items())[0]

    eqkey = eqkey.split('_')
    for key, data in list(nmdb_dict.items())[1:]:
        plt.rcParams.update({'font.size': 30})
        fig, axes = plt.subplots(1, 1, figsize=(20,10))

        ax1 = axes

        ax1.margins(0)


        ax1.plot(nmdb_dict['time'], eq, label=eqkey[1]+' '+eqkey[0]+r'$M_w < $', color='blue')
        ax1.plot(nmdb_dict['time'], data, label=key, color='red')
        ax1.plot(nmdb_dict['time'], np.linspace(1,1,len(nmdb_dict['time'])))

        plt.legend()
        plt.grid()
        plt.xlabel('time [years]')
        plt.ylabel('scaled data [1]')
        plt.title(str(val)+' day intervals with '+str(shift)+' units shift')
        plt.tight_layout()
        plt.show()
        plt.savefig('mag_'+key+'_'+eqkey[0]+eqkey[1]+'_'+str(val)+'avg_'+str(shift)+'shift.png')


# -----------------------------------------------------------------------------


def plot_shift(sigmas, shifts, eqkey, val, rest):
    """
    Plot correlation in sigma for different shifts. 
    """

    plt.rcParams.update({'font.size': 10})
    fig, axes = plt.subplots(1, 1, figsize=(20,10))

    ax1 = axes

    ax1.margins(0)
    eqkey = eqkey.split('_')
    df = pd.DataFrame(sigmas)
    df['shifts'] = shifts
    df.plot(x='shifts')
    
    plt.xlabel('shift units ['+str(val)+' days]')
    plt.ylabel(r'difference from mean [$\sigma$]')
    plt.title('correlation with earthquakes from '+str(eqkey[1]))
    plt.grid()
    plt.show()
    plt.savefig(str(eqkey[0])+str(eqkey[1])+rest+str(val)+'average'+'_shift.png')
    



# -----------------------------------------------------------------------------


def sign_check(nmdb_dict:dict, eq_dict:dict, val:int, shift:int):
    """
    Calculating the correlation in sigmas based on the following expression:
    c = (a-1)*(b-1)
    """
    
    p = 0.5

    eqkey, eq = list(eq_dict.items())[0]
    

    if os.path.exists('./sigma_corr/'+str(eqkey)+ str(val) + 'average_' + str(shift) + 'shift_correlation.txt'):
        os.remove('./sigma_corr/'+str(eqkey)+ str(val) + 'average_' + str(shift) + 'shift_correlation.txt')

    with open('./sigma_corr/'+str(eqkey)+ str(val) + 'average_' + str(shift) + 'shift_correlation.txt', 'a') as f:
            f.write(str(eqkey) + '   ' + str(val) + ' days average' + '   ' + str(shift*val) + ' days shift' +'\n\n')

    sigma_diff = {}
    for key, data in list(nmdb_dict.items())[1:]:
        c = (np.array(data)-1)*(eq-1)
        c = c[~np.isnan(c)]
        
        sigma_diff[key] = (np.count_nonzero(np.signbit(-c)) - len(c)/2)/np.sqrt(len(c)*p*(1-p))
        cdf = binom.cdf(np.count_nonzero(np.signbit(-c)),len(c),0.5)
        with open('./sigma_corr/'+str(eqkey)+ str(val) + 'average_' + str(shift) + 'shift_correlation.txt', 'a') as f:
            f.write(key + '>>>   ' + 'positive: ' + str(np.count_nonzero(np.signbit(-c))) + 
                ',   all: ' + str(len(c)) + ',   standard dev: ' + str(np.sqrt(len(c)*p*(1-p))) + 
                ',   sigma difference: ' + str( (np.count_nonzero(np.signbit(-c)) - len(c)/2)/np.sqrt(len(c)*p*(1-p)) ) + ',   cdf: ' +str(cdf) + '\n')
        
    return sigma_diff


# -----------------------------------------------------------------------------


def get_auger_data(sigma=False, filter=False, val=1, shift=0):
    """
    Read the cosmic ray data from the Pierre Auger observatory.
    """

    data = pd.read_csv("../scalers/scalers.csv")
    data = data[['time', 'rateCorr']]
    data['time'] = pd.to_datetime(data['time'],unit='s').dt.date
    

    data = data[['time', 'rateCorr']].groupby('time').mean().reset_index()
    data['time'] = pd.to_datetime(data['time'])

    df = pd.DataFrame({'time':pd.date_range(date(2010, 1, 1), periods=4929)})
    
    result = pd.merge(df, data, how="left", on="time").fillna(0.)
    if filter:
        result['rateCorr'] = result['rateCorr'].where((result['rateCorr']!=0.) & (result['rateCorr'] < 2000), np.nan)

    data = result['rateCorr'].to_numpy()
    data_dict = {}
    data_dict['Pierre Auger'] = data

    if sigma:
        data_dict = make_sigma(data_dict, val, shift)

    return data_dict

# -----------------------------------------------------------------------------


def get_earthquake_data(file, sigma=False, val=1):
    """
    Read the earthquake data.
    """

    name = file.split('-')[1][:].split('.csv')[0]
    eq = pd.read_csv(file)
    eq['time']= pd.to_datetime(eq['time']).dt.date
    
    eq = eq[['time', 'mag']].groupby('time').sum().reset_index() #mean()
    eq['time'] = pd.to_datetime(eq['time'])

    # saving the data
    """
    savedf = eq[['time', 'mag']].groupby('time').sum().reset_index()
    savedf['time'] = pd.to_datetime(savedf['time'])
    savedf.to_csv(name+'_1day_sum.csv')
    """

    df = pd.DataFrame({'time':pd.date_range(date(2010, 1, 1), periods=4929)})
    
    result = pd.merge(df, eq, how="outer", on="time").fillna(np.nan)
    data = result['mag'].to_numpy()
    data_dict = {}
    data_dict[name] = data

    if sigma:
        data_dict = make_sigma_eq(data_dict, val)
    
    return data_dict
    


# -----------------------------------------------------------------------------


def get_nmdb_data(file, filter=False, sigma=False, val=1, shift=0):
    """
    Read the cosmic ray data from Neutron Monitor Database (NMDB).
    """

    with open(file) as f:
        l = f.readlines()

    lines=[]
    for i in range(len(l[:])):
        if l[i][0] != '#':
            lines.append(l[i].strip().split(';'))
    nmdb = pd.DataFrame(lines[1:], columns=('time '+lines[0][0]).split()) #27
    nmdb['time'] = pd.to_datetime(nmdb.time).dt.date
    nmdb = nmdb.replace('   null', 0)
    nmdb.iloc[:, 1:] = nmdb.iloc[:, 1:].apply(pd.to_numeric)
    
    with np.errstate(invalid='ignore'):
        if filter:
            nmdb.iloc[:,1:] = nmdb.iloc[:,1:].where((nmdb.iloc[:,1:]!=0.) & (nmdb.iloc[:,1:] < 2000), np.nan)
            
    data_dict = nmdb.to_dict('list')
    if sigma:
        data_dict = make_sigma(data_dict, val, shift)
        
    return data_dict


# -----------------------------------------------------------------------------


def get_nmdb_hr(file, filter=False, sigma=False):
    """
    Read the high resolution cosmic ray data.
    //currently not used, work in progress//
    """
    with open(file) as f:
        l = f.readlines()

    lines=[]
    for i in range(len(l[33:])):
        lines.append(l[i+33].strip().split(';'))

    nmdb = pd.DataFrame(lines, columns=('time'+l[32]).split())
    nmdb['time'] = pd.to_datetime(nmdb.time).dt.date
    nmdb = nmdb.replace('   null', np.nan)
    nmdb.iloc[:, 1:] = nmdb.iloc[:, 1:].apply(pd.to_numeric)
    

    fig, axes = plt.subplots(1, 1, figsize=(20,10))

    ax1 = axes
    ax1.margins(0)

    labels = ['mxco', 'rome', 'mosc', 'newk', 'mgdn']
    for i in range(len(nmdb.columns[1:])):
        ax1.plot(np.linspace(0,len(nmdb),len(nmdb)), nmdb[nmdb.columns[1+i]], label=labels[i])

    plt.legend()
    plt.title('minutes')
    plt.tight_layout()
    plt.show()
    plt.savefig('cr.png')
    return nmdb

# -----------------------------------------------------------------------------


def make_sigma(data_dict, val, shift):
    """
    Rescaling the cosmic ray data:
    average of *val* day intervals then divided by its median
    shift the data by *shift*
    """

    for key, data in data_dict.items():
        if key == 'time': 
            time_data = []
            for i in range(int(len(data)/val)-1):
                time_data.append(data[val*i] + (data[val*i+val]-data[val*i]) / 2)
            data_dict['time'] = time_data       
        
        else:
            sigma_data = []
            for i in range(int(len(data)/val)-1):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    sigma_data.append(np.abs(np.nanmean(data[val*(i+1):val*(i+1)+val])-np.nanmean(data[val*i:val*i+val])))
            sigma_data = np.array(sigma_data)
            median = np.nanmedian(sigma_data)
            sigma_data = sigma_data/median
            
            sigma_data = make_shift(sigma_data, shift)
            
        
            data_dict[key] = sigma_data
    
    return data_dict


# -----------------------------------------------------------------------------


def make_sigma_eq(data_dict:dict, val:int):
    """
    Rescaling the earthquake data:
    summation of *val* day intervals then divided by its median
    """

    key = list(data_dict.keys())[0]
    data = list(data_dict.values())[0]
    sigma_data = []
    for i in range(int(len(data)/val)-1):
        sigma_data.append(np.nansum(data[val*(i):val*(i)+val]))

    sigma_data = np.array(sigma_data)
    sigma_data[sigma_data==0] = np.nan
    median = np.nanmedian(sigma_data)

    sigma_data = sigma_data/median
    data_dict[key] = sigma_data

    return data_dict

# -----------------------------------------------------------------------------



def cdf():
    """
    Calculate cdf.
    //currently not used//
    """
    # No of Data points
    N = 500
      
    # initializing random values
    data = np.random.randn(N)
      
    # getting data of the histogram
    count, bins_count = np.histogram(data, bins=10)
    
    # finding the PDF of the histogram using count values
    pdf = count / sum(count)
      
    # using numpy np.cumsum to calculate the CDF
    # We can also find using the PDF values by looping and adding
    cdf = np.cumsum(pdf)

      
    # plotting PDF and CDF
    fig, axes = plt.subplots(1, 1, figsize=(20,10))
    ax1 = axes
    ax1.plot(bins_count[1:], pdf, color="red", label="PDF")
    ax1.plot(bins_count[1:], cdf, label="CDF")
    plt.legend()
    plt.show()
    plt.savefig('cdf.png')


# -----------------------------------------------------------------------------


def make_shift(data, num):
    """
    shift an array by *num* and fill the empty places with np.nan
    """

    result = np.empty_like(data)
    if num > 0:
        result[:num] = np.nan
        result[num:] = data[:-num]
    elif num < 0:
        result[num:] = np.nan
        result[:num] = data[-num:]
    else:
        result[:] = data
    return result








# -----------------------------------------------------------------------------


if __name__ == '__main__':
    val=5 # length of intervals
    f = True # filter out the outliners from cosmic ray data
    sigma = True # rescaling data


    data = get_nmdb_data('../nmdb2010010120230630_PSNM_DJON_TSMB_ATHN_MXCO_NANM_HRMS_NEWK_MGDN_OULU.csv', filter=f, sigma=sigma, val=val, shift=0)
    auger = get_auger_data(sigma=sigma, filter=f, val=val, shift=0)
    data.update(auger)
    eq = get_earthquake_data('../2010010120230630-4_Mexico.csv' ,sigma=sigma, val=val)

    """
    savedf = pd.DataFrame({'time' : [], 'mag': []})
    savedf['mag'] = list(eq.values())[0]
    savedf['time'] = data['time']
    savedf.to_csv('4_Turkey_5day_sum.csv')
    """


 
    # Bayesian changepoint analysis
    bayesian = eq # dictionary containing the data or None

    T      = np.arange(0, 1000)   # Number of observations.
    hazard = 1/100  # Constant prior on changepoint probability.
    mean0  = 0      # The prior mean on the mean parameter.
    var0   = 2      # The prior variance for mean parameter.
    varx   = 1      # The known variance of the data.
    if type(bayesian)==dict:
        model          = GaussianUnknownMean(mean0, var0, varx)
        # select the value and the datalabel from the dictionary for the bayesian changepoint analysis
        R, pmean, pvar = bocd(list(bayesian.values())[0], model, hazard) 
        plot_posterior(data['time'], list(bayesian.values())[0], R, pmean, pvar, datalabel="Mexico")

    else:
        data, cps      = generate_data(varx, mean0, var0, T, hazard)
        model          = GaussianUnknownMean(mean0, var0, varx)
        R, pmean, pvar = bocd(data, model, hazard)
        plot_posterior(T, data, R, pmean, pvar, cps=cps)










    # TODO: making 3D plot for different averagings and different shifts
    
    corr = True
    if corr:
        # calculate correlation in sigma for different datashifts
        # making data snippets
        start = 0
        end = 300000000 

        
        # japan
        # 1: 300-600
        # 3: 0-300
        

        # mexico
        # 1:

        
        eq = get_earthquake_data('../2010010120230630-5.1_world.csv' ,sigma=sigma, val=val)
        eqkey = list(eq.keys())[0]
        eq_snipp = {}
        eq_snipp[list(eq.keys())[0]] = list(eq.values())[0][start:end]

        shifts = np.linspace(-10,30,41)
        sigmas = []
        sigmas_rest = []
        for s in shifts:
            shift = int(s)

            data = get_nmdb_data('../nmdb2010010120230630_PSNM_DJON_TSMB_ATHN_MXCO_NANM_HRMS_NEWK_MGDN_OULU.csv', filter=f, sigma=sigma, val=val, shift=shift)
            auger = get_auger_data(sigma=sigma, filter=f, val=val, shift=shift)
            data.update(auger)


            data_trunc = {}
            data_trunc_rest = {}
            data_trunc['time'] = data['time']
            # separating detectors based on locations
            japan = ['DJON', 'PSNM', 'MGDN']
            mexico = ['MXCO', 'NEWK', 'auger']
            turkey = ['ATHN', 'NANM', 'OULU']
            rest = ['TSMB', 'HRMS']
            for key, value in data.items():
                if key in mexico: # select detector group
                    data_trunc[key] = value
                else:
                    data_trunc_rest[key] = value
     
            data_snipp = {}
            for k, v in data_trunc.items():
                data_snipp[k] = v[start:end]

            data_snipp_rest = {}
            for k, v in data_trunc_rest.items():
                data_snipp_rest[k] = v[start:end]


            
            #plot_mag(data_snipp, eq_snipp, val=val, shift=shift)
            #plot_mag(data_snipp_rest, eq_snipp, val=val, shift=shift)


            sigmas.append(sign_check(data_snipp,eq_snipp, val=val, shift=shift))
            sigmas_rest.append(sign_check(data_snipp_rest,eq_snipp, val=val, shift=shift))

        plot_shift(sigmas, shifts, eqkey, val=val, rest='')
        plot_shift(sigmas_rest, shifts, eqkey, val=val, rest='rest')


        
        plot_data(data_snipp, val=val)
        




