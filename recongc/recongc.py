import numpy as np
from astropy.table import Table, vstack, Column
from scipy.interpolate import interp1d, griddata
import glob
from numpy import random
import os

pctsigma = [16., 84.]

# Wrapper class for griddata to make it easy to call with arbitrary points
class interp_wrapper(object):
    """interp_wrapper: Wrapper class for 2d interpolation.
    Initialize with:
        func = interp_wrapper(xpoints, ypoints, vals)
    where xpoints, ypoints are the locations at which vals is defined.
    To calculate the interpolated value at a set of points xc, yc (not gridded):
        interpolated_values = func(x, y)
        
    Default interpolation method is 'linear'. Other values (see scipy.interpolation.griddata)
    can be specified in the intialization (e.g. method='cubic') or changed after initializing
    using the func.change_method method."""
    
    def __init__(self, xpoints, ypoints, vals, method='linear'):
        self.points = (xpoints, ypoints)
        self.vals = vals
        self.method = method
        
    def change_method(self, newmethod):
        self.method = newmethod
        
    def __call__(self, xc, yc):
        result_shape = xc.shape
        result_flat = griddata(self.points, self.vals, np.vstack( (np.array(xc.flatten()), np.array(yc.flatten())) ).T, method=self.method)
        return np.reshape(result_flat, result_shape)
        
    
class DomainBoundary(object):
    """Defines boundary of grid points and checks if points are inside/outside the boundary.
    Note that this can only deal with boundaries that have fixed bounds in x, and then an
    upper and lower curve defined between those limits."""
    
    def __init__(self, xmin, xmax, lower_curve, upper_curve):
        """xmin and xmax are scalars that define the endpoints.
        lower_curve and upper_curve are 2-element tuples contain the x- and y-coordinates
        of the boundaries, which are interpolated between. The x-coordinates should always
        start with xmin and end with xmax. Either lower_curve or upper_curve can be None
        to eliminate the boundary."""
        self.xmin = xmin
        self.xmax = xmax
        
        # make sure lower_curve makes sense
        if lower_curve is not None:
            if len(lower_curve) != 2:
                raise ValueError("lower_curve must contain 2 elements")
            self.lower_curve_x = lower_curve[0]
            self.lower_curve_y = lower_curve[1]
            if len(self.lower_curve_x) != len(self.lower_curve_y):
                raise ValueError("lower_curve's x and y must contain the same number of elements")
            # removing these tests because of roundoff error
#             if self.lower_curve_x[0] != self.xmin:
#                 raise ValueError("lower_curve's x must begin at xmin")
#             if self.lower_curve_x[-1] != self.xmax:
#                 raise ValueError("lower_curve's x must end at xmax")
                
            # Create interpolation function
            self.lower_curve_func = interp1d(self.lower_curve_x, self.lower_curve_y, \
                bounds_error=False, fill_value=np.inf)
        else:
            self.lower_curve_func = None
        
        # ditto upper_curve
        if upper_curve is not None:
            if len(upper_curve) != 2:
                raise ValueError("upper_curve must contain 2 elements")
            self.upper_curve_x = upper_curve[0]
            self.upper_curve_y = upper_curve[1]
            if len(self.upper_curve_x) != len(self.upper_curve_y):
                raise ValueError("upper_curve's x and y must contain the same number of elements")
            # removing these tests because of roundoff error
#             if self.upper_curve_x[0] != self.xmin:
#                 raise ValueError("upper_curve's x must begin at xmin")
#             if self.upper_curve_x[-1] != self.xmax:
#                 raise ValueError("upper_curve's x must end at xmax")
                
            # Create interpolation function
            self.upper_curve_func = interp1d(self.upper_curve_x, self.upper_curve_y, \
                bounds_error=False, fill_value=-np.inf)
        else:
            self.upper_curve_func = None
            
            
    def within_bounds(self, xc, yc):
        """within_bounds(xpoints, ypoints) returns a True/False array whether each set
        of x,y points is within the boundary."""
        
        if len(xc) != len(yc):
            raise ValueError("xc and yc must contain the same number of elements")
        
        withinboundp = np.ones_like(xc, dtype=bool)
        if self.lower_curve_func:
            withinboundp *= (yc >= self.lower_curve_func(xc))
        if self.upper_curve_func:
            withinboundp *= (yc <= self.upper_curve_func(xc))
            
        return withinboundp
        
        
def MC_cloud(center, low, high, NMC=1000, force_positive=False):
    """Returns an NMC x Npts array of MC points drawn from asymmetric Gaussians.
    If force_positive is true, sets all negative values to 0."""
    Npt = len(center)
    
    # Possibly asymetric Gaussian cloud
    gaussian = random.normal(loc=0, scale=1, size=(NMC, Npt))
    whichlow = (gaussian <= 0.)
    whichhigh = (gaussian > 0.)
    
    results = np.zeros_like(gaussian)
    
    results += whichlow * (center + gaussian*(center-low))
    results += whichhigh * (center + gaussian*(high-center))
    
    if force_positive:
        negative = (results < 0.)
        results[negative] = 0.
    
    return results


class recongc(object):
	def __init__(self, altparam=None):
		# Read in model tracks
		track_directory = os.path.dirname(__file__)
		
		if altparam:
		    mean_track_fname = track_directory+'/mean-{0}_tracks.dat'.format(altparam)
		    sigmalow_track_fname = track_directory+'/sigmalow-{0}_tracks.dat'.format(altparam)
		    sigmahigh_track_fname = track_directory+'/sigmahigh-{0}_tracks.dat'.format(altparam)
		else:
		    mean_track_fname = track_directory+'/mean_tracks.dat'
		    sigmalow_track_fname = track_directory+'/sigmalow_tracks.dat'
		    sigmahigh_track_fname = track_directory+'/sigmahigh_tracks.dat'
		    
		mean_tracks = Table.read(mean_track_fname, format='ascii')
		sigmalow_tracks = Table.read(sigmalow_track_fname, format='ascii')
		sigmahigh_tracks = Table.read(sigmahigh_track_fname, format='ascii')
		self.trackZ = mean_tracks.group_by('FeHinit')
		self.nZ = len(self.trackZ.groups)
		self.Zlist = [g[0]['FeHinit'] for g in self.trackZ.groups]
		self.sigmalow_trackZ = sigmalow_tracks.group_by('FeHinit')
		self.sigmahigh_trackZ = sigmahigh_tracks.group_by('FeHinit')
		
		# Create wrapper functions for griddata: FeH-sigma plane
		self.track_Zsigma_FeHinit_func = interp_wrapper(mean_tracks['FeH'], mean_tracks['sigma_FeH'], mean_tracks['FeHinit'])
		self.track_Zsigmalow_FeHinit_func = interp_wrapper(sigmalow_tracks['FeH'], sigmalow_tracks['sigma_FeH'], sigmalow_tracks['FeHinit'])
		self.track_Zsigmahigh_FeHinit_func = interp_wrapper(sigmahigh_tracks['FeH'], sigmahigh_tracks['sigma_FeH'], sigmahigh_tracks['FeHinit'])
		
		# Minit-FeH plane
		self.track_MZ_FeHinit_func = interp_wrapper(mean_tracks['Minit'], mean_tracks['FeH'], mean_tracks['FeHinit'])
		self.track_MZlow_FeHinit_func = interp_wrapper(mean_tracks['Minit'], mean_tracks['FeH_low'], mean_tracks['FeHinit'])
		self.track_MZhigh_FeHinit_func = interp_wrapper(mean_tracks['Minit'], mean_tracks['FeH_high'], mean_tracks['FeHinit'])
		self.track_MZ_sigma_func = interp_wrapper(mean_tracks['Minit'], mean_tracks['FeH'], mean_tracks['sigma_FeH'])
		self.track_MZ_sigmalow_func = interp_wrapper(mean_tracks['Minit'], mean_tracks['FeH'], mean_tracks['sigma_FeH_low'])
		self.track_MZ_sigmahigh_func = interp_wrapper(mean_tracks['Minit'], mean_tracks['FeH'], mean_tracks['sigma_FeH_high'])
		
		# Find curves in each plane that define the edge of the computed grid
		# For Z-sigma plane, the top boundary is the low-Z curve plus the endpoints of the rest.
		# The bottom boundary is just 0s until the max-Z curve
		Zsigma_toppoints = self.trackZ.groups[0][:0]  #empty table of same format
		for g in self.trackZ.groups:
		    max_mass_index = np.argmax(g['Minit'])
		    Zsigma_toppoints.add_row(g[max_mass_index])
		FeH_topends = np.concatenate( (self.trackZ.groups[0]['FeH'], Zsigma_toppoints['FeH']) )
		sigma_topends = np.concatenate( (self.trackZ.groups[0]['sigma_FeH'], Zsigma_toppoints['sigma_FeH']) )
		FeH_bottomends = np.concatenate( ([self.trackZ.groups[0]['FeH'][0]], self.trackZ.groups[-2]['FeH']) )
		sigma_bottomends = np.zeros_like(FeH_bottomends) # just set to 0
		self.boundary_Zsigma = DomainBoundary(np.min(mean_tracks['FeH']), np.max(mean_tracks['FeH']), \
		    lower_curve=(FeH_bottomends, sigma_bottomends), upper_curve=(FeH_topends, sigma_topends))
		# ditto for sigmalow
		Zsigmalow_toppoints = self.sigmalow_trackZ.groups[0][:0]  #empty table of same format
		for g in self.sigmalow_trackZ.groups:
		    max_mass_index = -1 #np.argmax(g['Minit'])
		    Zsigmalow_toppoints.add_row(g[max_mass_index])
		FeH_topends = np.concatenate( (self.sigmalow_trackZ.groups[0]['FeH'], Zsigmalow_toppoints['FeH']) )
		sigma_topends = np.concatenate( (self.sigmalow_trackZ.groups[0]['sigma_FeH'], Zsigmalow_toppoints['sigma_FeH']) )
		FeH_bottomends = np.concatenate( ([self.sigmalow_trackZ.groups[0]['FeH'][0]], self.sigmalow_trackZ.groups[-2]['FeH']) )
		sigma_bottomends = np.concatenate( ([self.sigmalow_trackZ.groups[0]['sigma_FeH'][0]], self.sigmalow_trackZ.groups[-2]['sigma_FeH']) )
		self.boundary_Zsigmalow = DomainBoundary(np.min(sigmalow_tracks['FeH']), np.max(sigmalow_tracks['FeH']), \
		    lower_curve=(FeH_bottomends, sigma_bottomends), upper_curve=(FeH_topends, sigma_topends))
		# ditto for sigmahigh
		Zsigmahigh_toppoints = self.sigmahigh_trackZ.groups[0][:0]  #empty table of same format
		for g in self.sigmahigh_trackZ.groups:
		    max_mass_index = -1 #np.argmax(g['Minit'])
		    Zsigmahigh_toppoints.add_row(g[max_mass_index])
		FeH_topends = np.concatenate( (self.sigmahigh_trackZ.groups[0]['FeH'], Zsigmahigh_toppoints['FeH']) )
		sigma_topends = np.concatenate( (self.sigmahigh_trackZ.groups[0]['sigma_FeH'], Zsigmahigh_toppoints['sigma_FeH']) )
		FeH_bottomends = np.concatenate( ([self.sigmahigh_trackZ.groups[0]['FeH'][0]], self.sigmahigh_trackZ.groups[-2]['FeH']) )
		sigma_bottomends = np.concatenate( ([self.sigmahigh_trackZ.groups[0]['sigma_FeH'][0]], self.sigmahigh_trackZ.groups[-2]['sigma_FeH']) )
		self.boundary_Zsigmahigh = DomainBoundary(np.min(sigmahigh_tracks['FeH']), np.max(sigmahigh_tracks['FeH']), \
		    lower_curve=(FeH_bottomends, sigma_bottomends), upper_curve=(FeH_topends, sigma_topends))
		    
		# For MZ plane, top boundary and bottom boundaries are just highest and lowest metallicity tracks
		self.boundary_MZ = DomainBoundary(np.min(mean_tracks['Minit']), np.max(mean_tracks['Minit']), \
		    lower_curve=(self.trackZ.groups[0]['Minit'], self.trackZ.groups[0]['FeH']), \
		    upper_curve=(self.trackZ.groups[-2]['Minit'], self.trackZ.groups[-2]['FeH']))
		self.boundary_MZlow = DomainBoundary(np.min(mean_tracks['Minit']), np.max(mean_tracks['Minit']), \
		    lower_curve=(self.trackZ.groups[0]['Minit'], self.trackZ.groups[0]['FeH_low']), \
		    upper_curve=(self.trackZ.groups[-2]['Minit'], self.trackZ.groups[-2]['FeH_low']))
		self.boundary_MZhigh = DomainBoundary(np.min(mean_tracks['Minit']), np.max(mean_tracks['Minit']), \
		    lower_curve=(self.trackZ.groups[0]['Minit'], self.trackZ.groups[0]['FeH_high']), \
		    upper_curve=(self.trackZ.groups[-2]['Minit'], self.trackZ.groups[-2]['FeH_high']))
		    

	def recon(self, dat, NMC=1000):
		# data must have columns 'FeH', optionally 'sigma_FeH', and optionally 'Minit'.
		# If it has FeH, must have FeH_low and FeH_high. If it has sigma_FeH, must have
		# sigma_FeH_low and sigma_FeH_high. If it has Minit, must have Minit_low and Minit_high.
		result = {}
		
		# Create an MC cloud with appropriate limits
		FeH_MC = MC_cloud(dat['FeH'], dat['FeH_low'], dat['FeH_high'])
		
		# Interpolate in FeH-sigma space
		if 'sigma_FeH' in dat:
		    # Nominal value
		    fehinit_sigma_nominal = self.track_Zsigma_FeHinit_func(dat['FeH'], dat['sigma_FeH'])
		    
		    # Interpolate along low and high tracks (due entirely to model stochasticity)
		    fehinit_sigmalow = self.track_Zsigmalow_FeHinit_func(dat['FeH'], dat['sigma_FeH'])
		    fehinit_sigmahigh = self.track_Zsigmahigh_FeHinit_func(dat['FeH'], dat['sigma_FeH'])
		    fehinit_sigma_modelstocherr = np.vstack( (np.abs(fehinit_sigma_nominal - fehinit_sigmalow), \
		        np.abs(fehinit_sigma_nominal - fehinit_sigmahigh)) )
		    # fehinit_sigma_modelstocherr is a 2xNpt array
		    # Rearrange to make sure that lower limit is first
		    wrongway = (fehinit_sigmalow > fehinit_sigmahigh)
		    fehinit_sigma_modelstocherr[:,wrongway] = fehinit_sigma_modelstocherr[::-1,wrongway]
		    # If NaN, it's actually zero.
		    modelstocherr_is_nan = np.isnan(fehinit_sigma_modelstocherr)
		    fehinit_sigma_modelstocherr[modelstocherr_is_nan] = 0.
		    
		    # MC cloud
		    sigma_MC = MC_cloud(dat['sigma_FeH'], dat['sigma_FeH_low'], dat['sigma_FeH_high'], force_positive=True)
		    
		    # Nominal values from MC cloud
		    fehinit_sigma_MCcloud = self.track_Zsigma_FeHinit_func(FeH_MC, sigma_MC)
		    fehinit_sigma_MClowhigh = np.nanpercentile(fehinit_sigma_MCcloud, pctsigma, axis=0)
		    fehinit_sigma_obserr = np.vstack( (np.abs(fehinit_sigma_nominal - fehinit_sigma_MClowhigh[0,:]), \
		        np.abs(fehinit_sigma_nominal - fehinit_sigma_MClowhigh[1,:])) )
		    # fehinit_sigma_obserr is a 2xNpt array
		    
		    # It turns out that the cross terms are negligible, so the total error is
		    # just those added in quadrature
		    fehinit_sigma_toterr = np.sqrt( fehinit_sigma_modelstocherr**2 + fehinit_sigma_obserr**2 )
		    # and figure out what limits those correspond to
		    fehinit_sigma_totlowhigh = fehinit_sigma_nominal + np.array([[-1,1]]).T * fehinit_sigma_toterr

            # Check if it's an extrapolation outside of model tracks
		    fehinit_sigma_extrapolated = ~self.boundary_Zsigma.within_bounds(dat['FeH'], dat['sigma_FeH'])
		    
		    # store in result
		    result['predsig_fehi'] = fehinit_sigma_nominal
		    result['predsig_fehi_extrap'] = fehinit_sigma_extrapolated
		    result['predsig_fehi_toterr'] = fehinit_sigma_toterr
		    result['predsig_fehi_totlowhigh'] = fehinit_sigma_totlowhigh
		    result['predsig_fehi_modelstocherr'] = fehinit_sigma_modelstocherr
		    result['predsig_fehi_obserr'] = fehinit_sigma_obserr
            
		    
		# Interpolate in Minit-FeH space. Calculate both FeHinit and sigma_FeH
		if 'Minit' in dat:
		    # Nominal value
		    fehinit_minit_nominal = self.track_MZ_FeHinit_func(dat['Minit'], dat['FeH'])
		    sigma_minit_nominal = self.track_MZ_sigma_func(dat['Minit'], dat['FeH'])
		    
		    # Interpolate along low and high tracks (due entirely to model stochasticity)
		    fehinit_minit_Mlow = self.track_MZlow_FeHinit_func(dat['Minit'], dat['FeH'])
		    fehinit_minit_Mhigh = self.track_MZhigh_FeHinit_func(dat['Minit'], dat['FeH'])
		    fehinit_minit_modelstocherr = np.vstack( (np.abs(fehinit_minit_nominal - fehinit_minit_Mlow), \
		        np.abs(fehinit_minit_nominal - fehinit_minit_Mhigh)) )
		    # fehinit_minit_modelstocherr is a 2xNpt array
		    # Rearrange to make sure that lower limit is first
		    wrongway = (fehinit_minit_Mlow > fehinit_minit_Mhigh)
		    fehinit_minit_modelstocherr[:,wrongway] = fehinit_minit_modelstocherr[::-1,wrongway]
		    
		    # and the same for sigma
		    sigma_minit_Mlow = self.track_MZ_sigmalow_func(dat['Minit'], dat['FeH'])
		    sigma_minit_Mhigh = self.track_MZ_sigmahigh_func(dat['Minit'], dat['FeH'])
		    sigma_minit_modelstocherr = np.vstack( (np.abs(sigma_minit_nominal - sigma_minit_Mlow), \
		        np.abs(sigma_minit_nominal - sigma_minit_Mhigh)) )
		    # sigma_minit_modelstocherr is a 2xNpt array
		    # Rearrange to make sure that lower limit is first
		    wrongway = (sigma_minit_Mlow > sigma_minit_Mhigh)
		    sigma_minit_modelstocherr[:,wrongway] = sigma_minit_modelstocherr[::-1,wrongway]
		    
		    # MC cloud
		    Minit_MC = MC_cloud(dat['Minit'], dat['Minit_low'], dat['Minit_high'])
		    
		    # Nominal values from MC cloud
		    fehinit_minit_MCcloud = self.track_MZ_FeHinit_func(Minit_MC, FeH_MC)
		    fehinit_minit_MClowhigh = np.nanpercentile(fehinit_minit_MCcloud, pctsigma, axis=0)
		    fehinit_minit_obserr = np.vstack( (np.abs(fehinit_minit_nominal - fehinit_minit_MClowhigh[0,:]), \
		        np.abs(fehinit_minit_nominal - fehinit_minit_MClowhigh[1,:])) )
		    # fehinit_minit_obserr is a 2xNpt array
		    
		    # and the same for sigma
		    sigma_minit_MCcloud = self.track_MZ_sigma_func(Minit_MC, FeH_MC)
		    sigma_minit_MClowhigh = np.percentile(sigma_minit_MCcloud, pctsigma, axis=0)
		    sigma_minit_obserr = np.vstack( (np.abs(sigma_minit_nominal - sigma_minit_MClowhigh[0,:]), \
		        np.abs(sigma_minit_nominal - sigma_minit_MClowhigh[1,:])) )
		    # sigma_minit_obserr is a 2xNpt array
		    
		    # cross terms are negligible so we can just add the observational and model error in quadrature
		    fehinit_minit_toterr = np.sqrt( fehinit_minit_modelstocherr**2 + fehinit_minit_obserr**2 )
		    fehinit_minit_totlowhigh = fehinit_minit_nominal + np.array([[-1,1]]).T * fehinit_minit_toterr
		    sigma_minit_toterr = np.sqrt( sigma_minit_modelstocherr**2 + sigma_minit_obserr**2 )
		    sigma_minit_totlowhigh = sigma_minit_nominal + np.array([[-1,1]]).T * sigma_minit_toterr
		    
		    # Check for upper limits and extrapolations
		    minit_offgrid = ~self.boundary_MZ.within_bounds(dat['Minit'], dat['FeH'])
		    
		    # store in result
		    result['predmi_offgrid'] = minit_offgrid
		    result['predmi_fehi'] = fehinit_minit_nominal
		    result['predmi_fehi_toterr'] = fehinit_minit_toterr
		    result['predmi_fehi_totlowhigh'] = fehinit_minit_totlowhigh
		    result['predmi_fehi_modelstocherr'] = fehinit_minit_modelstocherr
		    result['predmi_fehi_obserr'] = fehinit_minit_obserr
		    result['predmi_sigma'] = sigma_minit_nominal
		    result['predmi_sigma_toterr'] = sigma_minit_toterr
		    result['predmi_sigma_totlowhigh'] = sigma_minit_totlowhigh
		    result['predmi_sigma_modelstocherr'] = sigma_minit_modelstocherr
		    result['predmi_sigma_obserr'] = sigma_minit_obserr
		    
		return result
			
			