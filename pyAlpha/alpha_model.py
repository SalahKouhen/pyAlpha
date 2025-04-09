from __future__ import print_function
import numpy as np
from numpy import pi
from . import model


class AlphaModel(model.Model):
    """Alpha turbulence model."""

    def __init__(
        self,
        beta=0.,                    # gradient of coriolis parameter
        Nb = 1.,                    # Buoyancy frequency
        rd=0.,                      # deformation radius
        H = 1.,                     # depth of layer
        U=0.,                       # max vel. of base-state
        forceamp=0,                 # factor used in forcing 
        dragf=0.,                   # factor used for linear drag term
        alpha=1.,                   # alpha turbulence parameter, 1 = SQG, 2 = Euler
        fk1=4.,                     # this is the lower wavenumber where forcing present
        fk2=7.,                     # upper forcing wavenumber 
        seed=0,                     # seed, to have different results each time must be randomly set
        **kwargs
        ):
        """
        Parameters
        ----------

        beta : number
            Gradient of coriolis parameter. Units: meters :sup:`-1`
            seconds :sup:`-1`
        Nb : number
            Buoyancy frequency. Units: seconds :sup:`-1`.
        U : number
            Background zonal flow. Units: meters.
        """

        # physical
        self.beta = beta
        self.Nb = Nb
        #self.rek = rek
        self.rd = rd
        self.H = H
        self.Hi = np.array(H)[np.newaxis,...]
        self.U = U
        self.forceamp = forceamp
        self.dragf = dragf
        self.alpha = alpha
        self.fk1 = fk1
        self.fk2 = fk2
        #self.UseWisdom = UseWisdom
        #self.filterfac = filterfac
        self.seed = seed

        self.nz = 1
        
        self.rng = np.random.default_rng(self.seed)
            
        super(AlphaModel, self).__init__(**kwargs)

        # initial conditions: (PV anomalies)
        #self.set_q(1e-3*np.random.rand(1,self.ny,self.nx))
        
    ### PRIVATE METHODS - not meant to be called by user ###

    def _initialize_background(self):
        """Set up background state (zonal flow and PV gradients)."""
        
        # the meridional PV gradients in each layer
        self.Qy = np.asarray(self.beta)[np.newaxis, ...]

        # background vel.
        self.set_U(self.U)

        # complex versions, multiplied by k, speeds up computations to pre-compute
        self.ikQy = self.Qy * 1j * self.k

        self.ilQx = 0.

    def _initialize_inversion_matrix(self):
        """ the inversion """
        # The sqg inversion is ph = f / (kappa * N) qh, for Euler it is ph = 1 / (kappa**2) qh, in general ph = 1 / (kappa**alpha) qh
        self.a = np.asarray(np.sqrt(self.wv2i)**self.alpha)[np.newaxis, np.newaxis, :, :]

    def _initialize_forcing(self):
        """Set up forcing filter."""
        #SALAH
        # this defines the spectral forcing 
        wvx=((self.k)**2.+(self.l)**2.)**(0.5)
        wvxs=1e5*((self.k)**2.+(self.l)**2.)**(-(self.alpha + 16)/12) #this is mag of tracer,
        # is just so when adjusting where forcing is amplitude is about ok without more fiddling with forceamp, result is from Bluemen PhD
        # -1 as this energy is distributed between more wavevectors when k is large and will go like circumference of circle
        sz = wvx.shape
        force = self.forceamp*wvxs
        force[wvx>self.fk2] = 0.
        force[wvx<self.fk1] = 0. 
        # I want to start with random directions for the forcing and then step these through with small brownian amounts
        self.forcerand = np.array(np.exp(2*np.pi*1j*self.rng.random((sz[0],sz[1]))))
        #self.force = self.forcerand * force
     
        
    def set_U(self, U):
        """Set background zonal flow"""
        self.Ubg = np.asarray(U)[np.newaxis,...]

    def _calc_diagnostics(self):
        # here is where we calculate diagnostics
        if (self.t>=self.dt) and (self.tc%self.taveints==0):
            self._increment_diagnostics()

    ### All the diagnostic stuff follows. ###
    def _calc_cfl(self):
        return np.abs(
            np.hstack([self.u + self.Ubg, self.v])
        ).max()*self.dt/self.dx

    # calculate KE: this has units of m^2 s^{-2}
    def _calc_ke(self):
        ke = .5*self.spec_var(self.wv*self.ph)
        return ke.sum()

    # calculate eddy turn over time
    # (perhaps should change to fraction of year...)
    def _calc_eddy_time(self):
        """ estimate the eddy turn-over time in days """
        ens = .5*self.H * spec_var(self, self.wv2*self.ph)
        return 2.*pi*np.sqrt( self.H / ens ) / year
    
    #SALAH
    def _return_streamfh(self):
        """ since normal return won't work for some reason"""
        return self.ph.copy()