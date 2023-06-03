import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import root

# ellipsoidal surface functions and its partial derivatives
function = lambda p, elps: np.array([elp.Q(p) for elp in elps])
jacobian = lambda p, elps: np.array([[elp.pd1Q(p, 0), elp.pd1Q(p, 1), elp.pd1Q(p, 2)] for elp in elps])

# least-squares cost functions
lsq_cost = lambda p, elps: np.square(function(p, elps))
lsq_jaco = lambda p, elps: 2*function(p, elps)[np.newaxis, :].T*jacobian(p, elps)


class ParallaxGradientSolver(object):

    def __init__(self, *args, **kwargs):

        self.elps = args[0] if len(args) > 0 else []
        self.elps = kwargs['elps'] if 'elps' in kwargs else self.elps
        self.p = self.closest_sample_distance()[0] if len(self.elps) > 0 else None
        self.p = kwargs['p'] if 'p' in kwargs else self.p
        self.val_thresh = kwargs['val_thresh'] if 'val_thresh' in kwargs else 1e-4

    def closest_sample_distance(self, elp_idx=0, elps: list=None):

        self.elps = self.elps if elps is None else elps

        assert elp_idx < len(self.elps), "Index larger than list length"

        loc_elps = self.elps.copy()
        loc_elps.pop(elp_idx)
        other_coords = np.array([elp.coords.T for elp in loc_elps]).reshape(len(loc_elps), -1, 3)
        candidates = self.elps[elp_idx].valid_points.T

        edist = cdist(np.vstack(other_coords), candidates)
        emins = np.min(edist, 0)
        self.p = candidates[emins.argmin()]

        return self.p, emins.min()

    def multivar_gradient_scipy(self, p=None, elps: list=None, max_iter=50):
        
        self.elps = self.elps if elps is None else elps
        self.p = self.closest_sample_distance()[0] if p is None else p

        self.p = root(function, self.p, jac=jacobian, args=(self.elps,), options={"maxfev":max_iter}).x

        eps = [elp.Q(self.p) for elp in self.elps]
        valid = self.check_validity()

        return [self.p], valid, eps

    def multivar_gradient(self, p=None, elps: list=None, l=0.1, tol=1e-15, eps=1, max_iter=50, surf_opt=False):
        
        self.elps = self.elps if elps is None else elps
        self.p = self.closest_sample_distance()[0] if p is None else p

        p_list = []
        while len(p_list) < max_iter and tol<eps<200:
            p_list.append(self.p.copy())
            f = function(self.p, self.elps)
            j = jacobian(self.p, self.elps)
            self.p -= l*np.dot(np.linalg.pinv(j), f)
            self.p = self.surf_project(self.p) if surf_opt else self.p
            eps = np.sum(np.abs(f))

        eps = [elp.Q(self.p) for elp in self.elps]
        valid = self.check_validity()

        return p_list, valid, eps

    def surf_project(self, p, elp_idx_meth='surface'):

        # identify nearest ellipsoid
        if elp_idx_meth == 'center':
            idx = np.argmin([np.linalg.norm(p-elp.center_pos) for elp in self.elps])
        elif elp_idx_meth == 'surface':
            idx = np.argmin([elp.Q(p) for elp in self.elps])
        else:
            idx = 0

        # translate to ellipsoid origin
        p -= self.elps[idx].center_pos

        # shrink radius to unit length
        #p /= np.linalg.norm(p)

        # get factor for vector scaling
        a = p[0]**2/self.elps[-1].eig_val[0]**2 + p[1]**2/self.elps[-1].eig_val[1]**2 + p[2]**2/self.elps[-1].eig_val[2]**2

        # scale vector to surface
        p *= (1/a)**.5

        # translate back to space
        p += self.elps[idx].center_pos

        return p

    def check_validity(self, threshold=None):
        self.val_thresh = self.val_thresh if threshold is None else threshold
        return np.linalg.norm([elp.Q(self.p) for elp in self.elps]) < self.val_thresh
