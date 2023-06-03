import numpy as np
from scipy import optimize

from pace.parallax_solver.gradient_solver import ParallaxGradientSolver
from pace.parallax_solver.gradient_solver import lsq_cost, lsq_jaco


class ParallaxHessianSolver(ParallaxGradientSolver):

    def __init__(self, *args, **kwargs):
        super(ParallaxHessianSolver, self).__init__(*args, **kwargs)

    def sci_lma(self, p=None, elps=None):
        """ Scipy's Levenberg-Marquardt  """

        self.p = self.p if p is None else p
        self.elps = self.elps if elps is None else elps

        lma = optimize.root(lsq_cost, x0=self.p, args=(self.elps), method='lm', jac=lsq_jaco)
        self.p = lma.x
        
        valid = self.check_validity()

        return [self.p], valid, None

    def lsq_lma(self, p=None, elps=None, tol=1e-15, tau=1e-3, meth='lev', rho1=.25, rho2=.75, bet=2, gam=3, max_iter=99,
                surf_opt=False):
        """
        Levenberg-Marquardt implementation for least-squares fitting of non-linear functions
        :param p: initial value(s)
        :param elps: list of ellipsoid objects
        :param tol: tolerance for stop condition
        :param tau: factor to initialize damping parameter
        :param meth: method which is default 'lev' for Levenberg and otherwise Marquardt
        :param rho1: first gain factor threshold for damping parameter adjustment for Marquardt
        :param rho2: second gain factor threshold for damping parameter adjustment for Marquardt
        :param bet: multiplier for damping parameter adjustment for Marquardt
        :param gam: divisor for damping parameter adjustment for Marquardt
        :param max_iter: maximum number of iterations
        :return: list of results, eps
        """

        self.p = self.p if p is None else p
        self.elps = self.elps if elps is None else elps

        j = lsq_jaco(self.p, self.elps)
        g = np.dot(j.T, lsq_cost(self.p, self.elps))
        H = np.dot(j.T, j)
        u = tau * np.max(np.diag(H.diagonal()))
        v = 2
        eps = 1
        p_list = [self.p]
        while len(p_list) < max_iter:
            D = np.identity(j.shape[1])
            D *= 1 if meth == 'lev' else np.max(np.maximum(H.diagonal(), D.diagonal()))
            h = -np.dot(np.linalg.inv(H + u * D), g)
            f = lsq_cost(self.p, self.elps)
            f_plus = lsq_cost(self.p + h, self.elps)
            rho = (np.dot(f.T, f) - np.dot(f_plus.T, f_plus)) / np.dot(.5 * h.T, u * h - g)
            if rho > 0:
                self.p += h
                self.p = self.surf_project(self.p) if surf_opt else self.p
                p_list.append(self.p.copy())
                j = lsq_jaco(self.p, self.elps)
                g = np.dot(j.T, lsq_cost(self.p, self.elps))
                H = np.dot(j.T, j)
            if meth == 'lev':
                u, v = (u * np.max([1 / 3, 1 - (2 * rho - 1) ** 3]), 2) if rho > 0 else (u * v, v * 2)
            else:
                u = u * bet if rho < rho1 else u / gam if rho > rho2 else u
            eps = np.sum(np.abs(f))#max(abs(g))
            if eps < tol:
                break

        valid = self.check_validity()

        return p_list, valid, eps
