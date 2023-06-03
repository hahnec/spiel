import numpy as np


class Ellipsoid(object):

    def __init__(self, 
                cen_pos:np.ndarray=None, 
                eig_val:np.ndarray=None, 
                eig_vec:np.ndarray=None, 
                n_coords:int=None, 
                *args, 
                **kwargs,
                ):

        self.center_pos = np.zeros(3) if cen_pos is None else cen_pos
        self.eig_val = np.ones(3) if eig_val is None else eig_val
        self.eig_vec = np.zeros((3, 3)) if eig_vec is None else eig_vec

        # initialize radii vectors
        self.axs_vec = np.zeros((3, 3))

        # provide n as imaginary number to serve number of samples to mgrid
        self._n = np.complex(0, 63) if n_coords is None else np.complex(0, n_coords)

        # spherical coordinates
        self.phi, self.theta = np.mgrid[0:np.pi:self._n, 0:2*np.pi:self._n]

    def calc_radii_vectors(self):

        # compute 3D axes vectors of ellipsoid
        for i in range(3):
            axs_vec = np.zeros(3)
            axs_vec[i] = 1 * self.eig_val[i]
            self.axs_vec[i] = np.dot(self.rot_mat, axs_vec)

    def rot_mat(self, x:float=0, y:float=0, z:float=0):
        """ create rotation matrix from angles [rad] """

        rot_x = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
        rot_y = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
        rot_z = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])

        return np.dot(rot_z, np.dot(rot_y, rot_x))

    @property
    def eig_rot(self):
        """
        https://math.stackexchange.com/questions/1246679/expression-of-rotation-matrix-from-two-vectors
        """
        
        vec = self.eig_vec[0] / np.linalg.norm(self.eig_vec[0])
        ref = np.array([1, 0, 0]) / np.linalg.norm(np.array([1, 0, 0]))
        crs = np.cross(vec, ref) if np.count_nonzero(abs(vec)-abs(ref)) != 0 else np.array([0, 0, 1])
        crs = crs / np.linalg.norm(crs) if np.linalg.norm(crs) != 0 else crs

        mat = np.array([vec, np.cross(crs, vec), crs]).T

        return mat 

    @staticmethod
    def angle_between_vecs(v1, v2):

        v1u = v1 / np.linalg.norm(v1)
        v2u = v2 / np.linalg.norm(v2)
        ret = np.arccos(np.clip(np.dot(v1u, v2u), -1.0, 1.0))

        #ret = np.arccos((v1.T@v2) / np.sqrt((v1.T@v1)*(v2.T@v2)))

        return ret

    @property
    def coords_origin(self):

        x = self.eig_val[0] * np.sin(self.phi) * np.cos(self.theta)
        y = self.eig_val[1] * np.sin(self.phi) * np.sin(self.theta)
        z = self.eig_val[2] * np.cos(self.phi)

        return np.array([x, y, z])

    @property
    def coords(self):

        # spherical coordinate parametrization
        x, y, z = self.coords_origin

        # rotate ellipsoidal coordinates around origin
        x, y, z = np.dot(self.eig_rot, np.array([x.flatten(), y.flatten(), z.flatten()]))

        # translate coordinates to relative position in space
        x, y, z = x + self.center_pos[0], y + self.center_pos[1], z + self.center_pos[2]

        return np.array([x.reshape(self.phi.shape), y.reshape(self.phi.shape), z.reshape(self.phi.shape)])

    def Q(self, p):
        """ surface equation for displaced and rotated ellipsoidal points """
        
        p_i = self.eig_rot.T @ (p - self.center_pos.T).T

        Q = (p_i[0] / self.eig_val[0])**2 + (p_i[1] / self.eig_val[1])**2 + (p_i[2] / self.eig_val[2])**2 - 1
        
        return Q
    
    def pd1Q(self, p, var_idx=0):
        """ partial derivative of surface equation with respect to variable index """
        
        r_inv = self.eig_rot.T

        p_i = r_inv @ (p - self.center_pos.T).T

        pd1Q = 2*(r_inv[0][var_idx]/self.eig_val[0]*(p_i[0] / self.eig_val[0]) + \
                  r_inv[1][var_idx]/self.eig_val[1]*(p_i[1] / self.eig_val[1]) + \
                  r_inv[2][var_idx]/self.eig_val[2]*(p_i[2] / self.eig_val[2])
                 )
        
        return pd1Q
