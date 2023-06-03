import numpy as np

from pace.parallax_solver.ellipsoid import Ellipsoid


class EllipsoidTransducer(Ellipsoid):

    def __init__(self, *args, **kwargs):
        super(EllipsoidTransducer, self).__init__(*args, **kwargs)

        src = kwargs['src'] if 'src' in kwargs else [[], [], []]
        sen = kwargs['sen'] if 'sen' in kwargs else [[], [], []]
        toa = kwargs['toa'] if 'toa' in kwargs else 1
        txn = kwargs['txn'] if 'txn' in kwargs else [0, 0, 1]
        rxn = kwargs['rxn'] if 'rxn' in kwargs else [0, 0, 1]
        
        self.plane_opt = kwargs['plane_opt'] if 'plane_opt' in kwargs else False

        assert np.isscalar(float(toa)), 'Time-of-Arrival is not a scalar'
        assert len(txn) == 3, 'Transmitter normal should be a 3-vector'
        assert len(rxn) == 3, 'Receiver normal should be a 3-vector'

        # positions
        self.source_pos = np.array([*src]) if len(src) == 3 else np.zeros(3)
        self.sensor_pos = np.array([*sen]) if len(sen) == 3 else np.zeros(3)
        self.center_pos = (self.sensor_pos + self.source_pos) / 2
        self.spacer_vec = (self.source_pos - self.sensor_pos)
        self.normal_txn = np.array(txn)
        self.normal_rxn = np.array(rxn)

        # sensor distance
        self.spacer_vec_norm = (self.spacer_vec**2).sum()**.5

        # radii
        self.eig_vec = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        self.eig_val = np.ones(3)
        self.update_radii(toa)

    def update_radii(self, toa):
        """ constract ellipsoid vector axes based on incoming Time-of-Arrival distance """

        assert self.spacer_vec_norm < abs(toa), 'Time-of-Arrival distance shorter than transducer spacing'

        # eigenvalues
        minor_axis = np.sqrt(toa**2 - self.spacer_vec_norm**2) / 2
        major_axis = toa / 2
        self.eig_val = np.array([major_axis, minor_axis, minor_axis]) if not self.plane_opt else np.array([major_axis, np.spacing(1), minor_axis])

        if minor_axis <= 0 or major_axis <= 0:
            import warnings
            warnings.warn('Echo distance shorter (or equal) than transducer distance for ellipsoid instance %s' % self)

        # eigenvectors
        self.eig_vec[0] = self.spacer_vec / self.spacer_vec_norm if self.spacer_vec_norm != 0 else np.array([1., 0., 0.])
        self.eig_vec[1] = np.dot(self.rot_mat(0, 0, np.pi/2), self.eig_vec[0])
        self.eig_vec[2] = np.dot(self.rot_mat(0, np.pi/2, 0), self.eig_vec[0])

    def rotate_tx(self, phi: float = 0, theta: float = 0):
        """ rotate transmitter about two axes as third is thought to be rotationally symmetric """

        self.normal_txn = self.rot_mat(x=phi/180*np.pi, y=theta/180*np.pi, z=0) @ self.normal_txn

    def rotate_rx(self, phi: float = 0, theta: float = 0):
        """ rotate receiver about two axes as third is thought to be rotationally symmetric """

        self.normal_rxn = self.rot_mat(x=phi/180*np.pi, y=theta/180*np.pi, z=0) @ self.normal_rxn

    def reset_normals(self):

        self.normal_txn = np.array([0, 0, 1])
        self.normal_rxn = np.array([0, 0, 1])

    def get_valid_coords_mask(self):

        tx_prod = np.einsum('kij,k->ij', (self.coords-self.source_pos[:, None, None]), self.normal_txn)
        rx_prod = np.einsum('kij,k->ij', (self.coords-self.sensor_pos[:, None, None]), self.normal_rxn)
        
        mask = (tx_prod > 0) & (rx_prod > 0)
        
        return mask

    def Q(self, p, penalty=0):
        
        Q = Ellipsoid.Q(self, p)

        tx_prod = np.dot((p-self.source_pos), self.normal_txn)
        rx_prod = np.dot((p-self.sensor_pos), self.normal_rxn)

        if tx_prod < 0 or rx_prod < 0:
            Q += np.ones_like(Q)*penalty

        return Q

    @property
    def valid_points(self):
        
        mask = self.get_valid_coords_mask()
        
        return self.coords[:, mask]
