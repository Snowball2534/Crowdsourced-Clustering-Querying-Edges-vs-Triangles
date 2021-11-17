import numpy as np
from math import ceil
from math import floor
from scipy.linalg import block_diag
from scipy.special import comb

class data_generation(object):
    '''
    This class generates the synthetic data.
    n: the total number of nodes
    K: the number of cluters
    m: the number of nodes in each cluster
    r: the probability of observing an edge
    p: the probability of two nodes' connecting if two nodes belong to the same cluster
    q: the probability of two nodes' connecting if two nodes do not belong to the same cluster
    mode: 1, return one-coin mode
          2, return sbm and tbm
          3, return sbm and cbm
    '''
    def __init__(self, n, K, r, p, q, mode):
        self.m = int(floor(n / K))
        self.K = K
        self.n = self.m * self.K
        self.r = r
        self.p = p
        self.q = q
        self.xi = 1 - self.p

        template_1 = np.ones((self.m, self.m))
        template_2 = np.ones((1, self.m))
        pre_1 = np.ones((self.m, self.m))
        pre_2 = np.zeros((1, self.m))
        for i in range(self.K-1):
            pre_1 = block_diag(pre_1, template_1)
            pre_2 = np.hstack((pre_2, (i+1)*template_2))
        self.ground_truth = pre_1
        self.label = np.squeeze(pre_2)

        (self.H_edge_one_coin, self.H_edge_sbm) = self.entropy_edge()
        (self.H_triangle_one_coin, self.H_triangle_tbm, self.H_triangle_cbm) = self.entropy_triangle()
        if mode == 1:   ## generate data from the one-coin model
            (self.adj1, self.adj2, self.adj3) = self.generation_1()
        elif mode == 2:   ## generate data from SBM for the edge querying and TBM for the triangle querying
            (self.adj1, self.adj2, self.adj3) = self.generation_2()
        elif mode == 3:   ## generate data from SBM for the edge querying and CBM for the triangle querying
            (self.adj1, self.adj2, self.adj3) = self.generation_3()

    def get_parameters(self):
        return self.adj1, self.adj2, self.adj3, self.label
    def entropy_edge(self):
        '''
        This function computes the entropy of edge querying
        '''
        xi = self.xi
        p = self.p
        q = self.q

        px_1 = self.K*comb(self.m,2)/comb(self.n,2)
        px_0 = 1-px_1

        # one-coin edge model
        py_1 = (1-xi)*px_1 + xi*px_0
        H_one_coin = -py_1*np.log2(py_1)-(1-py_1)*np.log2(1-py_1)
        # stochastic block model
        py_1 = p*px_1 + q*px_0
        H_sbm = -py_1*np.log2(py_1)-(1-py_1)*np.log2(1-py_1)

        return H_one_coin, H_sbm

    def entropy_triangle(self):
        '''
        This function computes the entropy of triangle querying
        '''
        xi = self.xi
        p = self.p
        q = self.q

        px_1 = self.K*comb(self.m, 3)/comb(self.n, 3)
        px_5 = comb(self.K, 3)*self.m**3/comb(self.n, 3)
        px_234 = 1-px_1-px_5
        px_2 = px_3 = px_4 = (1-px_1-px_5)/3

        # one-coin edge model
        py_1 = (1-xi)*px_1 + (xi/4)*px_234 + (xi/4)*px_5
        py_5 = (1-xi)*px_5 + (xi/4)*px_1 + (xi/4)*px_234
        py_2 = py_3 = py_4 = (1 - py_1 - py_5) / 3
        H_one_coin = -py_1*np.log2(py_1)-py_5*np.log2(py_5)-3*py_2*np.log2(py_2)

        # triangle block model
        py_1 = (p**3 + 3*(1-p)*p**2)*px_1 + (q**3)*px_5 + (p*q**2)*px_234
        py_5 = ((1-p)**3)*px_1 + ((1-q)**3+3*(1-q)*q**2)*px_5+((1-p)*(1-q)**2)*px_234
        py_2 = py_3 = py_4 = (1 - py_1 - py_5) / 3
        H_tbm = -py_1*np.log2(py_1)-py_5*np.log2(py_5)-3*py_2*np.log2(py_2)

        # conditional block model
        z_lll = 3*p**3 - 3*p**2 + 1
        z_llm = 3*p*q**2 - 2*p*q - q**2 + 1
        z_lmj = 3*q**3 - 3*q**2 + 1

        py_1 = (p**3/z_lll)*px_1 + (q**3/z_lmj)*px_5 + (p*q**2/z_llm)*px_234
        py_5 = ((1-p)**3/z_lll)*px_1 + ((1-q)**3/z_lmj)*px_5 + (((1-p)*(1-q)**2)/z_llm)*px_234
        py_2 = py_3 = py_4 = (1 - py_1 - py_5) / 3
        H_cbm = -py_1*np.log2(py_1)-py_5*np.log2(py_5)-3*py_2*np.log2(py_2)

        return H_one_coin, H_tbm, H_cbm

    # generate data from the one-coin model
    def generation_1(self):
        E = ceil(self.r*comb(self.n, 2))
        Tb = ceil(E*(self.H_edge_one_coin/self.H_triangle_one_coin))
        Te = ceil(E/3)

        # generate data for edge querying
        adjacency_matrix_E = np.zeros((self.n, self.n))
        for i in range(E):
            x = np.random.randint(0,self.n)
            y = np.random.randint(0,self.n)
            if self.ground_truth[x,y]==1:
                if np.random.uniform() > self.xi:
                    adjacency_matrix_E[x,y]=adjacency_matrix_E[y,x]=1
            else:
                if np.random.uniform() < self.xi:
                    adjacency_matrix_E[x,y]=adjacency_matrix_E[y,x]=1

        return adjacency_matrix_E, self.help_1(self.xi, Tb), self.help_1(self.xi, Te)

    # generate data for triangle querying of one coin edge model
    def help_1(self, xi, counts):
        adjacency_matrix = np.zeros((self.n, self.n))
        for i in range(counts):
            x = np.random.randint(0, self.n)
            y = np.random.randint(0, self.n)
            z = np.random.randint(0, self.n)
            random_number = np.random.uniform()
            if self.ground_truth[x, y] == 1 and self.ground_truth[y, z] == 1 and self.ground_truth[x, z] == 1:
                if random_number < 1 - xi:
                    adjacency_matrix[x, y] = adjacency_matrix[y, z] = adjacency_matrix[x, z] = 1
                    adjacency_matrix[y, x] = adjacency_matrix[z, y] = adjacency_matrix[z, x] = 1
                elif random_number < 1 - 3 * xi / 4:
                    adjacency_matrix[x, y] = adjacency_matrix[y, x] = 1
                elif random_number < 1 - xi / 2:
                    adjacency_matrix[x, z] = adjacency_matrix[z, x] = 1
                elif random_number < 1 - xi / 4:
                    adjacency_matrix[y, z] = adjacency_matrix[z, y] = 1
            elif self.ground_truth[x, y] == 1 and self.ground_truth[y, z] == 0 and self.ground_truth[x, z] == 0:
                if random_number < 1 - xi:
                    adjacency_matrix[x, y] = adjacency_matrix[y, x] = 1
                elif random_number < 1 - 3 * xi / 4:
                    adjacency_matrix[x, y] = adjacency_matrix[y, z] = adjacency_matrix[x, z] = 1
                    adjacency_matrix[y, x] = adjacency_matrix[z, y] = adjacency_matrix[z, x] = 1
                elif random_number < 1 - xi / 2:
                    adjacency_matrix[x, z] = adjacency_matrix[z, x] = 1
                elif random_number < 1 - xi / 4:
                    adjacency_matrix[y, z] = adjacency_matrix[z, y] = 1
            elif self.ground_truth[x, y] == 0 and self.ground_truth[y, z] == 0 and self.ground_truth[x, z] == 1:
                if random_number < 1 - xi:
                    adjacency_matrix[x, z] = adjacency_matrix[z, x] = 1
                elif random_number < 1 - 3 * xi / 4:
                    adjacency_matrix[x, y] = adjacency_matrix[y, z] = adjacency_matrix[x, z] = 1
                    adjacency_matrix[y, x] = adjacency_matrix[z, y] = adjacency_matrix[z, x] = 1
                elif random_number < 1 - xi / 2:
                    adjacency_matrix[x, y] = adjacency_matrix[y, x] = 1
                elif random_number < 1 - xi / 4:
                    adjacency_matrix[y, z] = adjacency_matrix[z, y] = 1
            elif self.ground_truth[x, y] == 0 and self.ground_truth[y, z] == 1 and self.ground_truth[x, z] == 0:
                if random_number < 1 - xi:
                    adjacency_matrix[y, z] = adjacency_matrix[z, y] = 1
                elif random_number < 1 - 3 * xi / 4:
                    adjacency_matrix[x, y] = adjacency_matrix[y, z] = adjacency_matrix[x, z] = 1
                    adjacency_matrix[y, x] = adjacency_matrix[z, y] = adjacency_matrix[z, x] = 1
                elif random_number < 1 - xi / 2:
                    adjacency_matrix[x, y] = adjacency_matrix[y, x] = 1
                elif random_number < 1 - xi / 4:
                    adjacency_matrix[x, z] = adjacency_matrix[z, x] = 1
            elif self.ground_truth[x, y] == 0 and self.ground_truth[y, z] == 0 and self.ground_truth[x, z] == 0:
                if random_number < xi / 4:
                    adjacency_matrix[x, y] = adjacency_matrix[y, z] = adjacency_matrix[x, z] = 1
                    adjacency_matrix[y, x] = adjacency_matrix[z, y] = adjacency_matrix[z, x] = 1
                elif random_number < xi / 2:
                    adjacency_matrix[x, y] = adjacency_matrix[y, x] = 1
                elif random_number < 3 * xi / 4:
                    adjacency_matrix[x, z] = adjacency_matrix[z, x] = 1
                elif random_number < xi:
                    adjacency_matrix[y, z] = adjacency_matrix[z, y] = 1
        return adjacency_matrix

    # generate data from sbm and tbm
    def generation_2(self):
        E = ceil(self.r*comb(self.n, 2))
        Tb = ceil(E*(self.H_edge_sbm/self.H_triangle_tbm))
        Te = ceil(E/3)

        return self.help_2(self.p, self.q, E), self.help_3(self.p, self.q, Tb), self.help_3(self.p, self.q, Te)

    # generate data from sbm and cbm
    def generation_3(self):
        E = ceil(self.r * comb(self.n, 2))
        Tb = ceil(E * (self.H_edge_sbm / self.H_triangle_cbm))
        Te = ceil(E / 3)

        return self.help_2(self.p, self.q, E), self.help_4(self.p, self.q, Tb), self.help_4(self.p, self.q, Te)

    # generate data for edge querying
    def help_2(self, p, q, counts):
        adjacency_matrix = np.zeros((self.n, self.n))
        for i in range(counts):
            x = np.random.randint(0, self.n)
            y = np.random.randint(0, self.n)
            if self.ground_truth[x, y] == 1:
                if np.random.uniform() < p:
                    adjacency_matrix[x, y] = adjacency_matrix[y, x] = 1
            else:
                if np.random.uniform() < q:
                    adjacency_matrix[x, y] = adjacency_matrix[y, x] = 1
        return adjacency_matrix

    # generate data for triangle querying of triangle block model
    def help_3(self, p, q, counts):
        adjacency_matrix = np.zeros((self.n, self.n))
        for i in range(counts):
            x = np.random.randint(0, self.n)
            y = np.random.randint(0, self.n)
            z = np.random.randint(0, self.n)
            random_number = np.random.uniform()
            if self.ground_truth[x, y] == 1 and self.ground_truth[y, z] == 1 and self.ground_truth[x, z] == 1:
                if random_number < p ** 3 + 3 * (1 - p) * p ** 2:
                    adjacency_matrix[x, y] = adjacency_matrix[y, z] = adjacency_matrix[x, z] = 1
                    adjacency_matrix[y, x] = adjacency_matrix[z, y] = adjacency_matrix[z, x] = 1
                elif random_number < p ** 3 + 3 * (1 - p) * p ** 2 + p * (1 - p) ** 2:
                    adjacency_matrix[x, y] = adjacency_matrix[y, x] = 1
                elif random_number < p ** 3 + 3 * (1 - p) * p ** 2 + 2 * p * (1 - p) ** 2:
                    adjacency_matrix[x, z] = adjacency_matrix[z, x] = 1
                elif random_number < p ** 3 + 3 * (1 - p) * p ** 2 + 3 * p * (1 - p) ** 2:
                    adjacency_matrix[y, z] = adjacency_matrix[z, y] = 1
            elif self.ground_truth[x, y] == 1 and self.ground_truth[y, z] == 0 and self.ground_truth[x, z] == 0:
                if random_number < p * q ** 2:
                    adjacency_matrix[x, y] = adjacency_matrix[y, z] = adjacency_matrix[x, z] = 1
                    adjacency_matrix[y, x] = adjacency_matrix[z, y] = adjacency_matrix[z, x] = 1
                elif random_number < p * q ** 2 + p * (1 - q) ** 2 + (1 - p) * q ** 2 + 2 * p * q * (1 - q):
                    adjacency_matrix[x, y] = adjacency_matrix[y, x] = 1
                elif random_number < p * q ** 2 + p * (1 - q) ** 2 + (1 - p) * q ** 2 + 2 * p * q * (1 - q) + (
                        1 - p) * q * (1 - q):
                    adjacency_matrix[x, z] = adjacency_matrix[z, x] = 1
                elif random_number < p * q ** 2 + p * (1 - q) ** 2 + (1 - p) * q ** 2 + 2 * p * q * (1 - q) + 2 * (
                        1 - p) * q * (1 - q):
                    adjacency_matrix[y, z] = adjacency_matrix[z, y] = 1
            elif self.ground_truth[x, y] == 0 and self.ground_truth[y, z] == 0 and self.ground_truth[x, z] == 1:
                if random_number < p * q ** 2:
                    adjacency_matrix[x, y] = adjacency_matrix[y, z] = adjacency_matrix[x, z] = 1
                    adjacency_matrix[y, x] = adjacency_matrix[z, y] = adjacency_matrix[z, x] = 1
                elif random_number < p * q ** 2 + p * (1 - q) ** 2 + (1 - p) * q ** 2 + 2 * p * q * (1 - q):
                    adjacency_matrix[x, z] = adjacency_matrix[z, x] = 1
                elif random_number < p * q ** 2 + p * (1 - q) ** 2 + (1 - p) * q ** 2 + 2 * p * q * (1 - q) + (
                        1 - p) * q * (1 - q):
                    adjacency_matrix[x, y] = adjacency_matrix[y, x] = 1
                elif random_number < p * q ** 2 + p * (1 - q) ** 2 + (1 - p) * q ** 2 + 2 * p * q * (1 - q) + 2 * (
                        1 - p) * q * (1 - q):
                    adjacency_matrix[y, z] = adjacency_matrix[z, y] = 1
            elif self.ground_truth[x, y] == 0 and self.ground_truth[y, z] == 1 and self.ground_truth[x, z] == 0:
                if random_number < p * q ** 2:
                    adjacency_matrix[x, y] = adjacency_matrix[y, z] = adjacency_matrix[x, z] = 1
                    adjacency_matrix[y, x] = adjacency_matrix[z, y] = adjacency_matrix[z, x] = 1
                elif random_number < p * q ** 2 + p * (1 - q) ** 2 + (1 - p) * q ** 2 + 2 * p * q * (1 - q):
                    adjacency_matrix[y, z] = adjacency_matrix[z, y] = 1
                elif random_number < p * q ** 2 + p * (1 - q) ** 2 + (1 - p) * q ** 2 + 2 * p * q * (1 - q) + (
                        1 - p) * q * (1 - q):
                    adjacency_matrix[x, y] = adjacency_matrix[y, x] = 1
                elif random_number < p * q ** 2 + p * (1 - q) ** 2 + (1 - p) * q ** 2 + 2 * p * q * (1 - q) + 2 * (
                        1 - p) * q * (1 - q):
                    adjacency_matrix[x, z] = adjacency_matrix[z, x] = 1
            elif self.ground_truth[x, y] == 0 and self.ground_truth[y, z] == 0 and self.ground_truth[x, z] == 0:
                if random_number < q ** 3:
                    adjacency_matrix[x, y] = adjacency_matrix[y, z] = adjacency_matrix[x, z] = 1
                    adjacency_matrix[y, x] = adjacency_matrix[z, y] = adjacency_matrix[z, x] = 1
                elif random_number < q ** 3 + q * (1 - q) ** 2:
                    adjacency_matrix[x, y] = adjacency_matrix[y, x] = 1
                elif random_number < q ** 3 + 2 * q * (1 - q) ** 2:
                    adjacency_matrix[x, z] = adjacency_matrix[z, x] = 1
                elif random_number < q ** 3 + 3 * q * (1 - q) ** 2:
                    adjacency_matrix[y, z] = adjacency_matrix[z, y] = 1
        return adjacency_matrix

    # generate data for triangle querying of conditional block model
    def help_4(self, p, q, counts):
        z_lll = 3*p**3 - 3*p**2 + 1
        z_llm = 3*p*q**2 - 2*p*q - q**2 + 1
        z_lmj = 3*q**3 - 3*q**2 + 1
        adjacency_matrix = np.zeros((self.n, self.n))
        for i in range(counts):
            x = np.random.randint(0, self.n)
            y = np.random.randint(0, self.n)
            z = np.random.randint(0, self.n)
            random_number = np.random.uniform()
            if self.ground_truth[x, y] == 1 and self.ground_truth[y, z] == 1 and self.ground_truth[x, z] == 1:
                if random_number < p ** 3/z_lll:
                    adjacency_matrix[x, y] = adjacency_matrix[y, z] = adjacency_matrix[x, z] = 1
                    adjacency_matrix[y, x] = adjacency_matrix[z, y] = adjacency_matrix[z, x] = 1
                elif random_number < (p ** 3 + p * (1 - p) ** 2)/z_lll:
                    adjacency_matrix[x, y] = adjacency_matrix[y, x] = 1
                elif random_number < (p ** 3 + 2 * p * (1 - p) ** 2)/z_lll:
                    adjacency_matrix[x, z] = adjacency_matrix[z, x] = 1
                elif random_number < (p ** 3 + 3 * p * (1 - p) ** 2)/z_lll:
                    adjacency_matrix[y, z] = adjacency_matrix[z, y] = 1
            elif self.ground_truth[x, y] == 1 and self.ground_truth[y, z] == 0 and self.ground_truth[x, z] == 0:
                if random_number < p * q ** 2/z_llm:
                    adjacency_matrix[x, y] = adjacency_matrix[y, z] = adjacency_matrix[x, z] = 1
                    adjacency_matrix[y, x] = adjacency_matrix[z, y] = adjacency_matrix[z, x] = 1
                elif random_number < (p * q ** 2 + p * (1 - q) ** 2)/z_llm:
                    adjacency_matrix[x, y] = adjacency_matrix[y, x] = 1
                elif random_number < (p * q ** 2 + p * (1 - q) ** 2 + (1 - p) * q * (1 - q))/z_llm:
                    adjacency_matrix[x, z] = adjacency_matrix[z, x] = 1
                elif random_number < (p * q ** 2 + p * (1 - q) ** 2 + 2*(1 - p) * q * (1 - q))/z_llm:
                    adjacency_matrix[y, z] = adjacency_matrix[z, y] = 1
            elif self.ground_truth[x, y] == 0 and self.ground_truth[y, z] == 0 and self.ground_truth[x, z] == 1:
                if random_number < p * q ** 2 / z_llm:
                    adjacency_matrix[x, y] = adjacency_matrix[y, z] = adjacency_matrix[x, z] = 1
                    adjacency_matrix[y, x] = adjacency_matrix[z, y] = adjacency_matrix[z, x] = 1
                elif random_number < (p * q ** 2 + p * (1 - q) ** 2) / z_llm:
                    adjacency_matrix[x, z] = adjacency_matrix[z, x] = 1
                elif random_number < (p * q ** 2 + p * (1 - q) ** 2 + (1 - p) * q * (1 - q)) / z_llm:
                    adjacency_matrix[x, y] = adjacency_matrix[y, x] = 1
                elif random_number < (p * q ** 2 + p * (1 - q) ** 2 + 2 * (1 - p) * q * (1 - q)) / z_llm:
                    adjacency_matrix[y, z] = adjacency_matrix[z, y] = 1
            elif self.ground_truth[x, y] == 0 and self.ground_truth[y, z] == 1 and self.ground_truth[x, z] == 0:
                if random_number < p * q ** 2 / z_llm:
                    adjacency_matrix[x, y] = adjacency_matrix[y, z] = adjacency_matrix[x, z] = 1
                    adjacency_matrix[y, x] = adjacency_matrix[z, y] = adjacency_matrix[z, x] = 1
                elif random_number < (p * q ** 2 + p * (1 - q) ** 2) / z_llm:
                    adjacency_matrix[y, z] = adjacency_matrix[z, y] = 1
                elif random_number < (p * q ** 2 + p * (1 - q) ** 2 + (1 - p) * q * (1 - q)) / z_llm:
                    adjacency_matrix[x, y] = adjacency_matrix[y, x] = 1
                elif random_number < (p * q ** 2 + p * (1 - q) ** 2 + 2 * (1 - p) * q * (1 - q)) / z_llm:
                    adjacency_matrix[x, z] = adjacency_matrix[z, x] = 1
            elif self.ground_truth[x, y] == 0 and self.ground_truth[y, z] == 0 and self.ground_truth[x, z] == 0:
                if random_number < q ** 3/z_lmj:
                    adjacency_matrix[x, y] = adjacency_matrix[y, z] = adjacency_matrix[x, z] = 1
                    adjacency_matrix[y, x] = adjacency_matrix[z, y] = adjacency_matrix[z, x] = 1
                elif random_number < (q ** 3 + q * (1 - q) ** 2)/z_lmj:
                    adjacency_matrix[x, y] = adjacency_matrix[y, x] = 1
                elif random_number < (q ** 3 + 2 * q * (1 - q) ** 2)/z_lmj:
                    adjacency_matrix[x, z] = adjacency_matrix[z, x] = 1
                elif random_number < (q ** 3 + 3 * q * (1 - q) ** 2)/z_lmj:
                    adjacency_matrix[y, z] = adjacency_matrix[z, y] = 1
        return adjacency_matrix

