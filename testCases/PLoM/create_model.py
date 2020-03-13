import numpy as np 
from scipy.spatial.distance import pdist, squareform
from scipy import linalg
import pybitup 

# Useless
def titan_createModel(Modelopts): 

    if (Modelopts.Type is not None):
        if Modelopts.Type == "PCE":   # PCE metamodel
                if Modelopts.Method == 'Quadrature': 
                    Model = titan_PCE_quadrature(Modelopts)
                    Model.Inputs = Modelopts.Inputs
                elif Modelopts.Method == 'OLS':
                    Model = titan_PCE_OLS(Modelopts)
                    Model.Inputs = Modelopts.Inputs
        elif Modelopts.Type == 'ME-PCE': # ME-PCE metamodel
            if Modelopts.Method == 'OLS': 
                    Model = titan_ME_PCE_OLS(Modelopts)
                    Model.Inputs = Modelopts.Inputs      
        elif Modelopts.Type == 'Kriging':   # Kriging metamodel
            Model = titan_kriging(Modelopts)
        elif Modelopts.MetaType == 'PLoM': # Probabilistic learning on manifolds
            #titan_PLoM(Modelopts)
            a = 1
        else: 
            raise ValueError('No model named "{}".'.format(Modelopts.Type)) 
    else:
        if (Modelopts.get("mFile") is not None):
            Model.mFile = Modelopts.mFile
        elif (Modelopts.get("mHandle") is not None):
            Model.mHandle = Modelopts.mHandle
        elif (Modelopts.get("mString") is not None):
            Model.mString = Modelopts.mString
        else:      
            raise ValueError('Error: No field mFile or mHandle or mString found \n')  
        
        if (Modelopts.get("Parameters") is not None):
            Model.Parameters = Modelopts.Parameters





class TitanModel():

    def __init__(self):

        self.Type = None 
        self.MetaType = None 
        self.ExpDesign = ExpDesign() 

        self.Opt = ModelOption() 

class TitanPLoM(TitanModel):

    def __init__(self):
        TitanModel.__init__(self)

    def titan_PLoM(self): 

        self.xd = np.transpose(np.concatenate((self.ExpDesign.X, self.ExpDesign.Y), axis=1)) # Initial dataset (n \times N)
        n = self.xd.shape[0] # n: dimension of the non-Gaussian random vector (X,Y)

        # (1) Scaling of dataset to avoid numerical problems
        #     Ref: Soize and Ghanem, Data-driven probability concentration and sampling on manifold, J. Comp. Phys, 2016 
        #     Ref: Soize et al., Entropy-based closure for probablistic learning on manifolds, J. Comp. Phys., 2019
        if self.Opt.scaling==1:
            epsilons = 1e-10 # small factor to avoid zero values
            for k in range(n): 
                self.xd[k,:] = (self.xd[k,:] - min(self.xd[k,:]))/(max(self.xd[k,:] ) - min(self.xd[k,:])) + epsilons    

        # (2) Normalization of the initial dataset using a principal component analysis (PCA)
        #     Ref: Soize and Ghanem, Data-driven probability concentration and sampling on manifold, J. Comp. Phys, 2016
        self.titan_PCA_normalization(self.xd)

        # (3) Construct the diffusion-map basis
        self.titan_diffusionmapbasis(self.etad)

        # (4) Reduced-order representation
        self.titan_reduced_order_representation(self.etad,self.g)

    def titan_PLoM_eval(self): 


        xbar =self.xbar
        phi = self.phi
        lambd = self.lambd
        etad = self.etad
        zd = self.zd
        g = self.g
        a = self.a

        xd = np.transpose(np.concatenate((self.ExpDesign.X, self.ExpDesign.Y), axis=1)) # Initial dataset (n \times N)
        n = self.xd.shape[0] # n: dimension of the non-Gaussian random vector (X,Y)

        # (1) Generation of additional realizations solving a reduced-order Ito
        # stochachtist differential equation (ISDE)

        nIterations = 1000 # nMC
        param_init = zd 
        f0 = 1.5
        C_matrix = "Identity"
        gradient = "Numerical"

        #pybitup.metropolis_hastings_algorithms.ito_SDE(IO_fileID, self.Type, nIterations, param_init, prob_distr, h, f0, C_matrix, gradient) 
        
        #etadd = self.titan_reduced_order_ISDE(a,g,etad,zd,Metamodel.Itoopt)

        # (2) Normalization of additional realizations

        # nMC = size(etadd,3);
        # N = size(etadd,2);
        # nu = size(etadd,1);
        # 
        # etaddint = zeros(nu,N*nMC);
        # 
        # for ll=1:nMC
        #     for ii=1:N
        #         etaddint(:,(ll-1)*N+ii) = etadd(:,ii,ll);
        #     end
        # end
        # 
        # meanetadd = mean(etaddint,2);
        # stdetadd = zeros(nu,nu);
        # for ii=1:nu
        #     for jj=ii:nu
        #         stdetadd(ii,jj) = N/(N-1)*mean((etaddint(ii,:)-meanetadd(ii)).*(etaddint(jj,:)-meanetadd(jj)));
        #         if(ii~=jj)
        #             stdetadd(jj,ii) = stdetadd(ii,jj);
        #         end    
        #     end
        # end    
        # R = chol(stdetadd);
        # R = R';
        # 
        # for ll = 1:nMC
        #       etadd(:,:,ll) = R\(etadd(:,:,ll)-meanetadd);
        # end

        # # for ll = 1:nMC
        # #     for k = 1:n
        # #         etadd(k,:,ll) = (etadd(k,:,ll)-meanetadd(k))/stdetadd(k);
        # #     end
        # # end

        # (3) Perform inverse transformation to the original variables (descaling)
        nMC = etadd.shape[2]
        for ll in range(nMC): 
            etadd[:,:,ll] = xbar + phi*np.diag(np.sqrt(lambd))*etadd[:,:,ll]
            if(self.Opt.scaling==1): 
                for k in range(n):
                    etadd[k,:,ll] = (np.max(xd[k,:])-np.min(xd[k,:]))*etadd[k,:,ll]+np.min(xd[k,:])

        Y = etadd

        return Y 



    def titan_PCA_normalization(self, X):             
        """ Copyright: K. Bulthuis 10 April 2019 (Original Matlab implementation)
                    J. Coheur 02 February 2020 (Python implementation)
        --- Ojective
                Perform a normalisation of the initial dataset with PCA
                [eta] = [lambda]^{-1/2} [phi]^T ([x]-[underbar{x}])
                with [eta] the normalized matrix (nu times N)
                        [x] the dataset (n times N)
                        [underbar{x}] the empirical mean of the columns of the
                        data set
                        [lambda] diagonal matrix composed of the nu positive
                        eigenvalues of the empirical estimate [Cov] of the
                        covariance matrix of [X](nu times nu)
                        [phi] matrix of associated eigenvectors (n times nu)

        ---INPUT
                X : dataset (n times N)

        ---- OUTPUT
                Eta : normalized dataset with PCA (nu times N). """

        espeig = 1e-14 # small factor to avoid zero values to avoid too small eigenvalues

        Xbar = np.mean(X, axis=1)                                 # empirical estimate of the mean vector 
        Sigma = np.cov(X)                    # empirical estimate of the covariance matrix of the mean vector 

        Lambda, Phi = np.linalg.eig(Sigma)                          # eigenvalues and eigenvectors of the empirical covariance matrix     
        
        ind = np.argsort(Lambda)                     # sort eigenvalues in ascending order 
        Lambda = Lambda[ind]

        Lambda = np.transpose(Lambda)

        ind_sup = [i for i,v in enumerate(Lambda) if v > espeig] # Find index of Lambda with positive eigenalues (or greater than espeig)
        Phi = Phi[:,ind]                                # Sort eigenvectors by ascending order of eigenvalues
        Phi = Phi[:,ind_sup]                    # Keep only eigenvectors with positive eigenvalue (or greater than espeig)
        Lambda = Lambda[ind_sup]              # Keep positive eigenvalue


        # Stabilized Gram-Schmidt orthornormalization 
        Phinorm = self.stabilized_gram_schmidt(Phi, np.eye(Phi.shape[1]))

        # Evaluate normalized dataset with PCA
        Eta_ = np.matmul(np.diag(1.0/np.sqrt(Lambda)), np.transpose(Phi))  
        Eta = np.matmul(Eta_, np.transpose(np.transpose(X)-Xbar))
    
        self.etad = Eta
        self.xbar = Xbar 
        self.phi = Phinorm
        self.lambd = Lambda

    def titan_diffusionmapbasis(self, etad):
        """ Copyright K. Bulthuis 10 April 2019 
            J. Coheur 02 February 2020 (Python implementation)
        --- Ojective
                    Build a diffusion-map basis [g]

                    [k] : affinity kernel matrix with
                    [k]_{ij} = exp(-1/(4*epsopt)*norm(eta^{i}-eta^{j}))
                    where epsopt is a hyperparameter that is predefined
                        or determined with an optimization algorithm

                    [p] = [b]^{-1}[k] is the transition matrix with
                        [b] a matrix whose entries are 
                        [b]_{ij} = sum_{j=1}^{N} [k]_{ij}

                    The diffusion-map basis is determined from the eigenvectors
                    and eigenvalues of the transition matrix

        ---INPUT
                etad : normalized dataset with PCA (nu times N)

        ---OUTPUT
                gnorm : normalized diffusion-map basis [g] (N times m)
                Lambda : diagonal matrix with sorted eigenvalues of [p] (m times m)
                b : matrix [b] whose entries are [b]_{ij} = sum_{j=1}^{N} [k]_{ij}
                k : affinity kernel matrix
        """


        # Determine hyperparameter epsopt in the affinity kernel matrix [k]
        # Two options:
        #   Model.Opt.optimizationeps==1: Use optimization algorithm in Soize et
        #       al.(2019)
        #   Ref: Soize et al., Entropy-based closure for probablistic learning on manifolds, J. Comp. Phys., 2019
        #   Model.Opt.optimizationeps==0: Predefined value

        epsopt = self.Opt.epsvalue

        # Bild affinity kernel matrix [k]

        N = etad.shape[1]
            
        distVec = pdist(np.transpose(etad)) # Pairwise distance between pairs of observations
        corVec = np.exp(-1.0/(4.0*epsopt)*distVec**2)  
        k = squareform(corVec)
        k[0:-1:N+1] = 1
        
        # Build [b] positive-definite diagonal real matrix
        # whose entries are [b]_{ij} = sum_{j=1}^{N} [k]_{ij}

        bvec = np.sum(k,axis=1)
        b = np.diag(bvec)

        # Build transition matrix [p] = [b]^{-1} [k]
        # Backslash matlab ? 
        p = np.matmul(linalg.inv(b),k)

        # Evaluate eigenvalues and eigenvectors of the transition matrix [p]
        Lambda, psi = np.linalg.eig(p) 
        psi = np.real(psi)
        Lambda = np.real(Lambda)    # to impose positive eigenvalues (has to be changed)
        
        ind = np.argsort(Lambda)    # sort eigenvalues in ascending order 
        ind = np.flip(ind)  # Flip to have them in descending order
        Lambda = Lambda[ind]
        psi = psi[:,ind]    # Sort eigenvectors by descending order of eigenvalues

        # Select the m first basis vectors (see Soize et al., 2019)

        if(self.Opt.optimizationm==1):
            L = 0.1
            r = Lambda/Lambda(2)
            r[0] = np.NaN
            r[1] = np.NaN
            vecm = [i for i,v in enumerate(r) if v < L] # m = find(r<L,1,'first')
            m = vecm[0]

            # m = -1 + find(r<L,1,'first')
            # m = m-1
        elif(self.Opt.optimizationm==2): 
            self.Opt.display = 0
            self.Opt.mmax = 0.1*N
            self.Opt.errmtol = 1e-2
            # m = titan_search_moptimal(Inputs)
        else:
            m = self.Opt.m
        

        # nm = 20;
        # figure;
        # semilogy(Lambda(1:nm),'MarkerFaceColor',[0 0 1],'MarkerEdgeColor',[0 0 1],'MarkerSize',10,'Marker','o','Color',[0 0 1]);
        # ylim([1e-4,1]);
        # set(gca,'YGrid','on');
        # 
        # r = Lambda(1:nm)/Lambda(2);
        # r(1) = NaN;
        # r(2) = NaN;
        # figure;
        # semilogy(1:nm,r,'MarkerFaceColor',[0 0 1],'MarkerEdgeColor',[0 0 1],'MarkerSize',10,'Marker','o','Color',[0 0 1]);
        # hold on;
        # semilogy(1:nm,0.1*ones(nm,1),'black');
        # set(gca,'YGrid','on');

        self.Lambda = Lambda[0:m]
        psi = psi[:,0:m]

        # Stabilized Gram-Schmidt orthornormalization (https://en.wikipedia.org/wiki/Gram-Schmidt_process)

        self.g = self.stabilized_gram_schmidt(psi, b) # psinorm 
        self.b = b 
        self.K = k

        # g = zeros(size(psinorm));
        # 
        # for ii=1:m
        #     g(:,ii) = Lambda(ii)*psinorm(:,ii);
        # end  

        #p*psinorm-g

        # modificatio
        # gnorm = gnorm(:,2:m); % remove first constant vector from the basis %
        # Lambda = Lambda(2:m); % remove first eigenvalue


    def  titan_reduced_order_representation(self, Eta,g):

        #                
        # Copyright K. Bulthuis 10 April 2019 
        #--- Ojective
        #            Perform a reduced-order representation [H^(m)] of the random matrix
        #            [H]
        #            [H^(m)] = [Z] [g]'
        #            where [Z] = [H] [a] or [H] = [Z] [g]'
        #                  [a] = [g] ([g]'*[g])^{-1}
        #            We have [H^(m)] the reduced-order representation (nu \times N)
        #                 of [H] (nu \times N)
        #                 [g] Matrix with the diffusion-map vectors (N \times m)
        #
        #---INPUT
        #         Eta : normalized dataset with PCA (nu \times N)
        #         g : Matrix with the diffusion-map vectors (N \times m)
        #
        #---OUTPUT
        #         Z : Realization of the random matrix [Z] (nu \times m)

        # Build [a] matrix for the least-squares approximation

        A = np.matmul(g, np.transpose(g))     # Normal matrix
        self.a = np.matmul(np.transpose(g), linalg.inv(A))   # Solve a*A = g

        # Reduced-order representation of eta

        self.zd = np.matmul(Eta, np.transpose(self.a))


    def stabilized_gram_schmidt(self, phi, b): 
        # Stabilized Gram-Schmidt orthornormalization (https://en.wikipedia.org/wiki/Gram-Schmidt_process)

        m = phi.shape[1]
        N = phi.shape[0]
        Phinorm = np.zeros((N , m))
        Phinorm[:,0] = np.matmul(phi[:,0],b) 
        Phinorm[:,0] = phi[:,0]/np.sqrt(np.matmul(Phinorm[:,0],phi[:,0]))

        for ii in range(1, m): 
            Phinorm[:,ii] = phi[:,ii]
            for jj in range(ii):
                phi_ii_b = np.matmul(Phinorm[:,ii], b)
                phi_jj_b = np.matmul(Phinorm[:,jj], b)
                Phinorm[:,ii] = Phinorm[:,ii] - np.matmul(phi_ii_b,Phinorm[:,jj])/np.matmul(phi_jj_b,Phinorm[:,jj])*Phinorm[:,jj]
            phi_ii_b = np.matmul(Phinorm[:,ii], b)
            Phinorm[:,ii] = Phinorm[:,ii]/np.sqrt(np.matmul(phi_ii_b,Phinorm[:,ii]))

        return Phinorm

class ModelOption(): 

    def __init__(self):

        self.scaling = 0
        self.optimizationm = 0
        self.epsvalue = 0   
        self.m = 4

class ExpDesign(): 

    def __init__(self):

        self.X = 0
        self.Y = 0

def titan_kriging(Model): 
    """ Still need to be reimplemented. """
    return 0 

def titan_PCE_quadrature(Model): 
    """ Still need to be reimplemented. """
    return 0 

def titan_PCE_OLS(Model): 
    """ Still need to be reimplemented. """
    return 0 

def titan_ME_PCE_OLS(Model): 
    """ Still need to be reimplemented. """
    return 0 







