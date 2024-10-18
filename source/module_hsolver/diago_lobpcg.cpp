#include "diago_lobpcg.h"

#include <module_hsolver/kernels/math_kernel_op.h>

#include <ATen/kernels/blas.h>
#include <ATen/kernels/lapack.h>

#include <algorithm> // std::min

namespace hsolver{

template <typename T, typename Device>
DiagoLOBPCG<T, Device>::DiagoLOBPCG(const int ld_psi, const int n_eigenpairs)
                                    : ldx(ld_psi), n_eigenpairs(n_eigenpairs),
                                    n_max_subspace(std::min(n_eigenpairs + 5, ld_psi)),
                                    size_space(3 * n_max_subspace)
{
    // Constructor for the DiagoLOBPCG class.
    // do:
    // 1. set the number of eigenpairs to compute
    // 2. set iteration parameters
    // 3. allocate memory for the solver according to the number of eigenpairs and leading dimension

    // according to Knyazev, a subspace size slightly larger than the number of eigenpairs
    // would imporve the convergence of LOBPCG.
    // this->n_max_subspace    = std::min(n_eigenpairs + 5, ld_psi); // not more than dimension of matrix

    this->r_type        = ct::DataTypeToEnum<Real>::value;
    this->t_type        = ct::DataTypeToEnum<T>::value;
    this->device_type   = ct::DeviceTypeToEnum<Device>::value;

    // We will allocate memory for the expansion space,
    // and corresponding Matrix-Vector results and residuals here.
    // LOBPCG requires 3 times the size of eigenpairs sought:
    // [X P W], P and W are working space that could reach a maximum size 
    // of X(ld_psi x nmax), so the total size prepared is 3 * (ld_psi x nmax) blocks.
    // this->size_space = 3 * n_max_subspace;  ///< total max space size of V = [x p w]

    this->v     = std::move(ct::Tensor(t_type, device_type, {ldx, size_space}));
    this->hv    = std::move(ct::Tensor(t_type, device_type, {ldx, size_space}));
    this->sv    = std::move(ct::Tensor(t_type, device_type, {ldx, size_space}));

    this->x     = std::move(ct::Tensor(t_type, device_type, {ldx, n_max_subspace}));
    this->hx    = std::move(ct::Tensor(t_type, device_type, {ldx, n_max_subspace}));
    this->sx    = std::move(ct::Tensor(t_type, device_type, {ldx, n_max_subspace}));

    this->h_reduced     = std::move(ct::Tensor(t_type, device_type, {size_space, size_space}));
    this->eig_reduced   = std::move(ct::Tensor(r_type, device_type, {size_space}));

    this->residual      = std::move(ct::Tensor(t_type, device_type, {ldx, n_max_subspace}));
    this->r_norm        = std::move(ct::Tensor(r_type, device_type, {n_max_subspace}));

    // this->overlap       = std::move(ct::Tensor(t_type, device_type, {n_max_subspace, n_max_subspace}));
}

template <typename T, typename Device>
hsolver::DiagoLOBPCG<T, Device>::~DiagoLOBPCG()
{
    // free memory allocated in the constructor
}

template <typename T, typename Device>
int DiagoLOBPCG<T, Device>::diag(const HPsiFunc& hpsi_func,
                                 const SPsiFunc& spsi_func,
                                 const PrecndFunc& precnd_func,
                                 const bool solving_generalized,
                                 Real* eigenvalue_in,
                                 T* psi_in,
                                 const double tolerance,
                                 const int max_iterations)
{
// 0. check init guess and ortho
    // first check initial guess
    // 1. is not zero
    // 2. is orthonormal
    // if not, we generate a random guess a set of vectors randomly distributed around the origin
    check_init_guess(this->ldx, this->n_max_subspace, psi_in);
    // then we will used this orthonormalized psi as the initial guess
    // and psi serves as the current approximation of eigenvectors
    this->psi = std::move(ct::TensorMap(psi_in, t_type, device_type, {this->ldx, this->n_max_subspace}));
    // and do S-ortho if needed
    if(solving_generalized){
        spsi_func(this->psi.template data<T>(), this->sv.template data<T>(), this->ldx, this->n_max_subspace);
        s_ortho(ldx, n_max_subspace, psi.template data<T>(), sv.template data<T>());
    }

// 1. first loop
    // 1.1 first round Rayleigh-Ritz
    // construct the reduced matrix and diagonalization
    // v = [x p w], v[x block] = psi
    syncmem_complex_op()(this->v.template data<T>(), this->psi.template data<T>(), this->ldx * this->n_max_subspace);
    // hv = H * v
    hpsi_func(this->v.template data<T>(), this->hv.template data<T>(), this->ldx, this->n_max_subspace);
    // h_reduced(n_max_subspace, n_max_subspace) = v'(n_max_subspace, ldx) Hv(ldx, n_max_subspace)
    // now topleft corner of h_reduced is v'Hv.
    gemm_op<T, Device>()(
        this->ctx, 'C','N', n_max_subspace, n_max_subspace, ldx,
        this->one, this->v.template data<T>(), ldx, this->hv.template data<T>(), ldx,
        this->zero, this->h_reduced.template data<T>(), n_max_subspace
    );
    eigh(n_max_subspace,size_space, this->h_reduced.template data<T>(), this->eig_reduced.template data<Real>());
    // v[x p w]
    compute_residual(ldx, n_max_subspace, this->hv.template data<T>(), this->sv.template data<T>(),
        this->eig_reduced.template data<Real>(),
        this->v.template data<T>() // to be discussed
    );
    // preconditioning
    // W = T(H - λS)X = TR
    // precnd_func(n, m, res, res, eig);


// 2. main loop
for(int iter = 0; iter < this->max_iterations; ++iter){
    // 2.1. Rayleigh-Ritz
    // construct the reduced matrix and diagonalization
    // Solve Generalized EigenValue Problem of Hermitian Matrix

    // eigh

    // update

    // 2.2 update X, AX and, if required BX

    // 2.3 compute residuals & norms

    // 2.4 check convergence and locking

    // 2.5 check active eigenvectors X and update corresponding blocks P, W

} // end of main loop

    // 3. post loop
    return 0;
}


// --- below private --- //


template <typename T, typename Device>
void DiagoLOBPCG<T, Device>::ortho(int n, int m, T* x)
{
    // ortho X(n,k) by Cholesky.
    // n >> k in general
    // First, do overlap X'X gemm
    // using member variable to store intermediate results
    ct::Tensor overlap(t_type, device_type, {m, m});
    // overlap(m, m) = X'(m, n) * X(n, m)
    gemm_op<T, Device>()(
        this->ctx, 'C','N', m,m,n,
        this->one, x,n, x,n,
        this->zero,overlap.data<T>(), m
    );
    
    // 2. solve Cholesky decomposition of X'X = LL'
    // dpotrf
    // overlap = L^T
    ct::kernels::lapack_potrf<T, ct_Device>()('U', m, overlap.data<T>(), m);
    // 3. solve Q = X * L^{-T} or QL' = X
    // dtrsm
    // or dtrtri overlap = L^(-T)
    ct::kernels::lapack_trtri<T, ct_Device>()('U', 'N', m, overlap.data<T>(), m);
    // gemm Q = X * L^{-T}
    ct::Tensor Q(t_type, device_type, {n, m}) ;
    gemm_op<T, Device>()(
        this->ctx, 'N','N', m,m,n,
        this->one, x,n, overlap.data<T>(),m,
        this->zero, this->Q.data<T>(), m
    );
    // copy back to X
    syncmem_complex_op()(x.template data<T>(), Q.template data<T>(), n*m);
}


template <typename T, typename Device>
void hsolver::DiagoLOBPCG<T, Device>::ortho_against_y(int n, int m, int k, T* x, T* y)
{
    // first check input y is orthonormal than do ortho_against_y
    // X = X - Y * y_coeff
    // Y'BY y_coeff = Y'BX
    // X = X - Y(Y'Y)^{-1}Y'X
}

template <typename T, typename Device>
void hsolver::DiagoLOBPCG<T, Device>::s_ortho(int n, int m, T* x, T* sx)
{
    // S-ortho X(n,k) by Cholesky.
    // First, do overlap X'SX gemm
    // overlap(m, m) = X'(m, n) * SX(n, m)
    ct::Tensor overlap(t_type, device_type, {m, m});
    gemm_op<T, Device>()(
        this->ctx, 'C','N', m,m,n,
        this->one, x,n, sx,n,
        this->zero,overlap.data<T>(), m
    );
    
    // 2. solve Cholesky decomposition of X'SX = LL'
    // dpotrf
    // overlap = L^T
    ct::kernels::lapack_potrf<T, ct_Device>()('U', m, overlap.data<T>(), m);
    // 3. solve Q = X * L^{-T} or QL' = X
    // dtrsm
    // or dtrtri overlap = L^(-T)
    ct::kernels::lapack_trtri<T, ct_Device>()('U', 'N', m, overlap.data<T>(), m);
    // gemm Q = X * L^{-T}
    ct::Tensor Q(t_type, device_type, {n, m});
    gemm_op<T, Device>()(
        this->ctx, 'N','N', m,m,n,
        this->one, x,n, overlap.data<T>(),m,
        this->zero, this->Q.data<T>(), m
    );
    // copy back to X
    syncmem_complex_op()(x.template data<T>(), Q.template data<T>(), n*m);

    gemm_op<T, Device>()(
        this->ctx, 'N','N', m,m,n,
        this->one, sx,n, overlap.data<T>(),m,
        this->zero, this->Q.data<T>(), m
    );
    // copy back to SX
    syncmem_complex_op()(sx.template data<T>(), Q.template data<T>(), n*m);
}

template <typename T, typename Device>
void hsolver::DiagoLOBPCG<T, Device>::s_ortho_against_y(int n, int m, int k, T* x, T* y, T* sy)
{
}

template <typename T, typename Device>
void DiagoLOBPCG<T, Device>::check_init_guess(int n, int m, T *evec)
{
    // first check if it is zero

    // if zero guess, generate a random set of vectors

    // then check orthonormality

    // do ortho if needed
}

template <typename T, typename Device>
void DiagoLOBPCG<T, Device>::eigh(int n, int ldx, T* h, Real* eig)
{
    // call standard dense symmetric generalized eigen solver
    ct::kernels::lapack_dnevd<T, ct_Device>()('V', 'U', h, n, eig);
    // we need dsyev to pass
    // 1. N The order of the matrix A
    // 2. lda The leading dimension of the array A
    // as reduced matrix A=h will be (size_space x size_space)
    // but only top-left corner is filled, so lda = size_space, n = n_max_subspace
}

template <typename T, typename Device>
void DiagoLOBPCG<T, Device>::compute_residual(int n, int m, const T* hx, const T *sx,const T* eig, T* res)
{
    // compute residual
    // R = HX - λSX
    // res = HX
    syncmem_complex_op()(res, hx, n*m);
    // res = HX - λSX
    // for loop axpy
}

template <typename T, typename Device>
void DiagoLOBPCG<T, Device>::check_convergence(const T* x,
                                               const T* hx,
                                               const T* sx,
                                               const T* eig,
                                               int n,
                                               int m,
                                               Real tol,
                                               int& nconv)
{
}

// template<typename T, typename Device>
// void DiagoLOBPCG<T, Device>::ortho_cho(
// 		ct::Tensor& workspace_in, 
// 		ct::Tensor& psi_out, 
// 		ct::Tensor& hpsi_out, 
// 		ct::Tensor& hsub_out)
// {
//     // hsub_out = psi_out * transc(psi_out)
//     ct::EinsumOption option(
//         /*conj_x=*/false, /*conj_y=*/true, /*alpha=*/1.0, /*beta=*/0.0, /*Tensor out=*/&hsub_out);
//     // 
//     hsub_out = ct::op::einsum("ij,kj->ik", psi_out, psi_out, option);

//     // set hsub matrix to lower format;
//     ct::kernels::set_matrix<T, ct_Device>()(
//         'L', hsub_out.data<T>(), this->n_band);

//     ct::kernels::lapack_potrf<T, ct_Device>()(
//         'U', this->n_band, hsub_out.data<T>(), this->n_band);
//     ct::kernels::lapack_trtri<T, ct_Device>()(
//         'U', 'N', this->n_band, hsub_out.data<T>(), this->n_band);

//     this->rotate_wf(hsub_out, psi_out, workspace_in);
//     this->rotate_wf(hsub_out, hpsi_out, workspace_in);
// }

} // namespace hsolver