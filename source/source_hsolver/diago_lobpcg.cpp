#include "source_hsolver/diago_lobpcg.h"

#include <random> // make random initial guess
#include <cmath>  // std::sqrt

#include <source_base/kernels/math_kernel_op.h>
// #include <source_base/global_function.h>
#include <source_base/module_container/ATen/core/tensor.h>
#include <source_base/module_container/ATen/core/tensor_types.h>
#include <source_base/module_container/ATen/core/tensor_map.h>

#include <source_hsolver/kernels/bpcg_kernel_op.h> // normalize_op, precondition_op, apply_eigenvalues_op

namespace ct = container;

namespace hsolver {

template <typename T, typename Device>
DiagoLOBPCG<T, Device>::DiagoLOBPCG(
    const Real *precondition,
    const int nband,
    const int ndim, const int nmax)
{
    // 0. setup: allocate and initialize
    // this block will take the dimension parameters,
    // and do the initialization, set up the memory space for the tensors
    // Initialize data types and device information
    this->r_type_ = ct::DataTypeToEnum<Real>::value;
    this->t_type_ = ct::DataTypeToEnum<T>::value;
    this->device_type_ = ct::DeviceTypeToEnum<Device>::value;

    this->one = &one_;
    this->zero = &zero_;
    this->neg_one = &neg_one_;

    // Set problem dimensions
    this->n_band_ = nband;
    this->n_dim_ = ndim;
    this->n_max_ = nmax;

    // Calculate search space parameters
    this->len_space_ = 3 * n_max_; // Total search space size (3 * n_max_), for [X, P, W]
    this->ind_x_ = 0;
    this->ind_p_ = n_max_;
    this->ind_w_ = 2 * n_max_;
    this->n_active_ = n_max_;

    // ----- Allocate memory for all tensors -----
    // --- mostly on device, and NOT set to zero!
    // prec: real!
    // Store reference to preconditioner data (host side)
    // only a reference, does not own memory!
    this->h_prec_ = std::move(ct::TensorMap((void*)precondition, r_type_, device_type_, {n_dim_}));
    // Preconditioner, device side
    this->prec_ = std::move(ct::Tensor(r_type_, device_type_, {n_dim_}));

    // Main search space and matrix-vector products
    // Eigenvectors of the problem [n_max_]
    // eig: Real!
    this->eig_ = std::move(ct::Tensor(r_type_, device_type_, {n_max_}));
    // Eigenvectors of the problem [n_dim_, n_max_]
    this->evec_ = std::move(ct::Tensor(t_type_, device_type_, {n_dim_, n_max_}));

    // 3 spaces: [X, P, W], H * [X, P, W], S * [X, P, W] (if generalized)
    this->space_ = std::move(ct::Tensor(t_type_, device_type_, {n_dim_, len_space_}));
    this->hspace_ = std::move(ct::Tensor(t_type_, device_type_, {n_dim_, len_space_}));
    this->sspace_ = std::move(ct::Tensor(t_type_, device_type_, {n_dim_, len_space_}));
    // Reduced problem
    this->h_red_ = std::move(ct::Tensor(t_type_, device_type_, {len_space_, len_space_}));
    // e_red: Real!
    this->e_red_ = std::move(ct::Tensor(r_type_, device_type_, {len_space_}));

    // Temporary storage
    this->x_new_ = std::move(ct::Tensor(t_type_, device_type_, {n_dim_, n_max_}));
    this->hx_new_ = std::move(ct::Tensor(t_type_, device_type_, {n_dim_, n_max_}));
    this->sx_new_ = std::move(ct::Tensor(t_type_, device_type_, {n_dim_, n_max_}));

    // Residuals: (R = H*X - λ*S*X)
    this->residual_ = std::move(ct::Tensor(t_type_, device_type_, {n_dim_, n_max_}));
    // convergence
    this->r_norm_ = std::move(ct::Tensor(r_type_, device_type_, {n_max_})); // 2-norm of vectors, Real
    this->done_ = std::move(ct::Tensor(ct::DataType::DT_INT, device_type_, {n_max_})); // use int here temporarily for bool

    // Expansion coefficients
    // u_x_ and u_p_ will be allocated inside iter.
    // this->u_x_ = std::move(ct::Tensor(t_type_, device_type_, {len_space_, n_max_}));
    // this->u_p_ = std::move(ct::Tensor(t_type_, device_type_, {len_space_, n_active__})); // maximum size

    // Initialize tensors to zero
    this->prec_.zero();
    this->eig_.zero();
    this->evec_.zero();
    this->space_.zero();
    this->hspace_.zero();
    this->sspace_.zero();
    this->h_red_.zero();
    this->e_red_.zero();

    this->x_new_.zero();
    this->hx_new_.zero();
    this->sx_new_.zero();

    this->residual_.zero();
    this->r_norm_.zero();
    this->done_.zero();

}

template <typename T, typename Device>
DiagoLOBPCG<T, Device>::~DiagoLOBPCG()
{
    // Free all allocated memory
    // All tensors will be automatically destroyed when going out of scope
    // No explicit deallocation needed for ct::Tensor

    // Note, we do not need to free the h_prec and psi pointer as they are refs to the outside data
}

template <typename T, typename Device>
bool DiagoLOBPCG<T, Device>::diag(
    const HPsiFunc& hpsi_func,
    const SPsiFunc& spsi_func,
    // const PrecndFunc& precnd_func,
    const bool gen_eig,
    Real *eigenvalue_in,
    T *psi_in, // psi_in will store the final eigenvectors output
    const int ld_psi_in,// should only be used with psi_in!!! as leading dimension of input/output psi_in
    const double tolerance,
    const int max_iter)
{
#ifdef LOCKING_BY_TRACE
std::cout << "Using locking by trace." << std::endl;
#endif
// #ifdef DEBUG_LOBPCG
#ifdef DEBUG_SCF
std::cout << "----- START LOBPCG -----" << std::endl;
std::cout << "n_band=" << this->n_band_ <<  ", n_dim=" << this->n_dim_ << ", n_max=" << this->n_max_ << std::endl;
std::cout << "ld_psi_in=" << ld_psi_in << std::endl;
std::cout << "tol=" << tolerance << std::endl;
std::cout << "max_iter=" << max_iter << std::endl;
//print input matrix: hpsi_func
// verified!
        // T* test_in = new T[this->n_dim_ * this->n_dim_];
        // T* test_out = new T[this->n_dim_ * this->n_dim_];
        // // Init Identity
        // for(int i = 0; i < this->n_dim_ * this->n_dim_; ++i) test_in[i] = T(0.0);
        // for(int i = 0; i < this->n_dim_; ++i) test_in[i + i * this->n_dim_] = T(1.0);

        // // Call hpsi
        // hpsi_func(test_in, test_out, this->n_dim_, this->n_dim_);

        // for (int i = 0; i < this->n_dim_; ++i) {
        //     for (int j = 0; j < this->n_dim_; ++j) {
        //         // std::cout << "H[" << i << "," << j << "] = " << std::real(test_out[i + j * this->n_dim_]) << std::endl;
        //         std::cout << std::real(test_out[i + j * this->n_dim_]) << " ";
        //     }
        //     std::cout << std::endl;
        // }
        // delete[] test_in;
        // delete[] test_out;
// return false; // for now just test the hpsi_func
#endif
    // [] for actual shape in memory; () for effective shape referenced

    // --- 0. init (mostly done by constructor) ---
    // Note:
    // inside iteration loop, eig_ and evec_ are used to store the current iteration's results.
    // They are initialized with the initial guess,
    // and written to input psi_in & eigenvalue_in on exit.

    // Copy initial guess to search space (X block)
    // copy psi_in(ld_psi_in)[n_dim_, n_band_] to evec_[n_dim_, n_max_] left block; different leading dimension!
    syncmem_complex_2d_op()(this->evec_.data<T>(), this->n_dim_, psi_in, ld_psi_in, this->n_dim_, this->n_band_);
    // syncmem_complex_op()(this->evec_.data<T>(), psi_in, this->n_band_ * this->n_dim_);

    // Copy precondition from host h_prec_ to device prec_
    syncmem_var_h2d_op()(this->prec_.data<Real>(), this->h_prec_.data<Real>(), this->n_dim_);

    this->check_init_guess(this->n_dim_, this->n_max_, evec_.data<T>(), this->n_dim_);

    // If generalized problem, compute S*X and S-orthogonalize
    if (gen_eig) {
    //     spsi_func(evec_.data<T>(), sx_new_.data<T>(), n_dim_, n_max_);
    //     this->s_ortho(x_block, sx_block);
    }

// --- 1. first iter --- explicit do the fist Rayleigh-Ritz for X'HX
#ifdef DEBUG_LOBPCG
std::cout << "--- Debug: Entering first iteration ---" << std::endl;
#endif

    // copy evec to space
    copy_op(n_dim_*n_max_, evec_.data<T>(), 1, space_.data<T>(), 1);
    if(gen_eig) {
        copy_op(n_dim_*n_max_, sx_new_.data<T>(), 1, sspace_.data<T>(), 1);
    }
    // compute hspace_
    hpsi_func(space_.data<T>(), hspace_.data<T>(), n_dim_, n_max_);
#ifdef DEBUG_LOBPCG
std::cout << "--- First iter Rayleigh-Ritz ---" << std::endl;
#endif
    // --- 1.1 initial Rayleigh-Ritz procedure ---
    len_working_ = n_max_; // first round, no p no w
    this->rayleigh_ritz(space_.data<T>(), hspace_.data<T>(), len_space_, n_dim_, len_working_,
        h_red_.data<T>(), e_red_.data<Real>());    // now (u, lambda) = (h_red, e_red)

    // eig_(1:n_max_) = e_red_(1:n_max_)
    copy_real_op(n_max_, e_red_.data<Real>(), 1, eig_.data<Real>(), 1);

    // --- 1.2 compute the Ritz vectors ---
    // store ritz vectors by updating space_ and hspace_, * h_red_
    // space[n_dim_, len_space_]
    // evec_(n_dim_, n_max_) = space_(n_dim_, n_max_) * h_red_(n_max_, n_max_)
    // space_(n_dim_, n_max_) = evec_(n_dim_, n_max_)
    gemm_op('N', 'N', n_dim_, n_max_, n_max_,
        one, space_.data<T>(), n_dim_, h_red_.data<T>(), len_space_,
        zero, evec_.data<T>(), n_dim_);
    // space_(n_dim_, n_max_) = evec_(n_dim_, n_max_)
    copy_op(n_dim_*n_max_, evec_.data<T>(), 1, space_.data<T>(), 1);
    // evec_ = hspace_ * h_red_
    // evec_(n_dim_, n_max_) = hspace_(n_dim_, n_max_) * h_red_(n_max_, n_max_)
    gemm_op('N', 'N', n_dim_, n_max_, n_max_,
        one, hspace_.data<T>(), n_dim_, h_red_.data<T>(), len_space_,
        zero, evec_.data<T>(), n_dim_);
    // hspace_(n_dim_, n_max_) = evec_(n_dim_, n_max_)
    copy_op(n_dim_*n_max_, evec_.data<T>(), 1, hspace_.data<T>(), 1);
    if(gen_eig){
        // // evec_ = sspace_ * h_red_
        // gemm_op('N', 'N', n_dim_, n_max_, n_max_,
        //     one, sspace_.data<T>(), n_dim_, h_red_.data<T>(), len_space_,zero, evec_.data<T>(), n_dim_);
        // // evec_ = sspace_ * h_red_
        // gemm_op('N', 'N', n_dim_, n_max_, n_max_,
        //     one, sspace_.data<T>(), n_dim_, h_red_.data<T>(), len_space_,zero, evec_.data<T>(), n_dim_);
    }
#ifdef DEBUG_LOBPCG
std::cout << "--- first iter residuals ---" << std::endl;
#endif
    // --- 1.3 compute the residuals and norms --- r <- Hx - eig Sx
    copy_op(n_dim_ * n_max_, hspace_.data<T>(), 1, residual_.data<T>(), 1);
    // !use axpy to compute the residuals, note that eig is real array, and should be cast to T before passed into axpy
    if(gen_eig){
        // residual = Hx - eig Sx
        for (int i = 0; i < n_max_; ++i){
            T *r_col = residual_.data<T>() + i * n_dim_;
            const T *sspace_col = sspace_.data<T>()  + i * n_dim_;
            const Real lambda = eig_.data<Real>()[i];
            T alpha = T(-lambda);
            axpy_op(n_dim_, &alpha, sspace_col, 1, r_col, 1);
        }
    } else {
        // residual = Hx - eig x
        for (int i = 0; i < n_max_; ++i){
            T *r_col = residual_.data<T>() + i * n_dim_;
            const T *space_col = space_.data<T>()  + i * n_dim_;
            const Real lambda = eig_.data<Real>()[i];
            T alpha = T(-lambda);
            axpy_op(n_dim_, &alpha, space_col, 1, r_col, 1);
        }
    }
#ifdef DEBUG_LOBPCG
std::cout << "--- first iter: preconditioned residuals ---" << std::endl;
#endif
    // --- 1.4 compute the preconditioned residuals W = TR ---
    // w = T * residual
    ind_x_ = 0;
    // first round: [ X | W ]
    //              ^   ^
    //              |   |
    //            ind_x | ind_w
    ind_w_ = ind_x_ + n_max_;
    // precondition_op
    // void operator()(const int& dim,
    //                 T* psi_iter,
    //                 const int& nbase,     // offset
    //                 const int& notconv,   // number of vectors
    //                 const Real* precondition,
    //                 const Real* eigenvalues);
    // space_(from ind_w_ column) = T * r
    copy_op(n_dim_ * n_max_,
            residual_.data<T>(), 1,
            space_.data<T>() + ind_w_ * n_dim_, 1);
    precondition_op(n_dim_, space_.data<T>(), ind_w_, n_max_, prec_.data<Real>(), eig_.data<Real>());

    // --- 1.5 orthogonalize W; and then orthonormalize it ---
    if(gen_eig){}
    // Corrected argument order: normalize W (2nd ptr) against X (1st ptr)
    ortho_against_y(n_dim_, n_max_, n_max_, space_.data<T>() + ind_w_ * n_dim_, n_dim_, space_.data<T>(), n_dim_);
#ifdef DEBUG_LOBPCG
    std::cout << "--- first iter over ---" << std::endl;
    std::cout << "Eigenvalues after first Rayleigh-Ritz:" << std::endl;
    for(int i=0; i<this->n_max_; ++i) {
        std::cout << this->eig_.data<Real>()[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Residual norms after first Rayleigh-Ritz:" << std::endl;
    for (int i = 0; i < this->n_max_; ++i) {
        T *r_col = this->residual_.data<T>() + i * this->n_dim_;
        Real norm = nrm2_op(this->n_dim_, r_col, 1) / std::sqrt(static_cast<double>(this->n_dim_));
        std::cout << norm << " ";
    }
    std::cout << std::endl;

    // return false;
// current working checkpoint
#endif
// --- 2. Main LOBPCG iteration loop ---
    bool converged = false;
    // before entering the loop
    // first set convergence parameters and flags:

    // tol_rms_ = tolerance;
    // tol_max_ = 10.0 * tolerance;

    n_active_ = n_max_;
    // Initialize convergence flags
#ifdef DEBUG_LOBPCG
    // return false; // Removed early exit
    std::cout << "--- main loop: start ---" << std::endl;
    std::cout << std::flush;
#endif
    setmem_int_op()(done_.data<int>(), false, n_max_);
    for (int iter = 0; iter < max_iter; ++iter) {
// #ifdef DEBUG_SCF
// std::cout << "----- LOBPCG main loop: iter " << iter << " -----" << std::endl;
// #endif

#ifdef DEBUG_LOBPCG
std::cout << "----- main loop: iter " << iter << " -----" << std::endl;
#endif
        // --- Rayleigh-Ritz ---
        // --- 2.1 hw = h * w ---
#ifdef DEBUG_LOBPCG
std::cout << "----- before hpsi  -----" << std::endl;
// print hpsi info
// std::cout << "hpsi input W (first 5 elements of first vector): " << std::endl;
// for(int i=0; i< std::min(5, n_dim_); ++i) {
//     std::cout << space_.data<T>()[ind_w_ * n_dim_ + i] << " ";
// }
// std::cout << std::endl;
std::cout << "n_dim_: " << n_dim_ << std::endl;
std::cout << "ind_w_: " << ind_w_ << std::endl;
std::cout << "n_active_: " << n_active_ << std::endl;
#endif        
        hpsi_func(space_.data<T>() + n_dim_ * ind_w_, hspace_.data<T>() + n_dim_ * ind_w_, n_dim_, n_active_);
        // --- 2.2 construct the reduced matrix and diagonalization ---
#ifdef DEBUG_LOBPCG
std::cout << "----- end hpsi  -----" << std::endl;
#endif    
        len_working_ = n_max_ + 2 * n_active_; // X P W
        if(0 == iter) { // first round, no P is constructed yet
            len_working_ = 2 * n_max_;
        }
#ifdef DEBUG_LOBPCG
std::cout << "->current work size: " << std::endl;
std::cout << "n_active_: " << n_active_ << std::endl;
std::cout << "n_max_: " << n_max_ << std::endl;
std::cout << "len_space_: " << len_space_ << std::endl;
std::cout << "len_working_: " << len_working_ << std::endl;
#endif

#ifdef DEBUG_LOBPCG
std::cout << "--- main loop: Rayleigh-Ritz ---" << std::endl;
#endif
        this->rayleigh_ritz(space_.data<T>(), hspace_.data<T>(), len_space_, n_dim_, len_working_,
                h_red_.data<T>(), e_red_.data<Real>());    // now (u, lambda) = (h_red, e_red)

#ifdef DEBUG_LOBPCG
        std::cout << "Eigenvalues after loop Rayleigh-Ritz:" << std::endl;
        for(int i=0; i< std::min(5, this->n_max_); ++i) {
            std::cout << e_red_.data<Real>()[i] << " ";
        }
        std::cout << std::endl;
        
        // if (iter == 0) {
        //      // force exit to check first loop iter
        //      return false;
        // }
#endif
        // eig_(1:n_max_) = e_red_(1:n_max_)
        copy_real_op(n_max_, e_red_.data<Real>(), 1, eig_.data<Real>(), 1);

        // --- 2.3 update X, AX and, if required BX ---
#ifdef DEBUG_LOBPCG
std::cout << "--- main loop: update X, AX and, if required BX ---" << std::endl;
#endif
        // x_new_ = space * h_red
        gemm_op('N', 'N', n_dim_, n_max_, len_working_,
            one, space_.data<T>(), n_dim_, h_red_.data<T>(), len_space_,
            zero, x_new_.data<T>(), n_dim_);
        // hx_new_ = hspace * h_red
        gemm_op('N', 'N', n_dim_, n_max_, len_working_,
            one, hspace_.data<T>(), n_dim_, h_red_.data<T>(), len_space_,
            zero, hx_new_.data<T>(), n_dim_);
        if (gen_eig) {
            // sx_new_ = sspace * h_red
            // gemm_op('N', 'N', n_dim_, n_max_, len_working_,
            //     one, sspace_.data<T>(), n_dim_, h_red_.data<T>(), len_space_,
            //     zero, sx_new_.data<T>(), n_dim_);
        }
        // --- 2.4 compute residuals & norms ---
#ifdef DEBUG_LOBPCG
std::cout << "--- main loop: residuals & norms ---" << std::endl;
#endif
        // residual_ = hx_new
        copy_op(n_dim_ * n_max_, hx_new_.data<T>(), 1, residual_.data<T>(), 1);
        // loop over eigenpairs
        for(int i = 0; i < n_max_; i++) {
            // if already converged, continue
            if(true == done_.data<int>()[i]) continue;
            // compute residual, residual <- Hx - eig x  | or | Hx - eig S x
            T *r_col = residual_.data<T>() + i * n_dim_;
            const Real lambda = eig_.data<Real>()[i];
            T alpha = T(-lambda);
            if(gen_eig){ // residual = Hx - eig Sx
                // Corrected: Remove inner loop that was accumulating all vectors
                const T *sspace_col = sspace_.data<T>()  + i * n_dim_;
                axpy_op(n_dim_, &alpha, sspace_col, 1, r_col, 1);
            } else { // residual = Hx - eig x
                // Corrected: Remove inner loop that was accumulating all vectors
                const T *space_col = space_.data<T>()  + i * n_dim_;
                axpy_op(n_dim_, &alpha, space_col, 1, r_col, 1);
            }
            // r_col, n_dim_ - elements vector
            r_norm_.data<Real>()[i] = nrm2_op(n_dim_, r_col, 1); // std::sqrt(static_cast<double>(n_dim_));
        }
        // --- 2.5 check convergence and locking ---
// !!!
// Maybe Use Trace to check!
#ifdef DEBUG_LOBPCG
std::cout << "--- main loop: check convergence and locking ---" << std::endl;
#endif
        // --- 2.5 check convergence and locking ---
#ifdef LOCKING_BY_TRACE
        // LOCKING STRATEGY BY TRACE MINIMIZATION
        for(int i = 0; i < n_max_; ++i){
            if (true == done_.data<int>()[i]) continue; // already locked
            // simply lock by norm of residuals
            if (iter > 0 && r_norm_.data<Real>()[i] < tolerance){
                done_.data<int>()[i] = 1;
            }
        }
        
        // checking overall convergence using subspace residual
        {
            ct::Tensor xax_tensor(t_type_, device_type_, {n_max_, n_max_});
            ct::Tensor sub_res_tensor(t_type_, device_type_, {n_dim_, n_max_});

            // xax = x^H * ax (Note: using 'C' for conjugate transpose)
            gemm_op('C', 'N', n_max_, n_max_, n_dim_,
                one, x_new_.data<T>(), n_dim_,
                hx_new_.data<T>(), n_dim_,
                zero, xax_tensor.data<T>(), n_max_);
            
            // sub_res = ax
            copy_op(n_dim_ * n_max_, hx_new_.data<T>(), 1, sub_res_tensor.data<T>(), 1);
            
            // sub_res -= x * xax
            gemm_op('N', 'N', n_dim_, n_max_, n_max_,
                neg_one, x_new_.data<T>(), n_dim_,
                xax_tensor.data<T>(), n_max_,
                one, sub_res_tensor.data<T>(), n_dim_);
            
            Real sub_res_norm = nrm2_op(n_dim_ * n_max_, sub_res_tensor.data<T>(), 1);
            Real xax_norm = nrm2_op(n_max_ * n_max_, xax_tensor.data<T>(), 1);
            
            if (xax_norm > 1e-20) {
                Real ratio = sub_res_norm / xax_norm;
                if(ratio < tolerance){
                     std::cout << "> subspace residual norm converged (" << ratio << ")" << std::endl;
                     setmem_int_op()(done_.data<int>(), 1, n_max_);
                }
            }
        }
#else
        // only lock the first converged eigenvalues/vectors
        for(int i = 0; i < n_max_; ++i){
            if (done_.data<int>()[i]) continue; // already locked
            if (iter > 0 && r_norm_.data<Real>()[i] < tolerance * std::sqrt(static_cast<double>(n_dim_))){
                // lock the vector
                done_.data<int>()[i] = 1;
            }
            if (!done_.data<int>()[i]) {
                // set done_(i:n_max_) to false
                for (int j = i; j < n_max_; ++j){
                    done_.data<int>()[j] = 0;
                }
                // ! need to break, or will be set by later eigenvectors
                break;
            }
        }
#endif
// --- check overall convergence ---
        bool all_converged = true;
        // only count n_band_ instead of n_max_
        for (int i = 0; i < n_band_; ++i) {
            // FIX: done_ is DT_INT, so must cast to int*, not bool*
            if (!done_.data<int>()[i]) {
                all_converged = false;
                break;
            }
        }
#ifdef DEBUG_CONV
        std::cout << "DEBUG: iter=" << iter << " all_converged=" << all_converged << std::endl;
        std::cout << "DEBUG: done_ = ";
        for(int i=0; i<n_max_; ++i) std::cout << done_.data<int>()[i] << " ";
        std::cout << std::endl;
        std::cout << "DEBUG: r_norm_ = ";
        for(int i=0; i<n_max_; ++i) std::cout << r_norm_.data<Real>()[i] << " ";
        std::cout << std::endl;
        // print the first few eigenvalues
        std::cout << "DEBUG: eig_ = ";
        for(int i=0; i<std::min(5, this->n_max_); ++i) std::cout << eig_.data<Real>()[i] << " ";
        std::cout << std::endl;
#endif
        if (all_converged){
            // copy x_new to input psi_in
            // syncmem_complex_2d_op()(this->evec_.data<T>(), this->n_dim_, psi_in, ld_psi_in, this->n_dim_, this->n_band_);
            syncmem_complex_2d_op()(psi_in, ld_psi_in, this->evec_.data<T>(), this->n_dim_, this->n_dim_, this->n_band_);
            // copy eig_ to input eigenvalue_in
            copy_real_op(n_band_, this->eig_.data<Real>(), 1, eigenvalue_in, 1);
// --- return ---
// converged
            std::cout << "Converged at iteration " << iter << " with RMS residual " << r_norm_.data<Real>()[0] << std::endl;
            return true;
        }
        // --- 2.6 check active eigenvalues and update blockvectors X, P, W ---
#ifdef DEBUG_LOBPCG
std::cout << "--- main loop: 2.6 check active eigenvalues and update blockvectors X, P, W ---" << std::endl;
#endif
        // 2.6.1 count active
        int n_conv = 0; // converged number
        for (int i = 0; i < n_max_; ++i) { if (done_.data<int>()[i]) ++n_conv; }
        n_active_ = n_max_ - n_conv;
        // [converged|active X|        P        |      W]
        //           | n_active_ | n_active_ | n_active_
        ind_x_ = n_max_ - n_active_; // [converged | active X]
        ind_p_ = ind_x_ + n_active_;
        ind_w_ = ind_p_ + n_active_;
#ifdef DEBUG_LOBPCG
std::cout << "n_conv = " << n_conv << ", n_active_ = "
            << n_active_ << std::endl;
// print ind
std::cout << "ind_x_ = " << ind_x_ << ", ind_p_ = " << ind_p_ << ", ind_w_ = " << ind_w_ << std::endl;
#endif

#ifdef DEBUG_LOBPCG
std::cout << "--- main loop: check active eigenvalues and update blockvectors X, P, W ---" << std::endl;
#endif
#ifdef DEBUG_LOBPCG
std::cout << "n_conv = " << n_conv << ", n_active_ = " << n_active_ << std::endl;
#endif
        /* 2.6.2 compute the new P and AP blockvector,
            by computing the coefficients u_p and then orthonalizing to u_x */
        /* [x p w]*/
        /* coefficients of p can be computed as follows:
            !!! notice that we only deal with active part of X, and thus W and P of size(n, n_active)
            0. we get coefficients from A_reduced(n_working_space, n_working_space)
            A_reduced [ c_x
                        c_p
                        c_w ]
            1. x^{k+1} = x^k * c_x + p^k * c_p + w^k * c_w
            2. p^k = x^{k+1} - x^k = x^k * (c_x - I) + w^k * c_w + p^k * c_p
            3. size of p: (n, n_active)
            4. we get active x^k here from x(:, n_max_subspace-n_active to n_max_subspace-1)
                corresponding coeff block(n_max_subspace-n_active, n_max_subspace-n_active,n_active,n_active)
        */

        u_x_ = std::move(ct::Tensor(t_type_, device_type_, {len_working_, n_max_}));
        u_p_ = std::move(ct::Tensor(t_type_, device_type_, {len_working_, n_active_})); // maximum size
        // ct::Tensor u_x_(t_type_, device_type_, {len_working_, n_max_});
        // ct::Tensor u_p_(t_type_, device_type_, {len_working_, n_active_});
// std::cout << "u_x_ = " << u_x_.data() << std::endl;
// std::cout << "u_p_ = " << u_p_.data() << std::endl;
        this->get_expansion_coeffs(len_space_, len_working_,n_max_, n_active_,
            h_red_.data<T>(), u_x_.data<T>(), u_p_.data<T>());

        // p: space block [u P w]
        // p = space * u_p
        // hp = hspace * u_p
        // sp = sspace * u_p
        gemm_op('N', 'N', n_dim_, n_active_, len_working_,
            one, space_.data<T>(), n_dim_, u_p_.data<T>(), len_working_, zero, evec_.data<T>(), n_dim_);
        copy_op(n_dim_*n_active_, evec_.data<T>(), 1, space_.data<T>() + ind_p_ * n_dim_, 1);
        gemm_op('N', 'N', n_dim_, n_active_, len_working_,
            one, hspace_.data<T>(), n_dim_, u_p_.data<T>(), len_working_, zero, evec_.data<T>(), n_dim_);
        copy_op(n_dim_*n_active_, evec_.data<T>(), 1, hspace_.data<T>() + ind_p_ * n_dim_, 1);
        if(gen_eig){
            gemm_op('N', 'N', n_dim_, n_active_, len_working_,
                one, sspace_.data<T>(), n_dim_, u_p_.data<T>(), len_working_, zero, evec_.data<T>(), n_dim_);
            copy_op(n_dim_*n_active_, evec_.data<T>(), 1, sspace_.data<T>() + ind_p_ * n_dim_, 1);
        }


#ifdef DEBUG_LOBPCG
        std::cout << "--- main loop: 2.7 update corresponding space from X, P, W ---" << std::endl;
#endif
        // --- 2.7 update corresponding space from X, P, W ---
        //
        // move x_new, hx_new, sx_new to space_, hspace_, sspace_
        copy_op(n_dim_*n_max_, x_new_.data<T>(), 1, space_.data<T>(), 1);
        copy_op(n_dim_*n_max_, hx_new_.data<T>(), 1, hspace_.data<T>(), 1);
        if(gen_eig){
            // copy_op(n_dim_*n_max_, sx_new_.data<T>(), 1, sspace_.data<T>(), 1);
        }

        // Compute residuals for active vectors
        // precondition_op
        // void operator()(const int& dim,
        //                 T* psi_iter,
        //                 const int& nbase,     // offset
        //                 const int& notconv,   // number of vectors
        //                 const Real* precondition,
        //                 const Real* eigenvalues);
        // space_(from ind_w_ column) = T * r
        copy_op(n_dim_ * n_active_,
                residual_.data<T>(), 1,
                space_.data<T>() + ind_w_ * n_dim_, 1);
        precondition_op(n_dim_, space_.data<T>(), ind_w_, n_active_, prec_.data<Real>(), eig_.data<Real>());

        // orthonormalize w against x and p, then orthonormalize w
        if (gen_eig){
            //
        }
        else{
            // [x p] - n_dim_ * (n_max_+n_active_)
            // [w] - n_dim_ * n_active_
            // Corrected argument order: normalize W (2nd ptr) against X+P (1st ptr)
            ortho_against_y(n_dim_, n_max_+n_active_, n_active_,
                space_.data<T>()+ind_w_ * n_dim_, n_dim_, space_.data<T>(), n_dim_);
        }

        //     this->compute_residuals(hx_active, sx_active, e_active, res_active, gen_eig);
        //     this->compute_residual_norms(res_active, r_norm_.slice({0, 0}, {2, n_active_}));

        //     // Check convergence
        //     int n_conv = this->check_convergence(r_norm_.slice({0, 0}, {2, n_active_}),
        //                                        done_.slice({0}, {n_active_}));

        //     if (n_conv == n_band_) {
        //         converged = true;
        //         break;
        //     }

        //     // Update active set
        //     this->n_active_ = n_band_ - n_conv;
        //     this->update_search_space_indices();

        //     // Apply preconditioner to residuals
        //     auto w_block = space_.slice({0, ind_w_}, {n_basis_, n_active_});
        //     Real shift = -e_red_.data<Real>()[0]; // Use first eigenvalue for shift
        //     this->apply_preconditioner(precnd_func, shift,
        //                              res_active.slice({0, n_conv}, {n_basis_, n_active_}),
        //                              w_block);

        //     // Orthogonalize W against X and P
        //     if (gen_eig) {
        //         auto sw_block = sspace_.slice({0, ind_w_}, {n_basis_, n_active_});
        //         spsi_func(w_block.data<T>(), sw_block.data<T>(), n_basis_, n_active_);
        //         this->s_ortho_against_y(w_block,
        //                               space_.slice({0, 0}, {n_basis_, ind_p_}),
        //                               sspace_.slice({0, 0}, {n_basis_, ind_p_}));
        //     } else {
        //         this->ortho_against_y(w_block, space_.slice({0, 0}, {n_basis_, ind_p_}));
        //     }

        //     // Perform Rayleigh-Ritz in expanded subspace
        //     int len_u = ind_p_ + n_active_; // Current subspace size: [X, P, W]
        //     this->rayleigh_ritz(len_u, hpsi_func, spsi_func, gen_eig);

        //     // Update current approximation
        //     this->update_approximation(len_u);

        //     // Lock converged vectors
        //     this->lock_converged_vectors();
// --- return---
// fail: too many
        if(iter >= max_iter-1){
            std::cerr << "LOBPCG did not converge within " << max_iter << " iterations." << std::endl;
            // Return best available results so far
            syncmem_complex_2d_op()(psi_in, ld_psi_in, this->space_.data<T>(), this->n_dim_, this->n_dim_, this->n_band_);
            copy_real_op(n_band_, this->eig_.data<Real>(), 1, eigenvalue_in, 1);
#ifdef DEBUG_LOBPCG            
            // print current best eigenvalue and residual for all bands
            std::cerr << "Current best eigenvalues and residuals:" << std::endl;
            for (int i = 0; i < n_band_; ++i) {
                std::cerr << "Band " << i << ": Eigenvalue = " << eig_.
                data<Real>()[i] << ", Residual = " << r_norm_.data<Real>()[i] << std::endl;
            }
#endif            
            return false;
        }
    } // end - main for loop

    // // Final Rayleigh-Ritz to get the best approximation
    // // this->rayleigh_ritz(n_max_, hpsi_func, spsi_func, gen_eig);
    // syncmem_var_d2h_op()(eigenvalue_in, e_red_.data<Real>(), n_band_);

    // // Copy final eigenvectors back to input
    // auto final_x = space_.slice({0, ind_x_}, {n_basis_, n_band_});
    // syncmem_complex_op()(psi_in, final_x.data<T>(), n_basis_ * n_band_);

    // return converged;
    return true;
}

// ==================== Core Algorithmic Functions ====================

template <typename T, typename Device>
void DiagoLOBPCG<T, Device>::check_init_guess(const int n, const int m, T *x, const int ldx)
{
    // check value of x[n, m]
    // if zero, generate random guess
    if (x == nullptr || ldx == 0) {
        // warn that no initial guess provided, and quit
        std::cerr << "No initial guess provided. Quitting." << std::endl;
        exit(1);
    }
#if defined(__CUDA) || defined(__ROCM)
    // if (this->device == base_device::GpuDevice)
    {
// Warning: GPU case?
        // do nothing
        // --- NOT implemented: This would need proper implementation for the norm calculation ---
        // --- for GPU ---
        // Maybe use Tensor to Device
        // to do on CPU simple and stupid!
    }
    // else
#endif
    {
        std::random_device rd;
        std::default_random_engine engine(rd());
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                // column major!
                if (x[i + j* ldx] == T(0.0)) {
                    // generate random guess(real values) for zero elements
                    x[i + j * ldx] = dist(engine);
                }
            }
        }
    }
    // if input x is not null, orthogonalize it.
    this->ortho(n, m, x, ldx);
#ifdef DEBUG_INIT
// verified!
    // // print x
    // std::cout << "Initial Guess:" << std::endl;
    // std::cout << "m = " << m << ", n = " << n << ", ldx = " << ldx << std::endl;
    // std::cout << "-------------------------" << std::endl;
    // for (int i = 0; i < n; ++i) {
    //     for (int j = 0; j < m; ++j) {
    //         std::cout << x[i + j * ldx] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << "-------------------------" << std::endl;
    // // output X^TX
    // // gemm overlap = x^T x
    // std::cout << "Overlap Matrix:" << std::endl;
    // std::cout << "-------------------------" << std::endl;
    // ct::Tensor overlap(t_type_, device_type_, {m, m});
    // overlap.zero();
    // gemm_op(
    //     'C', 'N', m, m, n, this->one, x, ldx, x, ldx, this->zero, overlap.data<T>(), m
    // );
    // check if overlap is identity
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            T *poverlap = overlap.data<T>() + i + j * m;
            if (std::abs(*poverlap) < 1e-15) *poverlap = T(0.0);
            // std::cout << *poverlap << " ";
            if (i == j) {
                if (std::abs(*poverlap - T(1.0)) > 1e-12) {
                    std::cerr << "Orthogonalization failed: diagonal element " << i << "," << j << " = " << *poverlap << " is not close to 1." << std::endl;
                }
            } else {
                if (std::abs(*poverlap) > 1e-12) {
                    std::cerr << "Orthogonalization failed: off-diagonal element " << i << "," << j << " = " << *poverlap << " is not close to 0." << std::endl;
                }
            }
        }
        std::cout << std::endl;
    }
    // std::cout << "-------------------------" << std::endl;
#endif
    // exit(0);

    // Generate random orthonormal vectors
    // Note: In practice, we would use a proper random number generator
    // For now, we'll create a simple basis
    // for (int i = 0; i < n_band_; ++i) {
    //     for (int j = 0; j < n_basis_; ++j) {
    //         x[i * ldx + j] = (i == j) ? T(1.0) : T(0.0);
    //     }
    // }

//     // Check if initial guess is zero or needs orthogonalization
//     T* evec_data = evec.data<T>();

//     // Compute norm of initial guess
//     Real norm = 0.0;
//     for (int i = 0; i < n_basis_ * n_band_; ++i) {
//         norm += std::norm(evec_data[i]);
//     }
//     norm = std::sqrt(norm);

//     // If zero or very small, generate random guess
//     if (norm < 1e-12) {
//         // Generate random orthonormal vectors
//         // Note: In practice, we would use a proper random number generator
//         // For now, we'll create a simple basis
//         for (int i = 0; i < n_band_; ++i) {
//             for (int j = 0; j < n_basis_; ++j) {
//                 evec_data[i * n_basis_ + j] = (i == j) ? T(1.0) : T(0.0);
//             }
//         }
//     }

//     // Orthogonalize the initial guess
//     // this->ortho_cholesky(evec);
}


template <typename T, typename Device>
void DiagoLOBPCG<T, Device>::rayleigh_ritz(
    T *space, T *hspace, const int len_space, const int dim, const int len_working,
    T *h_red, Real *e_red)
{
    /*
     * Solving a reduced standard eigenvalue problem on the space spanned by the S-ortho vectors.
     *
     * [IN] space, hspace -> [INNER] h_red, e_red -> [OUT] h_red overwritten by eigenvectors, e_red for eigenvalues
     *
     * DATA LAYOUT
     * INPUT space, aspace (dim, len_space) | use (dim * len_working) block
     * OUTPUT h_red, e_red to store eigenvectors / eigenvalues
     * h_red(len_space, len_space), use (len_working, len_working) block
     *
     * This function performs the following steps:
     * 1. Calculate the reduced matrix h_red = space^T * hspace, where hspace = H*space is given
     * 2. Perform the Rayleigh-Ritz procedure to find the eigenvalues and eigenvectors of h_red
     *
     * Returns eigenvectors in h_red, eigenvalues in e_red
     */

    // 1. first calculate h_red = space^T * hspace
#ifdef DEBUG_RR
    std::cout << "--- INNER Rayleigh-Ritz: calculate h_red = space^T * hspace ---" << std::endl;
#endif
    gemm_op('C', 'N', len_working, len_working, dim, one, space, dim, hspace, dim, zero, h_red, len_space);
    // now h_red is the reduced matrix
    //
    // 2. Perform the Rayleigh-Ritz procedure to find the eigenvalues and eigenvectors of h_red
    // use heevx for solving all len_working eigenpairs of h_red
    // void operator()(const int dim, const int lda,const T *Mat, const int neig, Real *eigen_val, T *eigen_vec);
    // h_red(len_space, len_space), ld = len_space, n = len_working, solve len_working eigenpairs
    // ct::kernels::lapack_heevx<T, ct_Device> heevx;
    // ct::kernels::lapack_heevx<T, ct_Device>()(nbase, nbase_x, hcc, nband, eigenvalue_gpu, vcc);
#ifdef DEBUG_RR
std::cout << "--- INNER Rayleigh-Ritz: heevx ---" << std::endl;
#endif
    heevx(len_working, len_space, h_red, len_working, e_red, h_red);
    // heevd(const int dim, T* Mat, const int lda, Real* eigen_val);
    // heevd(len_working, h_red, len_space, e_red);
    // now h_red is overwritten by eigenvectors, e_red for eigenvalues
}

// ==================== Ortho ====================

template <typename T, typename Device>
void DiagoLOBPCG<T, Device>::ortho(const int n, const int m, T *x, const int ldx)
{
    // ortho by QR
    ct::kernels::lapack_geqrf_inplace<T, ct_Device> qr;
    qr(n, m, x, ldx);
    // now x is Q, with orthogonal columns

    // ---
    // Orthonormalize the block vector u of shape (n, m) using Cholesky decomposition
    // Compute overlap = U^T * U
    // ct::Tensor overlap(t_type_, device_type_, {m, m});
    // gemm_op()(ctx_, 'C', 'N', m, m, n, one_,
    //           u, ldu, u, ldu,
    //           zero_, overlap.data<T>(), m);
    // Parallel_Reduce::reduce_pool(overlap.data<T>(), m * m);

    // // Cholesky decomposition: overlap = L * L^H
    // ct::kernels::lapack_potrf<T, ct_Device>()('L', m, overlap.data<T>(), m);

    // // Solve U = Q * L^H for Q (orthonormal)
    // ct::kernels::lapack_trtri<T, ct_Device>()('L', 'C', m, overlap.data<T>(), m);

}

template <typename T, typename Device>
void DiagoLOBPCG<T, Device>::ortho_against_y(const int n, const int m, const int k, T *x, const int ldx, const T *y, const int ldy)
{
#ifdef DEBUG_LOBPCG
    std::cout << "--- ortho_against_y: start ---" << std::endl;
#endif
    // !!! Assume input y is orthonormal!!!
    /*
     * y(n,m)
     * x(n,k)
     * ybx(m,k)
     */
    // Orthogonalizes block vector `x` against a given \b orthonormal set `y`
    // and orthonormalizes `x`. If `y` is not orthonormal, extra computation
    // is needed inside this function.
    //
    // `y(n, m)` is a given block vector, assumed to be orthonormal
    // `x(n, k)` is the block vector to be orthogonalized
    // first against y and then internally.
    //
    // if y is not orthonormal, we will do by solving (Y'Y)y_coeff = Y'X
    // X = X - Y(Y'Y)^{-1}Y'X = X - Y * y_coeff
    //
    // else if y is orthonormal, we can directly orthogonalize x against y
    // X = X - Y * Y'X

    Real tol_ortho = 1.0e-10; // 1.0e-13;

    // bool assume_y_orthonormal = true;

    // First check if input y is orthonormal
    bool is_y_orthonormal = true;

    // Compute Y'Y
#ifdef DEBUG_ORTHO_Y
// std::cout << "--- ortho_against_y: computing yby ---" << std::endl;
#endif
    // y(n, m), yby(m, m)
    ct::Tensor yby(t_type_, device_type_, {m, m});
#ifdef DEBUG_ORTHO_Y
// std::cout << "m = " << m << ", n = " << n << std::endl;
// std::cout << "ldy = " << ldy << ", ldx = " << ldx << std::endl;
// std::cout << "y = " << y << std::endl;
// std::cout << "yby = " << yby.data<T>() << std::endl;

// std::cout << "Before gemm_op: x=" << x << ", y=" << y
// // << ", y_coeff=" << y_coeff.data<T>()
// << std::endl;
#endif

// std::cout << "n=" << n << ", m=" << m << ", k=" << k << ", ldx=" << ldx << ", ldy=" << ldy << std::endl;
    gemm_op('C', 'N', m, m, n, one, y, ldy, y, ldy, zero, yby.data<T>(), m);

    // Check if Y'Y is identity
    // For now, assume y is orthonormal to avoid complexity
    // In practice, we would check the diagonal and off-diagonal norms
    // For this implementation, we'll proceed with the assumption that y is orthonormal

    // Start with initial orthogonalization of x
#ifdef DEBUG_ORTHO_Y
// std::cout << "--- ortho_against_y: initial ortho ---" << std::endl;
#endif
    this->ortho(n, k, x, ldx);

    // Temporary storage for coefficients and overlaps
    // ybx(m, k) = y'(m, n) * x(n, k)
    ct::Tensor ybx(t_type_, device_type_, {m, k});
    ct::Tensor y_coeff(t_type_, device_type_, {m, k});
    
    // We need overlap tensor to check convergence
    ct::Tensor overlap(t_type_, device_type_, {m, k});

    // Do ortho while overlap norm > tol_ortho
    Real norm_overlap = 10.0; // big init value
    const int ITER_MAX = 10;
    int iter_cnt = ITER_MAX;
#ifdef DEBUG_ORTHO_Y
std::cout << "--- ortho_against_y: ortho loop ---" << std::endl;
#endif
    while (norm_overlap >= tol_ortho && iter_cnt > 0) {
        // Compute Y'X
        // yb(m, k) = y'(m, n) * x(n, k)
        gemm_op('C', 'N', m, k, n, one, y, ldy, x, ldx, zero, ybx.data<T>(), m);

        if (!is_y_orthonormal) {
            // If y is not orthonormal, solve Y'Y y_coeff = Y'X
            // For now, we'll use the simple case where y is orthonormal
            // In practice, we would use a solver here
            // syncmem_complex_op()(y_coeff.data<T>(), ybx.data<T>(), m * k);
        } else {
            // y_coeff = Y'X directly when y is orthonormal
            syncmem_complex_op()(y_coeff.data<T>(), ybx.data<T>(), m * k);
        }
#ifdef DEBUG_ORTHO_Y
std::cout << "--- ortho_against_y: X - Y * y_coeff ---" << std::endl;
#endif
        // X = X - Y * y_coeff
        // Copy x to temp_x first
        // syncmem_complex_op()(temp_x.data<T>(), x, n * k);
        gemm_op('N', 'N', n, k, m, neg_one, y, ldy, y_coeff.data<T>(), m, one, x, ldx);

        // Check for linear dependence / collapse
#if !defined(__CUDA) && !defined(__ROCM)
        Real x_norm_check = 0.0;
        // Crude L1 norm check to see if vector is numerically zero
        for(int i=0; i<n*k; ++i) x_norm_check += std::abs(x[i]);
        
        if (x_norm_check < 1.0e-12) {
#ifdef DEBUG_ORTHO_Y
            std::cout << "--- ortho_against_y: vector collapsed (norm=" << x_norm_check << "), randomizing ---" << std::endl;
#endif
            std::random_device rd;
            std::default_random_engine engine(rd());
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            // Re-populate with random values
            for(int i=0; i<n*k; ++i) {
                x[i] = static_cast<T>(dist(engine));
            }
        }
#endif

#ifdef DEBUG_ORTHO_Y
std::cout << "--- ortho_against_y: loop ortho x ---" << std::endl;
#endif
        // Orthonormalize x
        this->ortho(n, k, x, ldx);

        // Compute overlap = Y^T X after orthonormalization
        gemm_op('C', 'N', m, k, n, one, y, ldy, x, ldx, zero, overlap.data<T>(), m);

        // Compute norm of overlap (Frobenius norm)
        norm_overlap = 0.0;
#if defined(__CUDA) || defined(__ROCM)
        // GPU implementation needed
        // For now, just break loop to avoid infinite loops if not implemented
        norm_overlap = 0.0; 
#else
        // CPU implementation
        T* overlap_data = overlap.data<T>();
        for(int i=0; i < m*k; ++i) {
            norm_overlap += std::norm(overlap_data[i]); // std::norm returns |z|^2
        }
        norm_overlap = std::sqrt(norm_overlap);
#endif
        --iter_cnt;
    }

    if (iter_cnt <= 0) {
        // Too many ortho iterations, exit with warning
        std::cerr << "Too many iterations in ortho_against_y. Failed to reach tolerance." << std::endl;
    }
#ifdef DEBUG_LOBPCG
std ::cout << "--- ortho_against_y: end ---" << std::endl;
#endif
}

template <typename T, typename Device>
void DiagoLOBPCG<T, Device>::get_expansion_coeffs(const int len_space, const int len_working,
    const int n_max, const int n_active,
    T *h_red, T *u_x, T *u_p)
{
#ifdef DEBUG_LOBPCG
std::cout << "--- get_expansion_coeffs: start ---" << std::endl;
// std::cout << "u_x = " << u_x << std::endl;
// std::cout << "u_p = " << u_p << std::endl;
#endif
    // h_red[len_space, len_space]
    // u_x[len_working, n_max]
    // u_p[len_working, n_active]
    // now only CPU version
#if defined(__CUDA) || defined(__ROCM)
    // if (this->device == base_device::GpuDevice)
    {
// Warning: GPU case not implemented
        // do nothing
        // --- NOT implemented: This would need proper implementation for the norm calculation ---
    }
    // else
#else
    // CPU here
    //
    int offset_x = n_max - n_active;
    // u_x(:len_working,:n_max) = h_red(:len_working,:n_max)
    const int ld_h_red = len_space;
    const int ld_u_x = len_working;
    syncmem_complex_2d_op()(u_x, ld_u_x, h_red, ld_h_red, len_working, n_max);
    // for(int j = 0; j < len_working; ++j) {
    //     for(int i = 0; i < n_max; ++i) {
    //         u_x[j + i*ld_u_x] = h_red[j + i*ld_h_red];
    //     }
    // }
    //

    // u_p = u_x(:,offset_x:n_max)
    const int ld_u_p = len_working;
    syncmem_complex_2d_op()(u_p, ld_u_p, u_x + offset_x * ld_u_x, ld_u_x, len_working, n_active);
    // for(int j = 0; j < len_working; ++j) {
    //     for(int i = 0; i < n_active; ++i) {
    //         u_p[j*ld_u_p + i] = u_x[j*ld_u_x + offset_x + i];
    //     }
    // }
    //
    // u_p -= Identity
    for(int i = 0; i < n_active; ++i) {
        // u_p(offset_x + i, i) -= 1
        u_p[(offset_x + i) + i*ld_u_p] -= T(1.0);
    }
#ifdef DEBUG_LOBPCG
std::cout << "--- get_expansion_coeffs: ortho ---" << std::endl;
// std::cout << "u_x = " << u_x << std::endl;
// std::cout << "u_p = " << u_p << std::endl;
#endif
    ortho_against_y(len_working, n_max, n_active, u_p, ld_u_p, u_x, ld_u_x);
#ifdef DEBUG_LOBPCG
std::cout << "--- get_expansion_coeffs: end ---" << std::endl;
#endif

#endif
}


// Explicit template instantiation
template class DiagoLOBPCG<std::complex<float>, base_device::DEVICE_CPU>;
template class DiagoLOBPCG<std::complex<double>, base_device::DEVICE_CPU>;
#if ((defined __CUDA) || (defined __ROCM))
template class DiagoLOBPCG<std::complex<float>, base_device::DEVICE_GPU>;
template class DiagoLOBPCG<std::complex<double>, base_device::DEVICE_GPU>;
#endif

#ifdef __LCAO
template class DiagoLOBPCG<double, base_device::DEVICE_CPU>;

#if ((defined __CUDA) || (defined __ROCM))
template class DiagoLOBPCG<double, base_device::DEVICE_GPU>;
#endif

#endif

} // namespace hsolver
