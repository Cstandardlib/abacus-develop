#include "source_hsolver/diago_lobpcg.h"

#include <random> // make random initial guess
#include <cmath>  // std::sqrt
#include <fstream>
#include <iostream>
#include <iomanip> 

#include <source_base/kernels/math_kernel_op.h>
#include <source_base/timer.h>
// #include <source_base/global_function.h>
#include <source_base/module_container/ATen/core/tensor.h>
#include <source_base/module_container/ATen/core/tensor_types.h>
#include <source_base/module_container/ATen/core/tensor_map.h>

#include <source_hsolver/kernels/bpcg_kernel_op.h> // normalize_op, precondition_op, apply_eigenvalues_op
#ifdef __MPI
#include <mpi.h>
#endif

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

    // ----- Preconditioner (depends only on n_dim_, allocated once) -----
    // Store reference to preconditioner data (host side); only a reference, does
    // not own memory.
    this->h_prec_ = std::move(ct::TensorMap((void*)precondition, r_type_, device_type_, {n_dim_}));
    // Preconditioner, device side
    this->prec_ = std::move(ct::Tensor(r_type_, device_type_, {n_dim_}));
    this->prec_.zero();

    // ----- Allocate the n_max_ / len_space_ dependent workspace -----
    // Kept in a re-callable helper: diag() may clamp n_max_ against the global
    // problem dimension and reallocate.
    this->allocate_workspace();
}

template <typename T, typename Device>
void DiagoLOBPCG<T, Device>::allocate_workspace()
{
    // All tensors whose shape depends on n_max_ / len_space_. Re-callable so that
    // diag() can resize after clamping n_max_ to the global problem dimension.
    // Eigenvalues [n_max_] (Real) and eigenvectors [n_dim_, n_max_]
    this->eig_ = std::move(ct::Tensor(r_type_, device_type_, {n_max_}));
    this->evec_ = std::move(ct::Tensor(t_type_, device_type_, {n_dim_, n_max_}));

    // 3 spaces: [X, P, W], H * [X, P, W], S * [X, P, W] (if generalized)
    this->space_ = std::move(ct::Tensor(t_type_, device_type_, {n_dim_, len_space_}));
    this->hspace_ = std::move(ct::Tensor(t_type_, device_type_, {n_dim_, len_space_}));
    this->sspace_ = std::move(ct::Tensor(t_type_, device_type_, {n_dim_, len_space_}));
    // Reduced problem
    this->h_red_ = std::move(ct::Tensor(t_type_, device_type_, {len_space_, len_space_}));
    this->e_red_ = std::move(ct::Tensor(r_type_, device_type_, {len_space_}));

    // Temporary storage for updated Ritz vectors and their products
    this->x_new_ = std::move(ct::Tensor(t_type_, device_type_, {n_dim_, n_max_}));
    this->hx_new_ = std::move(ct::Tensor(t_type_, device_type_, {n_dim_, n_max_}));
    this->sx_new_ = std::move(ct::Tensor(t_type_, device_type_, {n_dim_, n_max_}));

    // Residuals (R = H*X - lambda*S*X) and convergence bookkeeping
    this->residual_ = std::move(ct::Tensor(t_type_, device_type_, {n_dim_, n_max_}));
    this->r_norm_ = std::move(ct::Tensor(r_type_, device_type_, {n_max_})); // 2-norm of vectors, Real
    this->done_ = std::move(ct::Tensor(ct::DataType::DT_INT, device_type_, {n_max_})); // bool as int

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
void DiagoLOBPCG<T, Device>::set_diag_comm(const diag_comm_info& comm_info)
{
    this->comm_rank_ = comm_info.rank;
    this->comm_nproc_ = comm_info.nproc;
#ifdef __MPI
    this->comm_ = comm_info.comm;
#endif
}

template <typename T, typename Device>
void DiagoLOBPCG<T, Device>::allreduce_sum_inplace(T* data, const int count)
{
#ifdef __MPI
    if (this->comm_nproc_ <= 1 || data == nullptr || count <= 0) {
        return;
    }
    if (std::is_same<T, std::complex<float>>::value) {
        MPI_Allreduce(MPI_IN_PLACE, data, count, MPI_C_FLOAT_COMPLEX, MPI_SUM, this->comm_);
    } else if (std::is_same<T, std::complex<double>>::value) {
        MPI_Allreduce(MPI_IN_PLACE, data, count, MPI_DOUBLE_COMPLEX, MPI_SUM, this->comm_);
    } else if (std::is_same<T, float>::value) {
        MPI_Allreduce(MPI_IN_PLACE, data, count, MPI_FLOAT, MPI_SUM, this->comm_);
    } else {
        MPI_Allreduce(MPI_IN_PLACE, data, count, MPI_DOUBLE, MPI_SUM, this->comm_);
    }
#else
    (void)data;
    (void)count;
#endif
}

template <typename T, typename Device>
void DiagoLOBPCG<T, Device>::allreduce_sum_inplace_real(Real* data, const int count)
{
#ifdef __MPI
    if (this->comm_nproc_ <= 1 || data == nullptr || count <= 0) {
        return;
    }
    if (std::is_same<Real, float>::value) {
        MPI_Allreduce(MPI_IN_PLACE, data, count, MPI_FLOAT, MPI_SUM, this->comm_);
    } else {
        MPI_Allreduce(MPI_IN_PLACE, data, count, MPI_DOUBLE, MPI_SUM, this->comm_);
    }
#else
    (void)data;
    (void)count;
#endif
}

template <typename T, typename Device>
void DiagoLOBPCG<T, Device>::bcast_inplace(T* data, const int count)
{
#ifdef __MPI
    if (this->comm_nproc_ <= 1 || data == nullptr || count <= 0) {
        return;
    }
    if (std::is_same<T, std::complex<float>>::value) {
        MPI_Bcast(data, count, MPI_C_FLOAT_COMPLEX, 0, this->comm_);
    } else if (std::is_same<T, std::complex<double>>::value) {
        MPI_Bcast(data, count, MPI_DOUBLE_COMPLEX, 0, this->comm_);
    } else if (std::is_same<T, float>::value) {
        MPI_Bcast(data, count, MPI_FLOAT, 0, this->comm_);
    } else {
        MPI_Bcast(data, count, MPI_DOUBLE, 0, this->comm_);
    }
#else
    (void)data;
    (void)count;
#endif
}

template <typename T, typename Device>
void DiagoLOBPCG<T, Device>::bcast_inplace_real(Real* data, const int count)
{
#ifdef __MPI
    if (this->comm_nproc_ <= 1 || data == nullptr || count <= 0) {
        return;
    }
    if (std::is_same<Real, float>::value) {
        MPI_Bcast(data, count, MPI_FLOAT, 0, this->comm_);
    } else {
        MPI_Bcast(data, count, MPI_DOUBLE, 0, this->comm_);
    }
#else
    (void)data;
    (void)count;
#endif
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
    const int max_iter,
    const std::vector<double>& ethr_band)
{
    ModuleBase::timer::tick("Diago_LOBPCG", "diag");
    ModuleBase::timer::tick("Diago_LOBPCG", "init");

    // Locking strategy switch (Knyazev 2004). Default soft locking; opt into
    // classical hard locking for A/B comparison via env LOBPCG_HARD_LOCK=1.
    if (const char* e = std::getenv("LOBPCG_HARD_LOCK")) { this->hard_lock_ = (std::atoi(e) != 0); }
    // Per-iteration convergence trace (rank 0): "CONVTRACE iter nconv nband | r_norm[0..nband-1]".
    // A new iter==0 line marks a new diagonalization call. Default off; numerically inert.
    const bool conv_trace = (std::getenv("LOBPCG_CONV_TRACE") || std::getenv("DIAG_CONV_TRACE")) && (this->comm_rank_ == 0);

    // Clamp the subspace to the GLOBAL problem dimension. The search space [X,P,W]
    // spans 3 * n_max columns and must fit within the global dimension, otherwise
    // it cannot be orthonormalized to full rank and Rayleigh-Ritz becomes rank
    // deficient (e.g. the 20x20 readH unit test). n_dim_ is the LOCAL row count, so
    // the global dimension is the MPI sum across the diag communicator -- using the
    // local n_dim_ would wrongly trigger on highly parallel runs where the local
    // block is small but the global problem is large. set_diag_comm() has already
    // run, so the communicator is available here. Production PW runs have
    // global_dim >> 3 * n_max, so this clamp only fires for tiny systems.
    int global_dim = this->n_dim_;
#ifdef __MPI
    if (this->comm_nproc_ > 1) {
        MPI_Allreduce(MPI_IN_PLACE, &global_dim, 1, MPI_INT, MPI_SUM, this->comm_);
    }
#endif
    if (3 * this->n_max_ > global_dim) {
        this->n_max_ = std::max(this->n_band_, global_dim / 3);
        this->len_space_ = 3 * this->n_max_;
        this->ind_x_ = 0;
        this->ind_p_ = this->n_max_;
        this->ind_w_ = 2 * this->n_max_;
        this->n_active_ = this->n_max_;
        this->allocate_workspace();
    }

#ifdef LOCKING_BY_TRACE
std::cout << "Using locking by trace." << std::endl;
#endif
// #ifdef DEBUG_LOBPCG
#ifdef DEBUG_SCF
std::cout << "----- START LOBPCG -----" << std::endl;
// std::cout << "precnd 1.0" << std::endl;
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

        // // for (int i = 0; i < this->n_dim_; ++i) {
        // //     for (int j = 0; j < this->n_dim_; ++j) {
        // //         // std::cout << "H[" << i << "," << j << "] = " << std::real(test_out[i + j * this->n_dim_]) << std::endl;
        // //         std::cout << std::real(test_out[i + j * this->n_dim_]) << " ";
        // //     }
        // //     std::cout << std::endl;
        // // }
        // // not cout but save to a file Si2.txt for debugging
        // // auto mat = "H_matrix_Si2.txt";
        // // auto mat = "H_matrix_Si2_complex.txt";
        // // auto mat = "H_matrix_Si2_complex_cpp.txt";
        // auto mat = "H_matrix_Si2_complex_cpp_nmax50.txt";
        // std::ofstream outfile(mat);
        // outfile << std::setprecision(16);  // double 最多约 15~17 位有效数字，16 是安全选择
        // if (outfile.is_open()) {
        //     for (int i = 0; i < this->n_dim_; ++i) {
        //         for (int j = 0; j < this->n_dim_; ++j) {
        //             outfile << std::real(test_out[i + j * this->n_dim_]) << " ";
        //             outfile << test_out[i + j * this->n_dim_] << " ";
        //             // output with a+bi format
        //             // std::complex<double> val = test_out[i + j * this->n_dim_];
        //             // // outfile << val.real() << "+" << val.imag() << "i ";
        //             // outfile << val.real();
        //             // if (val.imag() >= 0.0) {
        //             //     outfile << "+" << val.imag() << "i ";
        //             // } else { // 负数自带减号，直接输出即可：如 1-2i
        //             //     outfile << val.imag() << "i ";
        //             // }
        //         }
        //         outfile << std::endl;
        //     }
        //     outfile.close();
        //     std::cout << "H matrix saved to " << mat << std::endl;
        // } else {
        //     std::cerr << "Unable to open file to save H matrix." << std::endl;
        // }
        // delete[] test_in;
        // delete[] test_out;
// return false; // for now just test the hpsi_func
// exit(0);
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
    ModuleBase::timer::tick("Diago_LOBPCG", "init");

// --- 1. first iter --- explicit do the fist Rayleigh-Ritz for X'HX
#ifdef DEBUG_LOBPCG
std::cout << "--- Debug: Entering first iteration ---" << std::endl;
#endif
    ModuleBase::timer::tick("Diago_LOBPCG", "first_iter");

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
    ModuleBase::timer::tick("Diago_LOBPCG", "first_iter_rr");
    len_working_ = n_max_; // first round, no p no w
    this->rayleigh_ritz(space_.data<T>(), hspace_.data<T>(), len_space_, n_dim_, len_working_,
        h_red_.data<T>(), e_red_.data<Real>());    // now (u, lambda) = (h_red, e_red)
    ModuleBase::timer::tick("Diago_LOBPCG", "first_iter_rr");

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
    // Keep X globally orthonormal before building residuals in distributed runs.
    this->ortho(n_dim_, n_max_, space_.data<T>(), n_dim_);
    hpsi_func(space_.data<T>(), hspace_.data<T>(), n_dim_, n_max_);
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
    ModuleBase::timer::tick("Diago_LOBPCG", "first_iter");
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
    Real global_dim_real = static_cast<Real>(n_dim_);
    this->allreduce_sum_inplace_real(&global_dim_real, 1);
    if (global_dim_real < static_cast<Real>(1.0)) {
        global_dim_real = static_cast<Real>(1.0);
    }
    const Real inv_sqrt_global_dim = static_cast<Real>(1.0) / std::sqrt(global_dim_real);
    // Initialize convergence flags
#ifdef DEBUG_LOBPCG
    // return false; // Removed early exit
    std::cout << "--- main loop: start ---" << std::endl;
    std::cout << std::flush;
#endif
    setmem_int_op()(done_.data<int>(), false, n_max_);

    // Convergence is judged by the eigenvalue change |lambda_i(iter) - lambda_i(iter-1)| <
    // tolerance, per band, matching dav_subspace/cg. The eigenvector residual converges much
    // more slowly than the eigenvalue, so the previous residual-norm test made LOBPCG run
    // roughly twice the inner iterations of Davidson with no SCF-accuracy benefit. eig_prev
    // holds the previous iteration's Ritz values, seeded from the initial Rayleigh-Ritz.
    std::vector<Real> eig_prev(n_max_);
    for (int i = 0; i < n_max_; ++i) { eig_prev[i] = eig_.data<Real>()[i]; }

    // Per-band |dlambda| thresholds (Tier 1 experiment). ethr_band (from
    // cal_smooth_ethr, looser for unoccupied bands) replaces the scalar tolerance
    // per band; buffer bands and the empty-vector legacy path fall back to scalar.
    std::vector<Real> tol_band(n_max_, static_cast<Real>(tolerance));
    if (!ethr_band.empty()) {
        for (int i = 0; i < n_band_ && i < static_cast<int>(ethr_band.size()); ++i) {
            tol_band[i] = static_cast<Real>(ethr_band[i]);
        }
    }

    for (int iter = 0; iter < max_iter; ++iter) {
        ModuleBase::timer::tick("Diago_LOBPCG", "main_iter");
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
    ModuleBase::timer::tick("Diago_LOBPCG", "iter_hpsi");
        hpsi_func(space_.data<T>() + n_dim_ * ind_w_, hspace_.data<T>() + n_dim_ * ind_w_, n_dim_, n_active_);
    ModuleBase::timer::tick("Diago_LOBPCG", "iter_hpsi");
        // --- 2.2 construct the reduced matrix and diagonalization ---
#ifdef DEBUG_LOBPCG
std::cout << "----- end hpsi  -----" << std::endl;
#endif    
        len_working_ = n_max_ + 2 * n_active_; // X P W
        if(0 == iter) { // first round, no P is constructed yet
            len_working_ = 2 * n_max_;
        }
        // --- locking mode (soft default / hard via env). ---
        // Soft: RR spans the whole [locked X | active X | P | W]; all n_max Ritz
        //   vectors recomputed (locked X improved by RR).
        // Hard: deflate the n_conv locked X out of the RR; iterate only the active
        //   contiguous block [active X | P | W] at offset ind_x_, freezing locked X.
        // ind_x_ == n_conv here. At iter 0 (n_conv==0) both modes coincide.
        const int rr_off_  = this->hard_lock_ ? ind_x_ : 0;          // RR subspace column offset
        const int rr_dim_  = len_working_ - rr_off_;                  // soft: len_working_; hard: 3*n_active (2*n_max @iter0)
        const int upd_off_ = this->hard_lock_ ? ind_x_ : 0;          // first Ritz column updated this iter
        const int upd_n_   = this->hard_lock_ ? n_active_ : n_max_;   // number of Ritz columns updated
        const int x_in_sub_= this->hard_lock_ ? n_active_ : n_max_;   // X columns inside the RR subspace
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
    ModuleBase::timer::tick("Diago_LOBPCG", "iter_rr");
        this->rayleigh_ritz(space_.data<T>() + rr_off_ * n_dim_, hspace_.data<T>() + rr_off_ * n_dim_,
                len_space_, n_dim_, rr_dim_,
                h_red_.data<T>(), e_red_.data<Real>());    // now (u, lambda) = (h_red, e_red)
    ModuleBase::timer::tick("Diago_LOBPCG", "iter_rr");

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
        // eig_(active) = e_red_(active); locked eigenvalues kept frozen under hard locking.
        copy_real_op(upd_n_, e_red_.data<Real>(), 1, eig_.data<Real>() + upd_off_, 1);

        // --- 2.3 update X, AX and, if required BX ---
#ifdef DEBUG_LOBPCG
std::cout << "--- main loop: update X, AX and, if required BX ---" << std::endl;
#endif
    ModuleBase::timer::tick("Diago_LOBPCG", "iter_update_x");
        // x_new_(active) = space(active subspace) * h_red ; locked X frozen under hard locking
        gemm_op('N', 'N', n_dim_, upd_n_, rr_dim_,
            one, space_.data<T>() + rr_off_ * n_dim_, n_dim_, h_red_.data<T>(), len_space_,
            zero, x_new_.data<T>() + upd_off_ * n_dim_, n_dim_);
        // hx_new_(active) = hspace(active subspace) * h_red
        gemm_op('N', 'N', n_dim_, upd_n_, rr_dim_,
            one, hspace_.data<T>() + rr_off_ * n_dim_, n_dim_, h_red_.data<T>(), len_space_,
            zero, hx_new_.data<T>() + upd_off_ * n_dim_, n_dim_);
        if (gen_eig) {
            // sx_new_ = sspace * h_red
            // gemm_op('N', 'N', n_dim_, n_max_, len_working_,
            //     one, sspace_.data<T>(), n_dim_, h_red_.data<T>(), len_space_,
            //     zero, sx_new_.data<T>(), n_dim_);
        }
        // hx_new_ is already H * x_new by linearity when the Ritz vectors stay
        // orthonormal. Fall back to the old refresh path if that check fails.
        const bool need_refresh_hx = gen_eig
            || !this->is_orthonormal(n_dim_, upd_n_, x_new_.data<T>() + upd_off_ * n_dim_, n_dim_, static_cast<Real>(1.0e-8));
        if (need_refresh_hx) {
            this->ortho(n_dim_, upd_n_, x_new_.data<T>() + upd_off_ * n_dim_, n_dim_);
            hpsi_func(x_new_.data<T>() + upd_off_ * n_dim_, hx_new_.data<T>() + upd_off_ * n_dim_, n_dim_, upd_n_);
        }
        ModuleBase::timer::tick("Diago_LOBPCG", "iter_update_x");
        // --- 2.4 compute residuals & norms ---
#ifdef DEBUG_LOBPCG
std::cout << "--- main loop: residuals & norms ---" << std::endl;
#endif
    ModuleBase::timer::tick("Diago_LOBPCG", "iter_residual");
        // residual_ = hx_new (active block only; locked residuals stay frozen under hard locking)
        copy_op(n_dim_ * upd_n_, hx_new_.data<T>() + upd_off_ * n_dim_, 1, residual_.data<T>() + upd_off_ * n_dim_, 1);
        // loop over updated eigenpairs [upd_off_, upd_off_ + upd_n_)
        for(int i = upd_off_; i < upd_off_ + upd_n_; i++) {
            // compute residual, residual <- Hx - eig x  | or | Hx - eig S x
            T *r_col = residual_.data<T>() + i * n_dim_;
            const Real lambda = eig_.data<Real>()[i];
            T alpha = T(-lambda);
            if(gen_eig){ // residual = Hx - eig Sx
                // residual must be based on current Ritz vectors x_new_/sx_new_
                const T *sx_col = sx_new_.data<T>()  + i * n_dim_;
                axpy_op(n_dim_, &alpha, sx_col, 1, r_col, 1);
            } else { // residual = Hx - eig x
                // residual must be based on current Ritz vectors x_new_
                const T *x_col = x_new_.data<T>()  + i * n_dim_;
                axpy_op(n_dim_, &alpha, x_col, 1, r_col, 1);
            }
            // r_col, n_dim_ - elements vector
            const Real local_norm = nrm2_op(n_dim_, r_col, 1);
            r_norm_.data<Real>()[i] = local_norm * local_norm;
        }
        this->allreduce_sum_inplace_real(r_norm_.data<Real>() + upd_off_, upd_n_);
        for (int i = upd_off_; i < upd_off_ + upd_n_; ++i) {
            r_norm_.data<Real>()[i] = std::sqrt(r_norm_.data<Real>()[i]) * inv_sqrt_global_dim;
        }
        ModuleBase::timer::tick("Diago_LOBPCG", "iter_residual");
        // --- 2.5 check convergence and locking ---
// !!!
// Maybe Use Trace to check!
#ifdef DEBUG_LOBPCG
std::cout << "--- main loop: check convergence and locking ---" << std::endl;
#endif
    ModuleBase::timer::tick("Diago_LOBPCG", "iter_lock");
        // Per-band convergence flag: eigenvalue-change criterion (dav_subspace/cg style).
        // tolerance = diag_thr * lobpcg_tol_scale is reused as the |dlambda| threshold.
        // Original residual-norm decision kept for reference:
        //     band_ok[i] = (r_norm_.data<Real>()[i] < tolerance) ? 1 : 0;
        std::vector<int> band_ok(n_max_, 0);
        for (int i = 0; i < n_max_; ++i) {
            band_ok[i] = (std::abs(eig_.data<Real>()[i] - eig_prev[i]) < tol_band[i]) ? 1 : 0;
        }
        // Snapshot current eigenvalues for the next iteration's delta.
        for (int i = 0; i < n_max_; ++i) { eig_prev[i] = eig_.data<Real>()[i]; }
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
            this->allreduce_sum_inplace(xax_tensor.data<T>(), n_max_ * n_max_);
            
            // sub_res = ax
            copy_op(n_dim_ * n_max_, hx_new_.data<T>(), 1, sub_res_tensor.data<T>(), 1);
            
            // sub_res -= x * xax
            gemm_op('N', 'N', n_dim_, n_max_, n_max_,
                neg_one, x_new_.data<T>(), n_dim_,
                xax_tensor.data<T>(), n_max_,
                one, sub_res_tensor.data<T>(), n_dim_);
            
            Real sub_res_norm = nrm2_op(n_dim_ * n_max_, sub_res_tensor.data<T>(), 1);
            Real xax_norm = nrm2_op(n_max_ * n_max_, xax_tensor.data<T>(), 1);
            sub_res_norm *= sub_res_norm;
            xax_norm *= xax_norm;
            this->allreduce_sum_inplace_real(&sub_res_norm, 1);
            this->allreduce_sum_inplace_real(&xax_norm, 1);
            sub_res_norm = std::sqrt(sub_res_norm);
            xax_norm = std::sqrt(xax_norm);
            
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
            if (iter > 0 && band_ok[i]){
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
        // --- per-iteration convergence trace (rank 0, env-gated, numerically inert) ---
        if (conv_trace) {
            int nconv_tr = 0;
            for (int i = 0; i < n_band_; ++i) { if (done_.data<int>()[i]) ++nconv_tr; }
            std::cout << "CONVTRACE " << iter << " " << nconv_tr << " " << n_band_ << " |";
            for (int i = 0; i < n_band_; ++i) { std::cout << " " << r_norm_.data<Real>()[i]; }
            std::cout << " |";   // eigenvalues, so |dlambda| (the locking criterion) can be plotted
            for (int i = 0; i < n_band_; ++i) { std::cout << " " << eig_.data<Real>()[i]; }
            std::cout << std::endl;
        }
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
        // Guard against sticky-lock false positives: require current bands to satisfy the criterion.
        if (all_converged) {
            for (int i = 0; i < n_band_; ++i) {
                if (!band_ok[i]) {
                    all_converged = false;
                    break;
                }
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
            syncmem_complex_2d_op()(psi_in, ld_psi_in, this->x_new_.data<T>(), this->n_dim_, this->n_dim_, this->n_band_);
            // copy eig_ to input eigenvalue_in
            copy_real_op(n_band_, this->eig_.data<Real>(), 1, eigenvalue_in, 1);
            ModuleBase::timer::tick("Diago_LOBPCG", "iter_lock");
            ModuleBase::timer::tick("Diago_LOBPCG", "main_iter");
// --- return ---
// converged
            if (this->comm_rank_ == 0) {
                std::cout << "Converged at iteration " << iter << " with RMS residual " << r_norm_.data<Real>()[0] << std::endl;
            }
            ModuleBase::timer::tick("Diago_LOBPCG", "diag");
            return true;
        }
        ModuleBase::timer::tick("Diago_LOBPCG", "iter_lock");
        // --- 2.6 check active eigenvalues and update blockvectors X, P, W ---
#ifdef DEBUG_LOBPCG
std::cout << "--- main loop: 2.6 check active eigenvalues and update blockvectors X, P, W ---" << std::endl;
#endif
    ModuleBase::timer::tick("Diago_LOBPCG", "iter_update_space");
        // 2.6.1 count active
        int n_conv = 0; // converged number
        for (int i = 0; i < n_max_; ++i) { if (done_.data<int>()[i]) ++n_conv; }
        n_active_ = n_max_ - n_conv;
        // All eigenvalues converged but sticky-lock guard prevented early exit above.
        // Copy best results and return to avoid calling hpsi_func with n_active_=0.
        if (n_active_ <= 0) {
            syncmem_complex_2d_op()(psi_in, ld_psi_in, this->x_new_.data<T>(), this->n_dim_, this->n_dim_, this->n_band_);
            copy_real_op(n_band_, this->eig_.data<Real>(), 1, eigenvalue_in, 1);
            if (this->comm_rank_ == 0) {
                std::cout << "Converged at iteration " << iter << " (all locked) with RMS residual " << r_norm_.data<Real>()[0] << std::endl;
            }
            ModuleBase::timer::tick("Diago_LOBPCG", "iter_update_space");
            ModuleBase::timer::tick("Diago_LOBPCG", "main_iter");
            ModuleBase::timer::tick("Diago_LOBPCG", "diag");
            return true;
        }
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

        // Coefficients live in the RR subspace: rr_dim_ rows, x_in_sub_ X-columns.
        // Hard locking: x_in_sub_ = n_active_ so get_expansion_coeffs' offset_x = 0
        // (active X is first in [active X | P | W]); soft: x_in_sub_ = n_max_, offset_x = n_conv.
        u_x_ = std::move(ct::Tensor(t_type_, device_type_, {rr_dim_, x_in_sub_}));
        u_p_ = std::move(ct::Tensor(t_type_, device_type_, {rr_dim_, n_active_})); // maximum size
        this->get_expansion_coeffs(len_space_, rr_dim_, x_in_sub_, n_active_,
            h_red_.data<T>(), u_x_.data<T>(), u_p_.data<T>());

        // p: space block [u P w]
        // p = space * u_p
        // hp = hspace * u_p
        // sp = sspace * u_p
        gemm_op('N', 'N', n_dim_, n_active_, rr_dim_,
            one, space_.data<T>() + rr_off_ * n_dim_, n_dim_, u_p_.data<T>(), rr_dim_, zero, evec_.data<T>(), n_dim_);
        copy_op(n_dim_*n_active_, evec_.data<T>(), 1, space_.data<T>() + ind_p_ * n_dim_, 1);
        gemm_op('N', 'N', n_dim_, n_active_, rr_dim_,
            one, hspace_.data<T>() + rr_off_ * n_dim_, n_dim_, u_p_.data<T>(), rr_dim_, zero, evec_.data<T>(), n_dim_);
        copy_op(n_dim_*n_active_, evec_.data<T>(), 1, hspace_.data<T>() + ind_p_ * n_dim_, 1);
        if(gen_eig){
            gemm_op('N', 'N', n_dim_, n_active_, rr_dim_,
                one, sspace_.data<T>() + rr_off_ * n_dim_, n_dim_, u_p_.data<T>(), rr_dim_, zero, evec_.data<T>(), n_dim_);
            copy_op(n_dim_*n_active_, evec_.data<T>(), 1, sspace_.data<T>() + ind_p_ * n_dim_, 1);
        }


#ifdef DEBUG_LOBPCG
        std::cout << "--- main loop: 2.7 update corresponding space from X, P, W ---" << std::endl;
#endif
        // --- 2.7 update corresponding space from X, P, W ---
        //
        // move updated Ritz vectors back into the search space. Under hard locking
        // only the active block [upd_off_, upd_off_+upd_n_) is written, leaving the
        // frozen locked X in space_[0, upd_off_) untouched. Soft locking writes all n_max.
        copy_op(n_dim_*upd_n_, x_new_.data<T>() + upd_off_ * n_dim_, 1, space_.data<T>() + upd_off_ * n_dim_, 1);
        copy_op(n_dim_*upd_n_, hx_new_.data<T>() + upd_off_ * n_dim_, 1, hspace_.data<T>() + upd_off_ * n_dim_, 1);
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
        // FIX: offset residual_ by ind_x_ to get residuals of active vectors
        copy_op(n_dim_ * n_active_,
                residual_.data<T>() + ind_x_ * n_dim_, 1, // here is the real bug fixed!!!
                space_.data<T>() + ind_w_ * n_dim_, 1);
        precondition_op(n_dim_, space_.data<T>(), ind_w_, n_active_, prec_.data<Real>(), eig_.data<Real>() + ind_x_);

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
        ModuleBase::timer::tick("Diago_LOBPCG", "iter_update_space");

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
            if (this->comm_rank_ == 0)
            {
                std::cerr << "LOBPCG did not converge within " << max_iter << " iterations." << std::endl;
            }
            // Return best available results so far
            syncmem_complex_2d_op()(psi_in, ld_psi_in, this->x_new_.data<T>(), this->n_dim_, this->n_dim_, this->n_band_);
            copy_real_op(n_band_, this->eig_.data<Real>(), 1, eigenvalue_in, 1);
            ModuleBase::timer::tick("Diago_LOBPCG", "main_iter");
#ifdef DEBUG_LOBPCG            
            // print current best eigenvalue and residual for all bands
            std::cerr << "Current best eigenvalues and residuals:" << std::endl;
            for (int i = 0; i < n_band_; ++i) {
                std::cerr << "Band " << i << ": Eigenvalue = " << eig_.
                data<Real>()[i] << ", Residual = " << r_norm_.data<Real>()[i] << std::endl;
            }
#endif            
            ModuleBase::timer::tick("Diago_LOBPCG", "diag");
            return false;
        }
        ModuleBase::timer::tick("Diago_LOBPCG", "main_iter");
    } // end - main for loop

    // // Final Rayleigh-Ritz to get the best approximation
    // // this->rayleigh_ritz(n_max_, hpsi_func, spsi_func, gen_eig);
    // syncmem_var_d2h_op()(eigenvalue_in, e_red_.data<Real>(), n_band_);

    // // Copy final eigenvectors back to input
    // auto final_x = space_.slice({0, ind_x_}, {n_basis_, n_band_});
    // syncmem_complex_op()(psi_in, final_x.data<T>(), n_basis_ * n_band_);

    // return converged;
    ModuleBase::timer::tick("Diago_LOBPCG", "diag");
    return true;
}

// ==================== Core Algorithmic Functions ====================

template <typename T, typename Device>
void DiagoLOBPCG<T, Device>::check_init_guess(const int n, const int m, T *x, const int ldx)
{
    ModuleBase::timer::tick("Diago_LOBPCG", "check_init_guess");
    // check value of x[n, m]
    // if zero, generate random guess
    if (x == nullptr || ldx == 0) {
        // warn that no initial guess provided, and quit
        if (this->comm_rank_ == 0)
        {
            std::cerr << "No initial guess provided. Quitting." << std::endl;
        }
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
    ModuleBase::timer::tick("Diago_LOBPCG", "check_init_guess");
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
    ModuleBase::timer::tick("Diago_LOBPCG", "rayleigh_ritz");
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
    setmem_complex_op()(h_red, T(0.0), len_space * len_space);
    gemm_op('C', 'N', len_working, len_working, dim, one, space, dim, hspace, dim, zero, h_red, len_space);
    this->allreduce_sum_inplace(h_red, len_space * len_space);
    // now h_red is the reduced matrix
    //
    // 2. Perform the Rayleigh-Ritz procedure to find the eigenvalues and eigenvectors of h_red
    // use heevx for solving all len_working eigenpairs of h_red
    // void operator()(const int dim, const int lda,const T *Mat, const int neig, Real *eigen_val, T *eigen_vec);
    // h_red(len_space, len_space), ld = len_space, n = len_working, solve len_working eigenpairs
#ifdef DEBUG_RR
std::cout << "--- INNER Rayleigh-Ritz: heevx ---" << std::endl;
#endif
    if (this->comm_rank_ == 0) {
        heevx(len_working, len_space, h_red, len_working, e_red, h_red);
    }
    this->bcast_inplace_real(e_red, len_working);
    this->bcast_inplace(h_red, len_space * len_space);
    // heevd(const int dim, T* Mat, const int lda, Real* eigen_val);
    // heevd(len_working, h_red, len_space, e_red);
    // now h_red is overwritten by eigenvectors, e_red for eigenvalues
    ModuleBase::timer::tick("Diago_LOBPCG", "rayleigh_ritz");
}

// ==================== Ortho ====================

template <typename T, typename Device>
void DiagoLOBPCG<T, Device>::ortho_local(const int n, const int m, T *x, const int ldx)
{
    if (n <= 0 || m <= 0) return;
    ct::kernels::lapack_geqrf_inplace<T, ct_Device> qr;
    qr(n, m, x, ldx);
}

template <typename T, typename Device>
void DiagoLOBPCG<T, Device>::ortho(const int n, const int m, T *x, const int ldx)
{
    ModuleBase::timer::tick("Diago_LOBPCG", "ortho");
    if (this->comm_nproc_ <= 1) {
        // Serial path: ortho by QR.
        this->ortho_local(n, m, x, ldx);
        ModuleBase::timer::tick("Diago_LOBPCG", "ortho");
        return;
    }

    // Parallel path: X <- X * (X^H X)^(-1/2)
    // 1) Build local Gram matrix.
    ct::Tensor gram(t_type_, device_type_, {m, m});
    gram.zero();
    gemm_op('C', 'N', m, m, n, one, x, ldx, x, ldx, zero, gram.data<T>(), m);
    this->allreduce_sum_inplace(gram.data<T>(), m * m);

    // 2) Solve Gram = U * D * U^H on root, then broadcast U,D.
    ct::Tensor eval(r_type_, device_type_, {m});
    eval.zero();
    if (this->comm_rank_ == 0) {
        heevx(m, m, gram.data<T>(), m, eval.data<Real>(), gram.data<T>());
    }
    this->bcast_inplace(gram.data<T>(), m * m);
    this->bcast_inplace_real(eval.data<Real>(), m);

    // 3) Compute U * D^(-1/2) * U^H.
    ct::Tensor u_scaled(t_type_, device_type_, {m, m});
    ct::Tensor g_inv_half(t_type_, device_type_, {m, m});
    copy_op(m * m, gram.data<T>(), 1, u_scaled.data<T>(), 1);

    const Real eps = 1e-14;
    for (int j = 0; j < m; ++j) {
        Real dj = eval.data<Real>()[j];
        if (dj < eps) {
            dj = eps;
        }
        const Real scale = 1.0 / std::sqrt(dj);
        for (int i = 0; i < m; ++i) {
            u_scaled.data<T>()[i + j * m] *= T(scale);
        }
    }
    gemm_op('N', 'C', m, m, m,
            one, u_scaled.data<T>(), m,
            gram.data<T>(), m,
            zero, g_inv_half.data<T>(), m);

    // 4) Apply normalization transform on local rows.
    ct::Tensor x_ortho(t_type_, device_type_, {n, m});
    gemm_op('N', 'N', n, m, m,
            one, x, ldx,
            g_inv_half.data<T>(), m,
            zero, x_ortho.data<T>(), ldx);
    copy_op(n * m, x_ortho.data<T>(), 1, x, 1);

    ModuleBase::timer::tick("Diago_LOBPCG", "ortho");

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
bool DiagoLOBPCG<T, Device>::is_orthonormal(const int n, const int m, const T *x, const int ldx, const Real tol)
{
#if defined(__CUDA) || defined(__ROCM)
    return false;
#else
    ct::Tensor gram(t_type_, device_type_, {m, m});
    gram.zero();
    gemm_op('C', 'N', m, m, n, one, x, ldx, x, ldx, zero, gram.data<T>(), m);
    this->allreduce_sum_inplace(gram.data<T>(), m * m);

    const T* gram_data = gram.data<T>();
    Real max_dev = 0.0;
    for (int j = 0; j < m; ++j) {
        for (int i = 0; i < m; ++i) {
            const T expected = (i == j) ? T(1.0) : T(0.0);
            const Real dev = std::abs(gram_data[i + j * m] - expected);
            if (dev > max_dev) {
                max_dev = dev;
            }
        }
    }
    return max_dev <= tol;
#endif
}

template <typename T, typename Device>
void DiagoLOBPCG<T, Device>::ortho_against_y(const int n, const int m, const int k, T *x, const int ldx, const T *y, const int ldy)
{
    ModuleBase::timer::tick("Diago_LOBPCG", "ortho_against_y");
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

    bool is_y_orthonormal = true;
    // Y (=[X,P]) is already orthonormal in all tested PW cases
    // (||Y^H Y - I|| <= ~1.3e-10), so use it directly instead of rebuilding an
    // orthonormal copy via QR on every call. The bounded re-orthogonalization
    // loop below self-corrects (and warns) if Y ever drifts. The original
    // QR-rebuild is kept here for reference, e.g. when swapping QR for a
    // Cholesky/polar orthogonalization:
    //     ct::Tensor y_ortho(t_type_, device_type_, {n, m});
    //     syncmem_complex_2d_op()(y_ortho.data<T>(), n, y, ldy, n, m);
    //     this->ortho(n, m, y_ortho.data<T>(), n);
    //     const T* y_ref = y_ortho.data<T>();
    //     const int ldy_ref = n;
    const T* y_ref = y;
    const int ldy_ref = ldy;
#ifdef DEBUG_ORTHO_Y
    {
        // Verify the orthonormality assumption on Y; warn if it drifts.
        ct::Tensor g(t_type_, device_type_, {m, m});
        gemm_op('C', 'N', m, m, n, one, y_ref, ldy_ref, y_ref, ldy_ref, zero, g.data<T>(), m);
        this->allreduce_sum_inplace(g.data<T>(), m * m);
        Real dev = 0.0;
        T* gd = g.data<T>();
        for (int j = 0; j < m; ++j)
            for (int i = 0; i < m; ++i) { T v = gd[i + j * m]; if (i == j) v -= T(1.0); dev += std::norm(v); }
        if (this->comm_rank_ == 0 && std::sqrt(dev) > 1.0e-6)
            std::cerr << "[ortho_against_y] WARNING: reference Y not orthonormal, ||Y^H Y - I|| = " << std::sqrt(dev) << std::endl;
    }
#endif

    // Compute Y'Y
#ifdef DEBUG_ORTHO_Y
// std::cout << "--- ortho_against_y: computing yby ---" << std::endl;
#endif
    // y(n, m), yby(m, m)
    // ct::Tensor yby(t_type_, device_type_, {m, m});
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
    // gemm_op('C', 'N', m, m, n, one, y, ldy, y, ldy, zero, yby.data<T>(), m);

    // Check if Y'Y is identity
    // For now, assume y is orthonormal to avoid complexity
    // In practice, we would check the diagonal and off-diagonal norms
    // For this implementation, we'll proceed with the assumption that y is orthonormal

    // Orthonormalize X before projecting. This keeps the W block well
    // conditioned; dropping it measurably increased the LOBPCG iteration count
    // (more hPsi), outweighing the saved ortho.
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
        gemm_op('C', 'N', m, k, n, one, y_ref, ldy_ref, x, ldx, zero, ybx.data<T>(), m);
        this->allreduce_sum_inplace(ybx.data<T>(), m * k);

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
        gemm_op('N', 'N', n, k, m, neg_one, y_ref, ldy_ref, y_coeff.data<T>(), m, one, x, ldx);

#if !defined(__CUDA) && !defined(__ROCM)
        Real x_norm_check = 0.0;
        for (int col = 0; col < k; ++col) {
            const T* x_col = x + col * ldx;
            for (int row = 0; row < n; ++row) {
                x_norm_check += std::abs(x_col[row]);
            }
        }
        this->allreduce_sum_inplace_real(&x_norm_check, 1);

        if (x_norm_check < 1.0e-12) {
#ifdef DEBUG_ORTHO_Y
            std::cout << "--- ortho_against_y: vector collapsed (norm=" << x_norm_check << "), stop re-orth loop ---" << std::endl;
#endif
            break;
        }
#endif

#ifdef DEBUG_ORTHO_Y
std::cout << "--- ortho_against_y: loop ortho x ---" << std::endl;
#endif
        // Orthonormalize x
        this->ortho(n, k, x, ldx);

        // Compute overlap = Y^T X after orthonormalization
        gemm_op('C', 'N', m, k, n, one, y_ref, ldy_ref, x, ldx, zero, overlap.data<T>(), m);
        this->allreduce_sum_inplace(overlap.data<T>(), m * k);

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

    if (iter_cnt <= 0 && this->comm_rank_ == 0) {
        // Too many ortho iterations, exit with warning
        std::cerr << "Too many iterations in ortho_against_y. Failed to reach tolerance." << std::endl;
    }
#ifdef DEBUG_LOBPCG
std ::cout << "--- ortho_against_y: end ---" << std::endl;
#endif
    ModuleBase::timer::tick("Diago_LOBPCG", "ortho_against_y");
}

template <typename T, typename Device>
void DiagoLOBPCG<T, Device>::ortho_against_y_local(const int n, const int m, const int k, T *x, const int ldx, const T *y, const int ldy)
{
    if (k <= 0) return;
    const Real tol_ortho = 1.0e-10;

    ct::Tensor y_ortho(t_type_, device_type_, {n, m});
    syncmem_complex_2d_op()(y_ortho.data<T>(), n, y, ldy, n, m);
    this->ortho_local(n, m, y_ortho.data<T>(), n);
    const T* y_ref = y_ortho.data<T>();
    const int ldy_ref = n;

    this->ortho_local(n, k, x, ldx);

    ct::Tensor ybx(t_type_, device_type_, {m, k});
    ct::Tensor y_coeff(t_type_, device_type_, {m, k});
    ct::Tensor overlap(t_type_, device_type_, {m, k});

    Real norm_overlap = 10.0;
    const int ITER_MAX = 10;
    int iter_cnt = ITER_MAX;

    while (norm_overlap >= tol_ortho && iter_cnt > 0) {
        gemm_op('C', 'N', m, k, n, one, y_ref, ldy_ref, x, ldx, zero, ybx.data<T>(), m);
        syncmem_complex_op()(y_coeff.data<T>(), ybx.data<T>(), m * k);
        gemm_op('N', 'N', n, k, m, neg_one, y_ref, ldy_ref, y_coeff.data<T>(), m, one, x, ldx);

#if !defined(__CUDA) && !defined(__ROCM)
        Real x_norm_check = 0.0;
        for (int col = 0; col < k; ++col) {
            const T* x_col = x + col * ldx;
            for (int row = 0; row < n; ++row) {
                x_norm_check += std::abs(x_col[row]);
            }
        }
        if (x_norm_check < 1.0e-12) {
            break;
        }
#endif

        this->ortho_local(n, k, x, ldx);

        gemm_op('C', 'N', m, k, n, one, y_ref, ldy_ref, x, ldx, zero, overlap.data<T>(), m);

        norm_overlap = 0.0;
#if defined(__CUDA) || defined(__ROCM)
        norm_overlap = 0.0;
#else
        T* overlap_data = overlap.data<T>();
        for (int i = 0; i < m * k; ++i) {
            norm_overlap += std::norm(overlap_data[i]);
        }
        norm_overlap = std::sqrt(norm_overlap);
#endif
        --iter_cnt;
    }

    if (iter_cnt <= 0 && this->comm_rank_ == 0) {
        std::cerr << "Too many iterations in ortho_against_y_local. Failed to reach tolerance." << std::endl;
    }
}

template <typename T, typename Device>
void DiagoLOBPCG<T, Device>::get_expansion_coeffs(const int len_space, const int len_working,
    const int n_max, const int n_active,
    T *h_red, T *u_x, T *u_p)
{
    ModuleBase::timer::tick("Diago_LOBPCG", "get_expansion_coeffs");
    if (n_active <= 0) {
        ModuleBase::timer::tick("Diago_LOBPCG", "get_expansion_coeffs");
        return;
    }
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
    // u_x/u_p are replicated dense coefficient matrices, not row-distributed
    // wavefunction blocks. Orthogonalize them locally; using MPI reductions here
    // would count the same coefficients once per rank and distort P.
    ortho_against_y_local(len_working, n_max, n_active, u_p, ld_u_p, u_x, ld_u_x);
#ifdef DEBUG_LOBPCG
std::cout << "--- get_expansion_coeffs: end ---" << std::endl;
#endif
    ModuleBase::timer::tick("Diago_LOBPCG", "get_expansion_coeffs");

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
