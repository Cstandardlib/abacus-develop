#ifndef DIAGO_LOBPCG_H_
#define DIAGO_LOBPCG_H_

#define DEBUG_SCF
// #define DEBUG_LOBPCG
// #define DEBUG_INIT
// #define DEBUG_RR
// #define DEBUG_ORTHO_Y

/**
 * @file diago_lobpcg.h
 * @brief Header file for the DiagoLOBPCG class template.
 *
 * This file contains the definition of the DiagoLOBPCG class template, which
 * implements the Locally Optimal Block Preconditioned Conjugate Gradient (LOBPCG)
 * method for solving large-scale eigenvalue problems.
 *
 * Reference: Knyazev, A. V. (2001). Toward the optimal preconditioned eigensolver:
 * Locally optimal block preconditioned conjugate gradient method.
 * SIAM journal on scientific computing, 23(2), 517-541.
 */

#include "source_hsolver/kernels/bpcg_kernel_op.h"
#include <functional>
#include <complex>

#include <source_base/macros.h>
#include <source_base/module_device/types.h>
#include <source_base/kernels/math_kernel_op.h> // gemm

#include <source_base/module_container/ATen/core/tensor.h>
#include <source_base/module_container/ATen/core/tensor_types.h>
#include <source_base/module_container/ATen/kernels/memory.h>
#include <source_base/module_container/ATen/kernels/blas.h> // copy, gemm, axpy, nrm2
#include <source_base/module_container/ATen/kernels/lapack.h> // geqrf, heevd
// #include <ATen/core/tensor_map.h>

namespace ct = container;

namespace hsolver {

// Now only standard eigenvalue problems implemented
// and s_ortho / s_ortho_against_y are still left blank


/**
 * @class DiagoLOBPCG
 * @brief LOBPCG eigensolver for large-scale eigenvalue problems.
 *
 * This implementation use s-ortho instead of solving generalized dense eigenvalue problems.
 *
 * @tparam T Data type for matrix elements (std::complex<double> or std::complex<float>)
 * @tparam Device Device type for computation (base_device::DEVICE_CPU or base_device::DEVICE_GPU)
 *
 * Implements the LOBPCG algorithm with support for both standard and generalized
 * eigenvalue problems. Uses block operations for better convergence and performance.
 */
template <typename T = std::complex<double>, typename Device = base_device::DEVICE_CPU>
class DiagoLOBPCG {
private:
    using Real = typename GetTypeReal<T>::type; ///< Real type for eigenvalues

public:
    /**
     * @brief Hamiltonian matrix-vector multiplication function type.
     *
     * Computes H * X for block vector X.
     *
     * @param[in] X Input block vector (size ld * nvec)
     * @param[out] HX Output block vector H*X (size ld * nvec)
     * @param[in] ld Leading dimension
     * @param[in] nvec Number of vectors in block
     */
    using HPsiFunc = std::function<void(T*, T*, const int, const int)>;

    /**
     * @brief Overlap matrix-vector multiplication function type.
     *
     * Computes S * X for block vector X (generalized eigenvalue problems only).
     *
     * @param[in] X Input block vector (size ld * nvec)
     * @param[out] SX Output block vector S*X (size ld * nvec)
     * @param[in] ld Leading dimension
     * @param[in] nvec Number of vectors in block
     */
    using SPsiFunc = std::function<void(T*, T*, const int, const int)>;

    // /**
    //  * @brief Preconditioner application function type.
    //  *
    //  * Applies preconditioner to residual vectors.
    //  *
    //  * @param[in] n Number of rows
    //  * @param[in] m Number of vectors
    //  * @param[in] shift Diagonal shift for preconditioner
    //  * @param[in] res Residual vectors (size n * m)
    //  * @param[out] pres Preconditioned residuals (size n * m)
    //  */
    // using PrecndFunc = std::function<void(const int, const int, const Real, T*, T*)>;

    /**
     * @brief Constructor for DiagoLOBPCG.
     *
     * Allocates necessary memory and initializes parameters.
     *
     * @param precondition Pointer to preconditioner data, diagonal elements array of size nbasis
     * @param nband Number of eigenvectors to compute
     * @param nbasis Matrix dimension (basis size)
     * @param ndim Valid dimension (may be less than nbasis)
     * @param nmax Maximum size of the search space (should be >= nband), and eig/evec should be n_max and (n_basis, n_max)
     *
     * @note
     * Please determine the nmax with nband when calling this interface.
     * recommended nmax: slightly greater than nband,
     * for example nmax = nband + 5
     */
    DiagoLOBPCG(
        const Real *precondition,
        const int nband,
        const int ndim, const int nmax);

    /**
     * @brief Destructor for DiagoLOBPCG.
     *
     * Automatically frees all allocated memory.
     * Does not need explicit deallocation for ct::Tensor.
     * Does not free external references.
     */
    ~DiagoLOBPCG();// = default;

    /**
     * @brief Perform diagonalization using LOBPCG method.
     *
     * @param hpsi_func Hamiltonian matrix-blockvector multiplication
     * @param spsi_func Overlap matrix-blockvector multiplication (nullptr for standard problems),
     *        only referenced if gen_eig is true
     * @param precnd_func Preconditioner application function
     * @param gen_eig Whether to solve generalized eigenvalue problem
     * @param[in,out] eigenvalue_in Output eigenvalues (size nband)
     * @param[in,out] psi_in Input/Output eigenvectors (size nbasis * nband, column major)
     * @param tolerance Convergence tolerance
     * @param max_iter Maximum number of iterations
     * @return bool True if convergence achieved, false otherwise
     */
    bool diag(
        const HPsiFunc& hpsi_func,
        const SPsiFunc& spsi_func, // only referenced if if_eig is true
        // const PrecndFunc& precnd_func,
        const bool gen_eig,
        Real *eigenvalue_in,
        T *psi_in, const int ld_psi_in,
        const double tolerance,
        const int max_iter
    );
    /*
     * psi_in(n_basis, n_band) is copied into
     * evec_(n_dim, n_max)
     */

private:
    // int ld_psi_ = 0;     ///< leading dimension of psi_in. should ONLY be used with psi_in!!!
    // directly use ld_psi_in instead


    // [] for actual shape in memory; () for effective shape referenced


    // Problem dimensions
    int n_band_ = 0;     ///< Number of eigenvectors to compute
    int n_dim_ = 0;      ///< leading dimension and matrix size (number of rows) of local space
    int n_max_ = 0;      ///< Maximum subspace size

    // Search space organization: [X, P, W]
    // X: current eigenvectors (n_dim × n_max)
    // P: conjugate directions (n_dim × n_active)
    // W: preconditioned residuals (n_dim × n_active)
    // where n_active is the number of active (non-converged) eigenvectors, <= n_max_
    // n_dim_ is the leading dimension and matrix size (number of rows) of local space
    int ind_x_ = 0;      ///< Start index of X block
    int ind_p_ = 0;      ///< Start index of P block
    int ind_w_ = 0;      ///< Start index of W block
    int len_space_ = 0;  ///< Total search space size (3 * n_max)
    int len_working_ = 0; ///< working search space size (n_max + 2 * n_active)

    // Convergence control
    int n_active_ = 0;   ///< Number of active (non-converged) eigenvectors
    Real tol_rms_ = 1e-6; ///< RMS residual tolerance
    Real tol_max_ = 1e-5; ///< Maximum residual tolerance

    // Main storage tensors (column-major format)
    ct::Tensor eig_ = {}; ///< Eigenvectors of the problem [n_max_]
    ct::Tensor evec_ = {}; ///< Eigenvectors of the problem [n_basis_, n_max_]
    ct::Tensor space_ = {};     ///< Search space [X, P, W] of shape [n_basis_, 3*n_max_]
    ct::Tensor hspace_ = {};    ///< H * space (Hamiltonian applied to search space)
    ct::Tensor sspace_ = {};    ///< S * space (overlap applied to search space, generalized problem only)
    /*
     * space
     * max(memory assigned)
     * [     X     |     P     |     W     ]
     *     n_max   |   n_max   |   n_max
     * active space:
     *    ind_x  ind_p    ind_w  -- these are indices of active vectors in space_, hspace_, sspace_
     *       |     |        |
     *       v     v        v
     * [convX|  X  |    P   |    W   ]
     *    n_max    |n_active|n_active
     *
     * and X,P,W construct the basis set -- search subspace for Rayleigh-Ritz
     */

    // Reduced problem storage
    ct::Tensor h_red_ = {};     ///< Reduced Hamiltonian matrix of shape [3*n_max_, 3*n_max_]
    ct::Tensor e_red_ = {};     ///< Eigenvalues of reduced problem, 1-D

    // Temporary storage for vector updates
    ct::Tensor x_new_ = {};     ///< New X vectors [n_basis_, n_max_]
    ct::Tensor hx_new_ = {};    ///< H * x_new [n_basis_, n_max_]
    ct::Tensor sx_new_ = {};    ///< S * x_new [n_basis_, n_max_] (generalized problem only)

    // Residual and convergence monitoring
    ct::Tensor residual_ = {};  ///< Residual vectors R = H*X - λ*S*X [n_basis_, n_max_]
    ct::Tensor r_norm_ = {};    ///< Residual norms for each eigenvector [2, n_max_]
    ct::Tensor done_ = {};      ///< Convergence flags for each eigenvector [n_max_]

    // Expansion coefficients for subspace updates
    ct::Tensor u_x_ = {};       ///< Expansion coefficients for X vectors [3*n_max_, n_max_]
    ct::Tensor u_p_ = {};       ///< Expansion coefficients for P vectors [3*n_max_, n_active_]

    // Preconditioner storage
    ct::Tensor h_prec_ = {};    ///< Host-side preconditioner data (read-only reference)
    ct::Tensor prec_ = {};      ///< Device-side preconditioner data

    // Mathematical constants
    const T *one = nullptr, *zero = nullptr, *neg_one = nullptr;
    const T one_ = static_cast<T>(1.0);
    const T zero_ = static_cast<T>(0.0);
    const T neg_one_ = static_cast<T>(-1.0);

    // Device context (nullptr for CPU)
    ct::DataType r_type_ = ct::DataType::DT_INVALID;     ///< Real data type identifier
    ct::DataType t_type_ = ct::DataType::DT_INVALID;     ///< T data type identifier
    ct::DeviceType device_type_ = ct::DeviceType::UnKnown; ///< Device type identifier
    // Device* ctx_ = nullptr;                                     ///< Device context for kernel operations
                ///< ctx is nothing but a pointer to the device as arguments in the kernel ops.


    // Memory operation types
    // used for copy psi_in into evec_ for different data layout
    // destination vs source, leading dimensions are not the same
    // void operator()(FPTYPE* arr_out, const size_t dpitch,
    //     const FPTYPE* arr_in, const size_t spitch,
    //     const size_t width, const size_t height)
    // d for destination, s for source
    using syncmem_complex_2d_op = base_device::memory::synchronize_memory_2d_op<T, Device, Device>;
    // note ct_Device is different from template Device
    using ct_Device = typename ct::PsiToContainer<Device>::type;
    using setmem_var_op = ct::kernels::set_memory<Real, ct_Device>;
    using resmem_var_op = ct::kernels::resize_memory<Real, ct_Device>;
    using delmem_var_op = ct::kernels::delete_memory<Real, ct_Device>;
    // sync: void operator()(T* arr_out, const T* arr_in, const size_t& size);
    // sync out = in (size)
    using syncmem_var_h2d_op = ct::kernels::synchronize_memory<Real, ct_Device, ct::DEVICE_CPU>;
    using syncmem_var_d2h_op = ct::kernels::synchronize_memory<Real, ct::DEVICE_CPU, ct_Device>;

    using setmem_complex_op = ct::kernels::set_memory<T, ct_Device>;
    using setmem_int_op = ct::kernels::set_memory<int, ct_Device>;
    using delmem_complex_op = ct::kernels::delete_memory<T, ct_Device>;
    using resmem_complex_op = ct::kernels::resize_memory<T, ct_Device>;
    using syncmem_complex_op = ct::kernels::synchronize_memory<T, ct_Device, ct_Device>;

    // Mathematical operation types
    // --- BLAS ---
    // Note: or ct_kernels::blas_gemm<T, ct_Device>?
    // using gemm_op = ct::kernels::blas_gemm<T, ct_Device>;
    ct::kernels::blas_gemm<T, ct_Device> gemm_op;   ///< T/Device gemm
    ct::kernels::blas_copy<T, ct_Device> copy_op;
    // note that real op is different from T op
    ct::kernels::blas_copy<Real, ct_Device> copy_real_op;
    // note: these APIs take T constants, if Real constant is provided, it should be converted to T *
    ct::kernels::blas_axpy<T, ct_Device> axpy_op;
    // Real operator()(const int n, const T *x, const int incx);
    ct::kernels::blas_nrm2<T, ct_Device> nrm2_op;

    // --- LAPACK ---
    ct::kernels::lapack_heevx<T, ct_Device> heevx;
    // ct::kernels::lapack_heevd<T,ct_Device> heevd;

    // ModuleBase op
    // using gemm_op = ModuleBase::gemm_op<T, Device>;              ///< Matrix multiplication operation
    // hsolver op
    hsolver::precondition_op<T, Device> precondition_op;
    // --- Core algorithmic functions ---
private:
    /**
     * @brief Check and orthogonalize initial guess.
     *
     * @param[in,out] x Eigenvector guess (input/output)
     */
    void check_init_guess(const int n, const int m, T *x, const int ldx);

    /**
     * @brief Perform Rayleigh-Ritz procedure in current subspace.
     * solve h_red * x = lambda * x
     * where h_red is built as space^T * H * space = space^T * hspace
     * h_red[len_space_, len_space_](len_working_, len_working_)
     *  = space[n_dim_, len_space_](n_dim_, len_working_) ^T * hspace[len_space_, len_working_](len_space_, len_working_)
     *
     * [IN] space, hspace -> [INNER] h_red, e_red -> [OUT] h_red overwritten by eigenvectors, e_red for eigenvalues
     *
     * solving a reduced standard eigenvalue problem on the space spanned by the S-ortho vectors.
     * hspace = H * space
     * 1. first calculate h_red = space^T * H * space = space^T * hspace
     * 2. then solve h_red * x = lambda * x
     * ! not update space = space * h_red
     *
     * INPUT space, aspace (dim, len_space) | use (dim * len_working) block
     * OUTPUT h_red, e_red to store eigenvectors / eigenvalues
     * h_red(len_space, len_space), use (len_working, len_working) block
     *
     * @param[in] space Search space
     * @param[in] hspace H*X matrix eigenvectors
     * @param[in] ldspace Leading dimension of space, hspace, h_red
     * @param[in] len_working Current working subspace size [X P W]
     * @param[out] h_red Reduced matrix eigenvectors, size(ldspace, ldspace),
     *             use (len_working, len_working) block
     * @param[out] e_red Reduced matrix eigenvalues, use (len_working) block
     */
    void rayleigh_ritz(T *space, T *hspace, const int len_space, const int dim, const int len_working,
        T *h_red, Real *e_red);

    // --- orthogonalization procedures ---
    /**
     * @brief Orthonormalize a set of m vectors (of size n) using QR decomposition.
     *
     * Orthogonalize given `x` of shape(n, m) using QR factorization.
     * Columns of the matrix `x` will be orthogonalized against each other.
     *
     * @param n Number of rows.
     * @param m Number of columns. There are `m` n-dim vectors in the block.
     * @param x Pointer to the block vector, x(n, m) to be orthogonalized.
     * @param ldx Leading dimension of the block vector x.
     *
     * @note Use geqrf + orgqr to obtain orthogonal Q.
     */
    void ortho(const int n, const int m, T *x, const int ldx);

    /**
     * @brief Orthogonalizes block vector `x` against a given \b orthonormal set `y`
     *         and orthonormalizes `x`. If `y` is not orthonormal, extra computation
     *         is needed inside this function.
     * `y(n, m)` is a given block vector, assumed to be orthonormal
     * `x(n, k)` is the block vector to be orthogonalized
     * first against `y` and then internally.
     * @param n The number of rows in `x` and `y`.
     * @param m The number of columns in `y`.
     * @param k The number of columns in `x`.
     * @param x The matrix to be orthogonalized. size of n x k.
     * @param ldx Leading dimension of the block vector.
     * @param y The matrix to orthogonalize against. size of n x m. assumed to be orthonormal
     * @param ldy Leading dimension of the reference matrix.
     */
    void ortho_against_y(int n, int m, int k, T *x, int ldx, const T *y, int ldy);

    /**
     * @brief S-orthogonalize vectors for generalized problems.
     *
     * @param x Vectors to orthogonalize (input/output)
     * @param sx S * x
     */
    // void s_ortho(int n, int m, T *x, int ldx, const T *sx);

    /**
     * @brief S-orthogonalize vectors against existing S-orthonormal set.
     *
     * @param n Number of rows
     * @param m Number of columns
     * @param x Vectors to orthogonalize (input/output)
     * @param sx S * x
     * @param y S-orthonormal reference vectors
     * @param sy S * y
     */
    // void s_ortho_against_y(int n, int m, T *x, int ldx, const T *y, int ldy, const T *sy);

    /**
     * @brief Solve dense symmetric eigenvalue problem.
     *
     * Computes eigenvalues and eigenvectors of a dense symmetric matrix
     * using appropriate LAPACK/CUDA routines depending on device type.
     *
     * @param n Dimension of the matrix
     * @param h Input matrix (column major, size n×n), will be overwritten with eigenvectors
     * @param ldh Leading dimension of h
     * @param eig Output eigenvalues (size n)
     */
    // void densi_eig(int n, T *h, int ldh, T *eig);


    /**
     * @brief Solve dense symmetric eigenvalue problem.
     *
     * @param h Input matrix (column major)
     * @param eig Output eigenvalues
     */
    // void dense_eigen_solve(ct::Tensor& h, ct::Tensor& eig);

    /**
     * @brief Compute residuals R = HX - λSX.
     *
     * @param hx H * x
     * @param sx S * x (or x for standard problems)
     * @param eig Eigenvalues
     * @param res Output residuals
     * @param gen_eig Whether generalized problem
     */
    // void compute_residuals(const ct::Tensor& hx, const ct::Tensor& sx,
    //                       const ct::Tensor& eig, ct::Tensor& res, const bool gen_eig);

    /**
     * @brief Check convergence and update active set.
     *
     * @param r_norms Residual norms [rms, max] for each vector
     * @param done Convergence flags (output)
     * @return int Number of converged vectors
     */
    // int check_convergence(const ct::Tensor& r_norms, ct::Tensor& done);

    /**
     * @brief Compute expansion coefficients for P vectors.
     *
     * Given the eigenvectors or reduced h_red
     * extract the expansion coefficients for x_new(u_x)
     * and assemble the ones for p_new in u_p
     *
     * @param len_u Total subspace size
     * @param n_active Number of active vectors
     * @param h_red[in] Reduced matrix eigenvectors
     * @param u_x[out] X coefficients (output)
     * @param u_p[out] P coefficients (output)
     */
    void get_expansion_coeffs(const int len_space, const int len_working,
        const int n_max, const int n_active,
        T *h_red, T *u_x, T *u_p);

    /**
     * @brief Update search space indices for next iteration.
     */
    // void update_search_space_indices();



    /**
     * @brief Apply preconditioner to residuals.
     *
     * @param precnd_func Preconditioner function
     * @param shift Diagonal shift
     * @param res Residual vectors
     * @param pres Preconditioned residuals (output)
     */
    // void apply_preconditioner(const PrecndFunc& precnd_func, const Real shift,
    //                          const ct::Tensor& res, ct::Tensor& pres);

    /**
     * @brief Compute residual norms.
     *
     * @param res Residual vectors
     * @param r_norms Output norms [rms, max] for each vector
     */
    // void compute_residual_norms(const ct::Tensor& res, ct::Tensor& r_norms);

    /**
     * @brief Update the current approximation.
     *
     * @param len_u Current subspace size
     */
    // void update_approximation(const int len_u);

    /**
     * @brief Lock converged eigenvectors.
     */
    // void lock_converged_vectors();

    // Memory management
    // void allocate_memory();
    // void free_memory();

}; // class DiagoLOBPCG

} // namespace hsolver

#endif // DIAGO_LOBPCG_H_
