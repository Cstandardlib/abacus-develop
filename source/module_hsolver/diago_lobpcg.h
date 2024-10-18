#ifndef DIAGO_LOBPCG_H_
#define DIAGO_LOBPCG_H_

/**
 * @file diago_lobpcg.h
 * @brief Header file for the DiagoLOBPCG class template.
 *
 * This file contains the definition of the DiagoLOBPCG class template, which
 * implements the Locally Optimal Block Preconditioned Conjugate Gradient (LOBPCG)
 * method for solving large-scale eigenvalue problems.
 */

#include <functional>                           // std::function
#include <complex>                              // std::complex

#include <module_base/macros.h>                 // GetTypeReal
#include <module_base/module_device/types.h>    // base_device::DEVICE_CPU or GPU
#include <module_hsolver/kernels/math_kernel_op.h>

#include <ATen/core/tensor.h>
#include <ATen/core/tensor_map.h>

namespace hsolver{

/**
 * @class DiagoLOBPCG
 * @brief A class template for the LOBPCG eigenvalue solver.
 *
 * @tparam T The data type for the matrix elements, default is std::complex<double>.
 * @tparam Device The device type for computation, default is base_device::DEVICE_CPU.
 *
 * This class implements the LOBPCG method for solving large-scale eigenvalue problems.
 * It supports user-defined functions for matrix-vector multiplications and preconditioning.
 */
template <typename T = std::complex<double>, typename Device = base_device::DEVICE_CPU>
class DiagoLOBPCG {
private:
    using Real = typename GetTypeReal<T>::type; ///< Real type for the matrix elements.
public:
    
    /**
     * @brief Constructor for the DiagoLOBPCG class.
     * 
     * It prepares the solver and allocates memory for the solver
     * according to dimension provided.
     * 
     * @tparam T The data type of the input array.
     * @param ld_psi Leading dimension of the input initial guess of eigenvectors.
     * @param n_eigenpairs Number of eigenpairs to compute.
     */
    DiagoLOBPCG(const int ld_psi, const int n_eigenpairs);

    /**
     * @brief Destructor for the DiagoLOBPCG class.
     */
    ~DiagoLOBPCG();
    /**
     * @brief Type alias for the Hamiltonian matrix-blockvector multiplication function.
     * 
     * For eigenvalue problem HX = λX or generalized eigenvalue problem HX = λSX,
     * this function computes the product of the Hamiltonian matrix H and a blockvector X.
     *
     * @param[out] X      Pointer to the input blockvector.
     * @param[out] HX     Pointer to the output blockvector.
     * @param[in]  ld     Leading dimension of matrix.
     * @param[in]  nvec   Number of eigenpairs, i.e. number of vectors in a block.
     * 
     * @warning X and HX are the exact address to read input X and store output H*X,
     * @warning both of size ld * nvec.
     */
    using HPsiFunc = std::function<void(T*, T*, const int, const int)>;
    /**
     * @brief Type alias for the overlap matrix-blockvector multiplication function.
     * 
     * For generalized eigenvalue problem HX = λSX,
     * this function computes the product of the overlap matrix S and a blockvector X.
     *
     * @param[in]  X      Pointer to the input blockvector.
     * @param[out] SX     Pointer to the output blockvector.
     * @param[in]  ld     Leading dimension of matrix.
     * @param[in]  nvec   Number of vectors in a block.
     */
    using SPsiFunc = std::function<void(T*, T*, const int, const int)>;
    /**
     * @brief Type alias for a function that performs a preconditioning operation.
     *
     * @param[in]  X        Pointer to the input blockvector.
     * @param[out] PX       Pointer to the output blockvector.
     * @param[in]  ld       Leading dimension of matrix.
     * @param[in]  nvec     Number of vectors in a block.
     */
    using PrecndFunc = std::function<void(T*, T*, const int, const int)>;

    /**
     * @brief Initializes the LOBPCG solver with the provided wavefunctions.
     *
     * This function sets up the solver using the given wavefunctions.
     * It prepares the solver to find the specified number of eigenvalues,
     * and allocates memory for the solver, (according to nmax set by neig)
     *
     * @tparam T The data type of the wavefunctions, typically a floating-point type.
     * @param psi Pointer to the array containing the initial wavefunctions.
     * @param ld_psi Leading dimension of the psi array.
     * @param neig Number of eigenvalues to compute.
     * 
     */
    // void init(T *psi_in, const int ld_psi, const int neig);

    /**
     * @brief Perform the diagonalization using the LOBPCG method.
     *
     * @param hpsi_func Function for Hamiltonian matrix-vector multiplication.
     * @param spsi_func Function for overlap matrix-vector multiplication.
     * @param precnd_func Function for preconditioning.
     * @param eigenvalue_in Pointer to store the resulting eigenvalues.
     * @param ldPsi Leading dimension of the psi input.
     * @param psi_in Pointer to the eigenvectors.
     * @param lobpcg_diag_thr Convergence threshold for the Davidson iteration.
     * @param lobpcg_maxiter Maximum allowed iterations for the Davidson method.
     * @param ntry_max Maximum number of diagonalization attempts (default is 5).
     * @param notconv_max Maximum number of allowed non-converged eigenvectors.
     * @return int Status of the diagonalization process.
     */
    int diag(
        const HPsiFunc& hpsi_func,
        const SPsiFunc& spsi_func,      // matrix-vector multiplication function
        const PrecndFunc& precnd_func,  // preconditioning function
        const bool solving_generalized, // whether to solve generalized eigenproblem
        Real *eigenvalue_in,            // initial guess of eigenvalues, 1-D array
        T *psi_in,                      // initial guess of eigenvectors, 2-D array, block-vector
        // const int ld_psi,               // leading dimension of psi
        // const int n_eigenpairs,         // number of eigenvectors to be computed
        const double tolerance,         // convergence threshold for the LOBPCG iteration
        const int max_iterations        // maximum allowed iterations for the LOBPCG method
        );
private:
    const int ldx;                  ///< Leading dimension of all block-vectors.
    const int n_eigenpairs;         ///< Number of eigenvectors sought.
    const int n_max_subspace;       ///< Actual number of eigenvectors computed. Initialized according to n_eigenpairs.
                                    ///< slightly larger than the number of eigenpairs to improve the convergence.
    const int size_space;           ///< Total maximum size of working space V = [x p w]
    int n_active;                   ///< Number of active eigenvectors.

    // iteration control parameters are passed as const arguments in the diag() function
    // int max_iterations;             ///< Maximum number of iterations.
    // double tolerance;               ///< Convergence tolerance.

    Device * ctx = {};              ///< ctx is nothing but a pointer to the device as arguments in the kernel ops.

    ct::DataType r_type  = ct::DataType::DT_INVALID;     ///< Real type
    ct::DataType t_type  = ct::DataType::DT_INVALID;     ///< T data type
    ct::DeviceType device_type = ct::DeviceType::UnKnown;///< Device type

    // data for expansion space, and corresponding Matrix-Vector results and residuals
    // here v x p w donotes the corresponding working space:
    // v is total space for X, P, W
    // x is eigenvectors
    // p is implicitly computed difference between current and previous eigenvector approximations
    // w is preconditioned residual
    ct::Tensor v    = {};         ///< Tensor for eigenvecotrs
    ct::Tensor hv   = {};         ///< Tensor for H * [x p w]
    ct::Tensor sv   = {};         ///< Tensor for S * [x p w]
    // x, p, w are blocks of v, hv, sv work space.

    // temporary memory for the last iter
    ct::Tensor x    = {};         ///< Tensor for the last x (i.e. eigenvectors)
    ct::Tensor hx   = {};         ///< Tensor for the last H * x
    ct::Tensor sx   = {};         ///< Tensor for the last S * x

    // psi is the current approximation of eigenvectors and read from the input
    ct::Tensor psi  = {};         ///< Tensor for the current approximation of eigenvectors

    // reduced problem
    // soft locking by Knyazev, all x, active p, active w will engage in Rayleigh-Ritz
    // dynamic size of h_reduced(n_working_space, n_working_space), eig_reduced(n_working_space)
    // the subspace expands from v=[x] to v=[x p w], h_reduced = v'Hv
    // where x has the width of n_max_subspace, while w and p have the width of n_active
    ct::Tensor h_reduced   = {};    ///< Tensor for the reduced Hamiltonian
    ct::Tensor eig_reduced = {};    ///< 1-D Tensor for the reduced eigenvalues

    // convergence check
    ct::Tensor residual = {};      ///< Tensor for the residual
    ct::Tensor r_norm = {};        ///< 1-D Tensor for the residual norm

    // working space for ortho
    // ct::Tensor overlap = {};        ///< Tensor for the overlap matrix used by ortho

    const T one = static_cast<T>(1.0), zero = static_cast<T>(0.0), neg_one = static_cast<T>(-1.0);

    // main steps


    // tools
    // all tool functions take direct T* pointer to the data
    // and the dimension information to perform linear algebra operations

    // orthogonalization procedures
    /**
     * @brief Orthonormalize the given block vector x.
     * 
     * Orthogonalize given `x` of shape(n, m) using Cholesky decomposition.
     * Columns of the matrix `x` will be orthogonalized against each other.
     * 
     * @param n Number of rows.
     * @param m Number of columns. There are `m` n-dim vectors in the block.
     * @param x Pointer to the block vector to be orthogonalized.
     */
    void ortho(int n, int m, T *x);
    /**
     * @brief Orthogonalizes block vector x against block vector y.
     *
     * Orthogonalizes block vector `x` against a given \b orthonormal set `y`
     * and orthonormalizes `x`. If `y` is not orthonormal, extra computation
     * is needed inside this function.
     * 
     * `y(n, m)` is a given block vector, assumed to be orthonormal
     * `x(n, k)` is the block vector to be orthogonalized
     * first against x and then internally.
     * 
     * if y is not orthonormal, we will do by solving (Y'Y)y_coeff = Y'X
     * X = X - Y(Y'Y)^{-1}Y'X = X - Y * y_coeff
     *
     * @tparam T The data type of the elements in the vectors (e.g., float, double).
     * @tparam Device The device type where the computation is performed (e.g., CPU, GPU).
     * @param n The number of rows in `x` and `y`.
     * @param m The number of columns in `y`.
     * @param k The number of columns in `x`.
     * @param x Pointer to the block-vector `x`.
     * @param y Pointer to the block-vector `y`.
     */
    void ortho_against_y(int n, int m, int k, T *x, T *y);

    // s-ortho
    /**
     * @brief S-Orthogonalizes block vector x.
     *
     * S-Orthogonalize given `x` of shape(n, m) using Cholesky decomposition.
     * Columns of the matrix `x` will be s-orthogonalized against each other.
     *
     * @tparam T The data type of the matrix elements.
     * @param n Number of rows.
     * @param m Number of columns. There are `m` n-dim vectors in the block.
     * @param x Pointer to the block vector to be S-orthogonalized.
     * @param sx Pointer to S * x.
     */
    void s_ortho(int n, int m, T *x, T *sx);
    /**
     * @brief S-Orthogonalizes block vector x against block vector y.
     *
     * @tparam T The data type of the elements in the vectors (e.g., float, double).
     * @tparam Device The device type where the computation is performed (e.g., CPU, GPU).
     * @param n The number of rows in `x` and `y`.
     * @param m The number of columns in `y`.
     * @param k The number of columns in `x`.
     * @param x Pointer to the block-vector `x`.
     * @param y Pointer to the block-vector `y`.
     * @param sy Pointer to S * y.
     */
    void s_ortho_against_y(int n, int m, int k, T *x, T *y, T *sy);

    // check initial guess and do first ortho
    
    /**
     * @brief Checks the initial guess for the eigenvector.
     *
     * This function verifies the initial guess for the eigenvector
     * to ensure they are orthonormal.
     *
     * @tparam T The data type of the eigenvector elements.
     * @param n The number of rows in the eigenvector guess matrix.
     * @param m The number of columns in the eigenvector guess matrix.
     * @param evec The guess matrix to be checked and orthogonalized if needed.
     */
    void check_init_guess(int n, int m, T *evec);

    // eigenvalue of reduced problem
    /**
     * @brief Compute the eigenvalues of the reduced Hamiltonian.
     * 
     * dense
     *
     * @param n Dimension of square matrix H.
     * @param h Pointer to the matrix H.
     * @param eig Pointer to the eigenvalues.
     */
    void eigh(int n, int ldh, T *h, Real *eig);

    /**
     * @brief Compute the residuals without preconditioning.
     * 
     * W = T(H - λS)X
     *   = T(HX - λSX)
     *
     * @tparam T The data type of the elements (e.g., float, double).
     * @param n The number of rows in the matrix.
     * @param m The number of columns in the matrix.
     * @param hx Pointer to Hx, H times block eigenvector approximation.
     * @param sx Pointer to Sx, S times block eigenvector approximation.
     * @param eig Pointer to the 1-D array containing the eigenvalues.
     * @param res Pointer to the array where the computed preconditioned residuals will be stored.
     */
    void compute_residual(int n, int m, const T *hx,const T *sx, const T *eig, T *res);

    void check_convergence(const T *x, const T *hx, const T *sx, const T *eig, int n, int m, Real tol, int &nconv);

    // op
    using ct_Device = typename ct::PsiToContainer<Device>::type;
    using setmem_var_op = ct::kernels::set_memory<Real, ct_Device>;
    using resmem_var_op = ct::kernels::resize_memory<Real, ct_Device>;
    using delmem_var_op = ct::kernels::delete_memory<Real, ct_Device>;
    using syncmem_var_h2d_op = ct::kernels::synchronize_memory<Real, ct_Device, ct::DEVICE_CPU>;
    using syncmem_var_d2h_op = ct::kernels::synchronize_memory<Real, ct::DEVICE_CPU, ct_Device>;

    using setmem_complex_op = ct::kernels::set_memory<T, ct_Device>;
    using delmem_complex_op = ct::kernels::delete_memory<T, ct_Device>;
    using resmem_complex_op = ct::kernels::resize_memory<T, ct_Device>;
    // sync(out, in, size)
    using syncmem_complex_op = ct::kernels::synchronize_memory<T, ct_Device, ct_Device>;

}; // class DiagoLOBPCG

} // namespace hsolver

#endif // DIAGO_LOBPCG_H_