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
#include "module_base/module_device/types.h"    // base_device::DEVICE_CPU or GPU

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
class DiagoLOBPCG final {
private:
    using Real = typename GetTypeReal<T>::type; ///< Real type for the matrix elements.
public:
    /**
     * @brief Constructor for the DiagoLOBPCG class.
     *
     * @param precondition Pointer to the preconditioner.
     * @param maxIterations Maximum number of iterations.
     * @param tolerance Convergence tolerance.
     */
    DiagoLOBPCG(const Real* precondition, int maxIterations, double tolerance);

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
     * @param[out] X    Pointer to input blockvector of type `T*`.
     * @param[in]  HX   Pointer to output blockvector of type `T*`.
     * @param[in]  nvec Number of eigebpairs, i.e. how many vectors in a block.
     * @param[in]  dim  Dimension of matrix.
     * @param[in]  id_start Start index of blockvector.
     * @param[in]  id_end   End index of blockvector.
     */
    using HPsiFunc = std::function<void(T*, T*, const int, const int, const int, const int)>;
    /**
     * @brief Type alias for the overlap matrix-blockvector multiplication function.
     * 
     * For generalized eigenvalue problem HX = λSX,
     * this function computes the product of the overlap matrix S and a blockvector X.
     *
     * @param[in]   X     Pointer to the input array.
     * @param[out] SX     Pointer to the output array.
     * @param[in] nrow    Dimension of SX: nbands * nrow.
     * @param[in] npw     Number of plane waves.
     * @param[in] nbands  Number of bands.
     * 
     * @note called by spsi_func(X, SX, dim, dim, 1).
     */
    using SPsiFunc = std::function<void(T*, T*, const int, const int, const int)>;

    
    /**
     * @brief Type alias for a function that performs a preconditioning operation.
     *
     * @param[in]  X    Pointer to the input blockvector.
     * @param[out] PX   Pointer to the output blockvector.
     * @param[in]  nvec Number of eigebpairs, i.e. how many vectors in a block.
     * @param[in]  dim  Dimension of matrix.
     */
    using PrecndFunc = std::function<void(T*, T*, const int, const int, const int)>;
    /**
     * @brief Perform the diagonalization using the LOBPCG method.
     *
     * @param hpsi_func Function for Hamiltonian matrix-vector multiplication.
     * @param spsi_func Function for overlap matrix-vector multiplication.
     * @param precnd_func Function for preconditioning.
     * @param ldPsi Leading dimension of the psi input.
     * @param psi_in Pointer to the eigenvectors.
     * @param eigenvalue_in Pointer to store the resulting eigenvalues.
     * @param lobpcg_diag_thr Convergence threshold for the Davidson iteration.
     * @param lobpcg_maxiter Maximum allowed iterations for the Davidson method.
     * @param ntry_max Maximum number of diagonalization attempts (default is 5).
     * @param notconv_max Maximum number of allowed non-converged eigenvectors.
     * @return int Status of the diagonalization process.
     */
    int diag(
        const HPsiFunc& hpsi_func,
        const SPsiFunc& spsi_func,
        const PrecndFunc& precnd_func,
        const int ldPsi, T *psi_in, Real* eigenvalue_in, const Real lobpcg_diag_thr, const int lobpcg_maxiter, const int ntry_max = 5, const int notconv_max = 0);
private:
    int maxIterations_; ///< Maximum number of iterations.
    double tolerance_; ///< Convergence tolerance.
    /**
     * @brief Orthogonalize the given matrix.
     * @param X Pointer to the matrix to be orthogonalized.
     * @param n Number of rows in the matrix.
     * @param k Number of columns in the matrix.
     */
    void orthogonalize(T* X, int n, int k);

    /**
     * @brief Compute the residual matrix.
     *
     * @param A Pointer to the matrix A.
     * @param X Pointer to the matrix X.
     * @param eigenvalues Pointer to the eigenvalues.
     * @param R Pointer to the residual matrix.
     * @param n Number of rows in the matrices.
     * @param k Number of columns in the matrices.
     */
    void computeResidual(const T* A, const T* X, const T* eigenvalues, T* R, int n, int k);


} // namespace hsolver

#endif // DIAGO_LOBPCG_H_