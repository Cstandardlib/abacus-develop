#include <gtest/gtest.h>
#include <complex>
#include <vector>
#include <cmath>
#include <iostream>
#include <random>
#include "source_hsolver/diago_lobpcg.h"

// Define complex double type
using Complex = std::complex<double>;

// Declare LAPACK zheev helper
extern "C" {
    void zheev_(const char* jobz, const char* uplo, const int* n, Complex* a, const int* lda, double* w, Complex* work, const int* lwork, double* rwork, int* info);
    void zgemm_(const char* transa, const char* transb, const int* m, const int* n, const int* k,
                const Complex* alpha, const Complex* a, const int* lda,
                const Complex* b, const int* ldb,
                const Complex* beta, Complex* c, const int* ldc);
}

class DiagoLobpcgTest : public testing::Test {
protected:
    std::vector<Complex> matrix;
    
    // Generate matrix
    // type 0: Deterministic (Original)
    // type 1: Random Diagonally Dominant Complex Hermitian
    void GenerateMatrix(int n, int type) {
        matrix.resize(n * n);
        if (type == 0) {
            for (int j = 0; j < n; ++j) {
                for (int i = 0; i < n; ++i) {
                    if (i == j) {
                        matrix[j * n + i] = static_cast<double>(i + 1); // Diagonal 1..n
                    } else {
                        // Off-diagonal 
                        double val = 1.0 / (std::abs(i - j) + 1.0);
                        matrix[j * n + i] = val * 0.1; 
                    }
                }
            }
        } else if (type == 1) {
            // Random Hermitian matrix
            // Use specific seed for reproducibility
            std::mt19937 gen(42); 
            // Diagonal elements: spaced out to ensure good conditioning for basic LOBPCG
            // Off-diagonal: small random values
            std::uniform_real_distribution<> val_dist(-0.5, 0.5);

            for (int j = 0; j < n; ++j) {
                // Diagonal (real)
                // j+1 plus small noise. Keeps eigenvalues well separated approx 1.0 apart.
                matrix[j * n + j] = static_cast<double>(j + 1) + val_dist(gen) * 0.5;
                
                // Off-diagonal (complex)
                for (int i = j + 1; i < n; ++i) {
                    Complex val(val_dist(gen) * 0.05, val_dist(gen) * 0.05);
                    // A(i, j) at matrix[j*n + i]
                    matrix[j * n + i] = val;
                    // A(j, i) at matrix[i*n + j]
                    matrix[i * n + j] = std::conj(val);
                }
            }
        }
    }

    void VerifyLobpcg(int n, int nband, double check_tol = 1e-3, double cg_tol = 1e-5) {
        // ---------------------------------------------------------
        // 1. Solve with LAPACK (Gold Standard)
        // ---------------------------------------------------------
        std::vector<Complex> mat_lapack = matrix; // Deep copy
        std::vector<double> ev_lapack(n);
        
        Complex work_query;
        std::vector<double> rwork(3 * n - 2);
        int lwork_query = -1;
        int info = 0;
        int n_val = n;
        
        char jobz = 'N'; 
        char uplo = 'U';

        // Query workspace
        zheev_(&jobz, &uplo, &n_val, mat_lapack.data(), &n_val, ev_lapack.data(), &work_query, &lwork_query, rwork.data(), &info);
        
        int lwork = static_cast<int>(work_query.real()) + 1;
        std::vector<Complex> work(lwork);
        
        // Compute
        zheev_(&jobz, &uplo, &n_val, mat_lapack.data(), &n_val, ev_lapack.data(), work.data(), &lwork, rwork.data(), &info);
        
        ASSERT_EQ(info, 0) << "LAPACK zheev computation failed with info=" << info;
        
        // Output LAPACK eigenvalues for debug
        // std::cout << "LAPACK computed eigenvalues (first 5): ";
        // for(int i=0; i<5 && i<n; ++i) std::cout << ev_lapack[i] << " ";
        // std::cout << std::endl;

        // ---------------------------------------------------------
        // 2. Solve with LOBPCG
        // ---------------------------------------------------------
        std::vector<double> precondition(n, 1.0); // Identity Preconditioner
        
        int n_max = nband + 5; 
        hsolver::DiagoLOBPCG<Complex> lobpcg(precondition.data(), nband, n, n_max);
        
        std::vector<double> ev_lobpcg(nband);
        std::vector<Complex> psi(n * nband); 
        
        // Initialize psi with values
        for(auto &val : psi) val = static_cast<double>(rand()) / RAND_MAX;
        
        auto hpsi_func = [&](Complex* in, Complex* out, const int ld, const int nvec) {
            char transa = 'N';
            char transb = 'N';
            int m_ = n;
            int n_ = nvec;
            int k_ = n;
            Complex alpha = 1.0;
            Complex beta = 0.0;
            int lda = n;
            
            zgemm_(&transa, &transb, &m_, &n_, &k_, 
                   &alpha, matrix.data(), &lda, 
                   in, &ld, 
                   &beta, out, &ld);
        };
        
        int max_iter = 2000;
        bool converged = lobpcg.diag(
            hpsi_func,
            nullptr, 
            false, 
            ev_lobpcg.data(),
            psi.data(),
            n, 
            cg_tol,
            max_iter
        );
        
        EXPECT_TRUE(converged) << "LOBPCG did not converge in " << max_iter << " iterations";
        
        // Output LOBPCG eigenvalues for debug
        // std::cout << "LOBPCG computed eigenvalues (first 5): ";
        // for(int i=0; i<5 && i<nband; ++i) std::cout << ev_lobpcg[i] << " ";
        // std::cout << std::endl;
        
        // ---------------------------------------------------------
        // 3. Compare Results
        // ---------------------------------------------------------
        for(int i = 0; i < nband; ++i) {
            EXPECT_NEAR(ev_lobpcg[i], ev_lapack[i], check_tol) 
                << "Mismatch at eigenvalue index " << i 
                << " LAPACK: " << ev_lapack[i] << " LOBPCG: " << ev_lobpcg[i];
        }
        // output anyway even if test passed, for debug
        std::cout << "Eigenvalues comparison (LOBPCG vs LAPACK):" << std::endl;
        std::cout << "LAPACK eigenvalues: ";
        for(int i=0; i<nband; ++i) {
            std::cout << "Index " << i << ": " << ev_lapack[i] << " vs " << ev_lapack[i] << std::endl;
        }
        std::cout << std::endl;
        std::cout << "LOBPCG eigenvalues: ";
        for(int i=0; i<nband; ++i) {
            std::cout << "Index " << i << ": " << ev_lobpcg[i] << " vs " << ev_lapack[i] << std::endl;
        }
    }
};

TEST_F(DiagoLobpcgTest, CompareWithLapack) {
    int n = 100;
    int nband = 10;
    GenerateMatrix(n, 0);
    VerifyLobpcg(n, nband);
}

TEST_F(DiagoLobpcgTest, LargeScale) {
    int n = 200; 
    int nband = 20;
    GenerateMatrix(n, 0);
    VerifyLobpcg(n, nband);
}

TEST_F(DiagoLobpcgTest, RandomMatrix) {
    int n = 50;
    int nband = 10;
    GenerateMatrix(n, 1);
    VerifyLobpcg(n, nband, 0.1, 1e-2); 
}
