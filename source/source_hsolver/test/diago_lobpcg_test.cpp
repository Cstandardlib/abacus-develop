#include "source_base/inverse_matrix.h"
#include "source_base/module_external/lapack_connector.h"
#include "source_pw/module_pwdft/structure_factor.h"
#include "source_psi/psi.h"
#include "source_hamilt/hamilt.h"
#include "source_pw/module_pwdft/hamilt_pw.h"
#include "../diago_iter_assist.h"
#include "../diago_lobpcg.h"
#include "diago_mock.h"
#include "mpi.h"
#include "source_basis/module_pw/test/test_tool.h"

#include <gtest/gtest.h>
#include <complex>
#include <random>

// mock diago_hs_para (Tier 2): the standalone LOBPCG unit tests run single-process, so the
// distributed Rayleigh-Ritz path in diago_lobpcg.cpp is never taken at runtime. This stub
// (mirroring test_hsolver_pw.cpp) satisfies the linker without pulling in ELPA/ScaLAPACK.
#ifdef __MPI
#include "source_base/macros.h"
namespace hsolver {
template <typename T>
void diago_hs_para(T* h, T* s, const int lda, const int nband,
                   typename GetTypeReal<T>::type* const ekb, T* const wfc,
                   const MPI_Comm& comm, const int diag_subspace, const int block_size)
{}
template void diago_hs_para<double>(double*, double*, const int, const int,
    GetTypeReal<double>::type* const, double* const, const MPI_Comm&, const int, const int);
template void diago_hs_para<std::complex<double>>(std::complex<double>*, std::complex<double>*, const int, const int,
    GetTypeReal<std::complex<double>>::type* const, std::complex<double>* const, const MPI_Comm&, const int, const int);
template void diago_hs_para<float>(float*, float*, const int, const int,
    GetTypeReal<float>::type* const, float* const, const MPI_Comm&, const int, const int);
template void diago_hs_para<std::complex<float>>(std::complex<float>*, std::complex<float>*, const int, const int,
    GetTypeReal<std::complex<float>>::type* const, std::complex<float>* const, const MPI_Comm&, const int, const int);
}
#endif

/************************************************
 *  unit test of functions in Diago_LOBPCG
 ***********************************************/

/**
 * Class Diago_LOBPCG is an approach for eigenvalue problems using the Locally Optimal Block Preconditioned Conjugate Gradient method.
 * This unittest tests the function Diago_LOBPCG::diag() for FPTYPE=double, Device=cpu
 * with different examples:
 *  - Hermite matrices (npw=500,1000) with sparsity 0%, 60%, 80%
 *  - Hamiltonian matrix read from "H-KPoints-Si64.dat"
 *  - a small 2x2 Hermite matrix for verification
 *
 * Note:
 * The test passes if eigenvalues are close to those computed by LAPACK within a tolerance.
 */

// call lapack in order to compare to lobpcg
void lapackEigen(int npw, std::vector<std::complex<double>> &hm, double *e, bool outtime = false)
{
    clock_t start, end;
    start = clock();
    int lwork = 2 * npw;
    std::complex<double> *work2 = new std::complex<double>[lwork];
    double *rwork = new double[3 * npw - 2];
    int info = 0;
    char tmp_c1 = 'V', tmp_c2 = 'U';
    zheev_(&tmp_c1, &tmp_c2, &npw, hm.data(), &npw, e, work2, &lwork, rwork, &info);
    end = clock();
    if (outtime) {
        std::cout << "Lapack Run time: " << (double)(end - start) / CLOCKS_PER_SEC << " S" << std::endl;
    }
    delete[] rwork;
    delete[] work2;
}

class DiagoLOBPCGPrepare
{
public:
    DiagoLOBPCGPrepare(int nband, int npw, int sparsity, bool reorder, double eps, int maxiter, double threshold)
        : nband(nband), npw(npw), sparsity(sparsity), reorder(reorder), eps(eps), maxiter(maxiter),
          threshold(threshold)
    {
#ifdef __MPI
        MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
        MPI_Comm_rank(MPI_COMM_WORLD, &mypnum);
#endif
    }

    int nband, npw, sparsity, maxiter, notconv;
    double eps, avg_iter;
    bool reorder;
    double threshold;
    int nprocs = 1, mypnum = 0;

    void CompareEigen(double *precondition)
    {
        // LAPACK reference eigenvalues
        double *e_lapack = new double[npw];
        auto ev = DIAGOTEST::hmatrix;
        if (mypnum == 0) {
            lapackEigen(npw, ev, e_lapack, false);
        }
#ifdef __MPI
        MPI_Bcast(e_lapack, npw, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif

        // Initial guess psi
        ModuleBase::ComplexMatrix psiguess(nband, npw);
        std::default_random_engine p(1);
        std::uniform_int_distribution<unsigned> u(1, 10);
        for (int i = 0; i < nband; i++) {
            for (int j = 0; j < npw; j++) {
                double rand = static_cast<double>(u(p)) / 10.;
                psiguess(i, j) = ev[j * DIAGOTEST::h_nc + i] * rand;
            }
        }

        // Setup psi and preconditioner
        int ik = 1;
        psi::Psi<std::complex<double>> psi;
        psi.resize(ik, nband, npw);
        for (int i = 0; i < nband; i++) {
            for (int j = 0; j < npw; j++) {
                psi(i, j) = psiguess(i, j);
            }
        }

        psi::Psi<std::complex<double>> psi_local;
        double* precondition_local;

        DIAGOTEST::npw_local = new int[nprocs];
#ifdef __MPI
        DIAGOTEST::cal_division(DIAGOTEST::npw);
        DIAGOTEST::divide_hpsi(psi, psi_local, DIAGOTEST::hmatrix, DIAGOTEST::hmatrix_local);
        precondition_local = new double[DIAGOTEST::npw_local[mypnum]];
        DIAGOTEST::divide_psi<double>(precondition, precondition_local);
#else
        DIAGOTEST::hmatrix_local = DIAGOTEST::hmatrix;
        DIAGOTEST::npw_local[0] = DIAGOTEST::npw;
        psi_local = psi;
        precondition_local = new double[DIAGOTEST::npw];
        for (int i = 0; i < DIAGOTEST::npw; ++i) precondition_local[i] = precondition[i];
#endif

        // Recommended: nmax slightly larger than nband
        const int nmax = nband + 5;
        // const int nmax = nband + 1;
        // output parameters
        std::cout << "LOBPCG parameters: nband=" << nband << ", npw=" << npw
                  << ", sparsity=" << sparsity << ", eps=" << eps
                  << ", maxiter=" << maxiter << std::endl;
        hsolver::DiagoLOBPCG<std::complex<double>, base_device::DEVICE_CPU> lobpcg(
            precondition_local, nband, DIAGOTEST::npw_local[mypnum], nmax);

        psi_local.fix_k(0);
        double start = MPI_Wtime();

        using T = std::complex<double>;
        const int dim = DIAGOTEST::npw_local[mypnum];
        const std::vector<T>& h_mat = DIAGOTEST::hmatrix_local;

        auto hpsi_func = [h_mat, dim](T* psi_in, T* hpsi_out, const int ld_psi, const int nvec) {
            auto one = T(1.0);
            auto zero = T(0.0);
            // base_device::DEVICE_CPU ctx{};
            ModuleBase::gemm_op<T, base_device::DEVICE_CPU>()(
                'N', 'N',
                dim, nvec, dim,
                &one,
                h_mat.data(), dim,
                psi_in, ld_psi,
                &zero,
                hpsi_out, ld_psi);
        };

        // For standard eigenvalue problem, spsi_func = nullptr, gen_eig = false
        double* en = new double[npw];
        // output parameters
        std::cout << "Starting LOBPCG diagonalization..." << std::endl;
        std::cout << "dim: " << dim << ", nband: " << nband << ", nmax: " << nmax << std::endl;
        std::cout << "npw: " << DIAGOTEST::npw_local[mypnum] << std::endl;
        std::cout << "ld_psi: " << psi_local.get_current_ngk() << std::endl;
        bool converged = lobpcg.diag(
            hpsi_func,
            nullptr,      // spsi_func
            false,        // gen_eig = false (standard problem)
            en,
            psi_local.get_pointer(),
            npw,  // leading dimension of psi = ndim
            eps,
            maxiter
        );

        double end = MPI_Wtime();
        // if (mypnum == 0) printf("LOBPCG diago time: %7.3f s, converged: %s\n", end - start, converged ? "true" : "false");

        // Check eigenvalues against LAPACK
        for (int i = 0; i < nband; i++) {
            EXPECT_NEAR(en[i], e_lapack[i], threshold);
        }

        delete[] en;
        delete[] e_lapack;
        delete[] DIAGOTEST::npw_local;
        delete[] precondition_local;
    }
};

class DiagoLOBPCGTest : public ::testing::TestWithParam<DiagoLOBPCGPrepare>
{
};

TEST_P(DiagoLOBPCGTest, RandomHamilt)
{
    DiagoLOBPCGPrepare dcp = GetParam();
    hsolver::DiagoIterAssist<std::complex<double>>::PW_DIAG_NMAX = dcp.maxiter;
    hsolver::DiagoIterAssist<std::complex<double>>::PW_DIAG_THR = dcp.eps;

    HPsi<std::complex<double>> hpsi(dcp.nband, dcp.npw, dcp.sparsity);
    DIAGOTEST::hmatrix = hpsi.hamilt();
    DIAGOTEST::npw = dcp.npw;

    dcp.CompareEigen(hpsi.precond());
}

INSTANTIATE_TEST_SUITE_P(VerifyLOBPCG,
                         DiagoLOBPCGTest,
                         ::testing::Values(
                             DiagoLOBPCGPrepare(10, 500, 0, true, 1e-5, 300, 5e-2),
                             DiagoLOBPCGPrepare(20, 500, 6, true, 1e-5, 300, 5e-2),
                             DiagoLOBPCGPrepare(20, 1000, 8, true, 1e-5, 300, 5e-2),
                             DiagoLOBPCGPrepare(40, 1000, 8, true, 1e-6, 300, 5e-2)));

// Small matrix sanity check
TEST(DiagoLOBPCGTest, Hamilt)
{
    int dim = 2;
    int nbnd = 2;
    HPsi<std::complex<double>> hpsi(nbnd, dim);
    std::vector<std::complex<double>> hm = hpsi.hamilt();
    EXPECT_EQ(DIAGOTEST::h_nr, 2);
    EXPECT_EQ(DIAGOTEST::h_nc, 2);
    EXPECT_EQ(hm[0].imag(), 0.0);
    EXPECT_EQ(hm[DIAGOTEST::h_nc + 1].imag(), 0.0);
    EXPECT_EQ(conj(hm[DIAGOTEST::h_nc]).real(), hm[1].real());
    EXPECT_EQ(conj(hm[DIAGOTEST::h_nc]).imag(), hm[1].imag());
}

// Test with real Hamiltonian from file
TEST(DiagoLOBPCGTest, readH)
{
    std::vector<std::complex<double>> hm;
    // std::ifstream ifs("H-KPoints-Si64.dat");
    // std::ifstream ifs("H-small-3x3.dat");
    std::ifstream ifs("H-small-20x20.dat");
    if (!ifs.is_open()) {
        GTEST_SKIP() << "File H-small-20x20.dat not found. Skipping test.";
        return;
    }
    DIAGOTEST::readh(ifs, hm);
    ifs.close();

    int dim = DIAGOTEST::npw;
    // int nband = std::min(10, dim - 1); // ensure nband < dim
    // For small systems, we must ensure nband is small enough so that 
    // the subspace size (approx 3*nband) does not exceed the physical dimension.
    // For 20x20, we set nband = 4.
    int nband = std::min(4, dim / 3);

    DiagoLOBPCGPrepare dcp(nband, dim, 0, true, 1e-5, 500, 1e-1);
    hsolver::DiagoIterAssist<std::complex<double>>::PW_DIAG_NMAX = dcp.maxiter;
    hsolver::DiagoIterAssist<std::complex<double>>::PW_DIAG_THR = dcp.eps;
    hsolver::DiagoIterAssist<std::complex<double>>::SCF_ITER = 1;

    HPsi<std::complex<double>> hpsi;
    hpsi.create(nband, dim);
    DIAGOTEST::hmatrix = hpsi.hamilt();
    DIAGOTEST::npw = dim;

    dcp.CompareEigen(hpsi.precond());
}

int main(int argc, char **argv)
{
    int nproc = 1, myrank = 0;

#ifdef __MPI
    int nproc_in_pool, kpar = 1, mypool, rank_in_pool;
    setupmpi(argc, argv, nproc, myrank);
    divide_pools(nproc, myrank, nproc_in_pool, kpar, mypool, rank_in_pool);
    MPI_Comm_split(MPI_COMM_WORLD, myrank, 0, &BP_WORLD);
    GlobalV::NPROC_IN_POOL = nproc;
#else
    MPI_Init(&argc, &argv);
#endif

    testing::InitGoogleTest(&argc, argv);
    ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();
    if (myrank != 0) {
        delete listeners.Release(listeners.default_result_printer());
    }

    int result = RUN_ALL_TESTS();

    if (myrank == 0 && result != 0) {
        std::cout << "ERROR: some tests are not passed" << std::endl;
    }

    MPI_Finalize();
    return result;
}
