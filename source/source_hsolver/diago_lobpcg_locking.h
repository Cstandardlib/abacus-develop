#ifndef DIAGO_LOBPCG_LOCKING_H
#define DIAGO_LOBPCG_LOCKING_H

#include <vector>

namespace hsolver
{

// Locking policy for LOBPCG converged-band deflation.
//   Cascade     : current production default. Locks the contiguous converged prefix only
//                 (a band can lock iff every lower band is locked). Kept for A/B + fallback.
//                 Note: the cascade never actually un-locks a band in normal operation -- the
//                 wipe below only ever touches already-zero entries, because done_ is always a
//                 contiguous prefix and locked bands are skipped by the early `continue`.
//   Independent : per-band sticky lock; a band locks when its own criterion fires, regardless
//                 of lower bands. Produces a possibly NON-CONTIGUOUS done_, handled downstream
//                 by the active-order gather.
enum class LobpcgLockPolicy
{
    Cascade = 0,
    Independent = 2
};

// Update per-band lock flags `done` (length n_max) in place from this iteration's per-band
// convergence flags `band_ok` (length n_max) and the iteration index. Pure integer logic --
// no allocation, no device/MPI state -- so it is unit-testable in isolation.
inline void lobpcg_update_locks(int* done,
                                const int* band_ok,
                                int n_max,
                                int iter,
                                LobpcgLockPolicy policy)
{
    if (policy == LobpcgLockPolicy::Cascade)
    {
        // only lock the first (contiguous) converged eigenvalues/vectors
        for (int i = 0; i < n_max; ++i)
        {
            if (done[i])
            {
                continue;
            }
            if (iter > 0 && band_ok[i])
            {
                done[i] = 1;
            }
            if (!done[i])
            {
                for (int j = i; j < n_max; ++j)
                {
                    done[j] = 0;
                }
                break;
            }
        }
    }
    else // Independent
    {
        for (int i = 0; i < n_max; ++i)
        {
            if (done[i])
            {
                continue; // sticky: never un-lock
            }
            if (iter > 0 && band_ok[i])
            {
                done[i] = 1;
            }
        }
    }
}

// Build a stable partition permutation listing LOCKED bands first (ascending home index),
// then ACTIVE bands (ascending home index).
//   perm[k]      = home band index occupying gathered slot k (length n_max)
//   active_order = home indices of the active (unconverged) bands (= perm tail, length
//                  n_max - n_conv)
//   n_conv       = number of locked bands
// Pure; the only allocation is the output vectors (cleared and refilled).
inline void lobpcg_build_active_order(const int* done,
                                      int n_max,
                                      std::vector<int>& perm,
                                      std::vector<int>& active_order,
                                      int& n_conv)
{
    perm.clear();
    active_order.clear();
    for (int i = 0; i < n_max; ++i)
    {
        if (done[i])
        {
            perm.push_back(i);
        }
    }
    n_conv = static_cast<int>(perm.size());
    for (int i = 0; i < n_max; ++i)
    {
        if (!done[i])
        {
            perm.push_back(i);
            active_order.push_back(i);
        }
    }
}

} // namespace hsolver

#endif // DIAGO_LOBPCG_LOCKING_H
