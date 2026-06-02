#include "../diago_lobpcg_locking.h"

#include <gtest/gtest.h>
#include <vector>

using namespace hsolver;

/************************************************
 *  unit tests for the pure LOBPCG locking helpers
 *  (diago_lobpcg_locking.h): lock-policy update +
 *  active-order stable partition.
 ***********************************************/

TEST(LobpcgLocking, CascadeIterZeroLocksNothing)
{
    std::vector<int> done(5, 0);
    std::vector<int> ok = {1, 1, 1, 1, 1};
    lobpcg_update_locks(done.data(), ok.data(), 5, /*iter=*/0, LobpcgLockPolicy::Cascade);
    EXPECT_EQ(done, (std::vector<int>{0, 0, 0, 0, 0}));
}

TEST(LobpcgLocking, CascadeStopsAtFirstFailure)
{
    std::vector<int> done(5, 0);
    std::vector<int> ok = {1, 1, 0, 1, 1};
    lobpcg_update_locks(done.data(), ok.data(), 5, 1, LobpcgLockPolicy::Cascade);
    EXPECT_EQ(done, (std::vector<int>{1, 1, 0, 0, 0})); // bands 3,4 blocked by band 2
}

TEST(LobpcgLocking, IndependentLocksNonContiguous)
{
    std::vector<int> done(5, 0);
    std::vector<int> ok = {1, 1, 0, 1, 1};
    lobpcg_update_locks(done.data(), ok.data(), 5, 1, LobpcgLockPolicy::Independent);
    EXPECT_EQ(done, (std::vector<int>{1, 1, 0, 1, 1})); // 3,4 lock despite band 2
}

TEST(LobpcgLocking, IndependentIsSticky)
{
    std::vector<int> done = {1, 1, 0, 1, 1};
    std::vector<int> ok = {1, 1, 1, 0, 0}; // bands 3,4 regress
    lobpcg_update_locks(done.data(), ok.data(), 5, 2, LobpcgLockPolicy::Independent);
    EXPECT_EQ(done, (std::vector<int>{1, 1, 1, 1, 1})); // locked stay locked; band 2 now locks
}

TEST(LobpcgLocking, IndependentIterZeroLocksNothing)
{
    std::vector<int> done(5, 0);
    std::vector<int> ok = {1, 1, 1, 1, 1};
    lobpcg_update_locks(done.data(), ok.data(), 5, /*iter=*/0, LobpcgLockPolicy::Independent);
    EXPECT_EQ(done, (std::vector<int>{0, 0, 0, 0, 0}));
}

TEST(LobpcgLocking, ActiveOrderPartition)
{
    std::vector<int> done = {1, 1, 0, 1, 0};
    std::vector<int> perm, active;
    int n_conv = -1;
    lobpcg_build_active_order(done.data(), 5, perm, active, n_conv);
    EXPECT_EQ(n_conv, 3);
    EXPECT_EQ(perm, (std::vector<int>{0, 1, 3, 2, 4})); // locked first, then active
    EXPECT_EQ(active, (std::vector<int>{2, 4}));
}

TEST(LobpcgLocking, ActiveOrderAllActive)
{
    std::vector<int> done = {0, 0, 0};
    std::vector<int> perm, active;
    int n_conv = -1;
    lobpcg_build_active_order(done.data(), 3, perm, active, n_conv);
    EXPECT_EQ(n_conv, 0);
    EXPECT_EQ(perm, (std::vector<int>{0, 1, 2}));
    EXPECT_EQ(active, (std::vector<int>{0, 1, 2}));
}

TEST(LobpcgLocking, ActiveOrderAllLocked)
{
    std::vector<int> done = {1, 1, 1};
    std::vector<int> perm, active;
    int n_conv = -1;
    lobpcg_build_active_order(done.data(), 3, perm, active, n_conv);
    EXPECT_EQ(n_conv, 3);
    EXPECT_EQ(perm, (std::vector<int>{0, 1, 2}));
    EXPECT_TRUE(active.empty());
}
