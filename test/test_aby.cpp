//*****************************************************************************
// Copyright 2018-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <random>

#include "ENCRYPTO_utils/crypto/crypto.h"
#include "ENCRYPTO_utils/parse_options.h"
#include "aby/aby_util.hpp"
#include "aby/kernel/relu_aby.hpp"
#include "abycore/aby/abyparty.h"
#include "abycore/circuit/booleancircuits.h"
#include "abycore/circuit/share.h"
#include "abycore/sharing/sharing.h"
#include "gtest/gtest.h"

namespace ngraph::runtime::aby {

TEST(aby, trivial) {
  int a = 1;
  int b = 2;
  EXPECT_EQ(3, a + b);
}

TEST(aby, create_party) {
  auto a = new ABYParty(CLIENT, "localhost", 30001, get_sec_lvl(128), 32, 2,
                        MT_OT, 100000);
  delete a;
  EXPECT_EQ(1, 1);
}

TEST(aby, create_unique_ptr_party) {
  auto a = std::make_unique<ABYParty>(CLIENT, "localhost", 30001,
                                      get_sec_lvl(128), 32, 2, MT_OT, 100000);
  EXPECT_EQ(1, 1);
}

TEST(aby, mod_reduce_zero_centered) {
  // Already in range
  EXPECT_DOUBLE_EQ(mod_reduce_zero_centered(0.1, 2.0), 0.1);

  // Below range
  EXPECT_DOUBLE_EQ(mod_reduce_zero_centered(-1.1, 2.0), 0.9);

  // Far below range
  EXPECT_DOUBLE_EQ(mod_reduce_zero_centered(-9.1, 2.0), 0.9);

  // Above range
  EXPECT_DOUBLE_EQ(mod_reduce_zero_centered(1.1, 2.0), -0.9);

  // Far above range
  EXPECT_DOUBLE_EQ(mod_reduce_zero_centered(9.1, 2.0), -0.9);
}

TEST(aby, split_vector) {
  {
    auto splits = split_vector(10, 3);
    EXPECT_EQ(splits.size(), 3);
    EXPECT_EQ(splits[0].first, 0);
    EXPECT_EQ(splits[0].second, 4);
    EXPECT_EQ(splits[1].first, 4);
    EXPECT_EQ(splits[1].second, 7);
    EXPECT_EQ(splits[2].first, 7);
    EXPECT_EQ(splits[2].second, 10);
  }
  {
    auto splits = split_vector(10, 4);
    EXPECT_EQ(splits.size(), 4);
    EXPECT_EQ(splits[0].first, 0);
    EXPECT_EQ(splits[0].second, 3);
    EXPECT_EQ(splits[1].first, 3);
    EXPECT_EQ(splits[1].second, 6);
    EXPECT_EQ(splits[2].first, 6);
    EXPECT_EQ(splits[2].second, 8);
    EXPECT_EQ(splits[3].first, 8);
    EXPECT_EQ(splits[3].second, 10);
  }
}

}  // namespace ngraph::runtime::aby
