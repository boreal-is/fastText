/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include "dictionary.h"
#include "fasttext.h"

namespace fasttext {

class CompactDictionary : public Dictionary {
  public:
    void writeCompact(std::string word_fn, std::string data_fn, std::string map_fn, const FastText& ft, int32_t nrwords) const;
    void getSubwordsFrequency(const std::string& word, std::vector<int32_t>& sub_count) const;
    std::vector<float> transform(int32_t ndim, const float* data, const float* map, int32_t map_size) const;
};

}
