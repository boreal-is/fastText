/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "compact_dictionary.h"
#include "fasttext.h"
#include <cmath>
#include <list>


namespace fasttext {

std::vector<float> CompactDictionary::transform(int32_t ndim, const float* data, const float* map, int32_t map_size) const
{
    std::vector<float> r(ndim,0.0);
    if(map_size != ndim)
        memcpy(r.data(), data, ndim*sizeof(ndim));
    else
    {
        for(int32_t i = 0; i < ndim; ++i)
            for(int32_t j = 0; j < ndim; ++j)
                r[i] += data[j]*map[i*ndim+j];
    }
    return r;
}

void CompactDictionary::getSubwordsFrequency(const std::string& word, std::vector<int32_t>& sub_count) const 
{
  for (size_t i = 0; i < word.size(); i++) {
    std::string ngram;
    // check if utf8 continuation
    if ((word[i] & 0xC0) == 0x80) continue;
    for (size_t j = i, n = 1; j < word.size() && n <= args_->maxn; n++) {
      ngram.push_back(word[j++]);
      while (j < word.size() && (word[j] & 0xC0) == 0x80) {
        ngram.push_back(word[j++]);
      }
      if (n >= args_->minn && !(n == 1 && (i == 0 || j == word.size()))) {
        int32_t h = hash(ngram) % args_->bucket;
        sub_count[h] +=  1;
      }
    }
  }
}

void CompactDictionary::writeCompact(std::string word_fn, std::string data_fn, std::string map_fn, const FastText& ft, int32_t nrwords) const
{
    std::shared_ptr<const Matrix> m = ft.getInputMatrix();
    FILE *fw, *fd, *fmap;
    fw = fopen(word_fn.c_str(), "wb");
    fd = fopen(data_fn.c_str(), "wb");

    printf("Before map\n");

    fmap = map_fn == "" ? fopen(map_fn.c_str(), "rb") : NULL;

    printf("After map\n");

    std::list<std::pair<int32_t,int32_t> > ord_subs;
    
    int32_t nsubs_bucket, nwords_bucket, n_, nchars;;
    nsubs_bucket = args_->bucket;
    nwords_bucket = word2int_.size();
    n_ = m->cols();
    nchars = 0;
    int8_t minn, maxn;
    minn = args_->minn;
    maxn = args_->maxn;
    printf("Before assign map\n");
    std::vector<double> map_((fmap == NULL ? 1 : n_*n_), 0.0);
    std::vector<float> map((fmap == NULL ? 1 : n_*n_), 0.0);
    std::vector<int32_t> sub_count(nsubs_bucket, 0);

    printf("Compute subwords frequecies... ");
    for(int i = 0; i < nwords_; ++i)
        getSubwordsFrequency(words_[i].word, sub_count);
    printf("done.");

    for(int i = 0; i < nwords_; ++i)
    {
        std::string w = getWord(i);
        fwrite (w.c_str(), sizeof(char), w.size()+1, fw);
        if(i < nrwords)
            nchars += (w.size()+1);
    }
    fclose(fw);

    if(fmap != NULL)
    {
        fread(map_.data(), sizeof(double), n_*n_, fmap);
        for(int i = 0; i < n_*n_; ++i)
            map[i] = (float)map_[i];
        fclose(fmap);
    }

    fwrite(&nwords_, sizeof(int32_t), 1, fd); // Number of words
    fwrite(&nrwords, sizeof(int32_t), 1, fd); // Number of restricted words
    fwrite(&nwords_bucket, sizeof(int32_t), 1, fd); // Number of words hash
    fwrite(&(nsubs_bucket), sizeof(int32_t), 1, fd); // Number of subs
    fwrite(&n_, sizeof(int32_t), 1, fd); // Number of dims
    fwrite(&nchars, sizeof(int32_t), 1, fd); // min sub
    fwrite(&minn, sizeof(int8_t), 1, fd); // min sub
    fwrite(&maxn, sizeof(int8_t), 1, fd); // max sub

    printf("Processing sizes... done.\n");
    printf("Sorting subs... ");
    double total_subs = 0.0;
    for(int i = 0; i < nsubs_bucket; ++i)
    {
        ord_subs.push_back(std::pair<int32_t,int32_t>(sub_count[i], i));
        total_subs += sub_count[i];
    }
    ord_subs.sort(std::greater<std::pair<int32_t,int32_t> >());
    printf("done.\n");
    
    printf("Processing hash2array... ");
    // hash2array nsubs_bucket
    for(int i = 0; i < nwords_bucket; ++i)
        fwrite(&(word2int_[i]), sizeof(int32_t), 1, fd);
    std::vector<int32_t> reverse_sub_map(nsubs_bucket, 0);
    int32_t rit = 0;
    for(auto it = ord_subs.begin(); it != ord_subs.end(); ++it, ++rit)
        reverse_sub_map[it->second] = rit;
    fwrite(reverse_sub_map.data(), sizeof(int32_t), nsubs_bucket, fd);
    printf("done.\n");
    
    printf("Processing words... ");
    for(int i = 0; i < nrwords; ++i)
    {
        std::string w = getWord(i);
        fwrite (w.c_str(), sizeof(char), w.size()+1, fd);
    }
    printf("done.\n");
    
    printf("Processing freq... ");
    // freq
    for(int i = 0; i < nrwords; ++i)
    {
        float freq = words_[i].count/double(ntokens_);
        fwrite(&freq, sizeof(float), 1, fd); 
    }
    for(auto it = ord_subs.begin(); it != ord_subs.end(); ++it)
    {
        float freq = it->first/total_subs;
        fwrite(&freq, sizeof(float), 1, fd);
    }
    printf("done.\n");
    
    printf("Processing top words... ");
    Vector vec(300);
    const real* data = m->data();
    for(int i = 0; i < nrwords; ++i)
    {
        ft.getWordVector(vec, getWord(i));
        std::vector<float> v = transform(n_, vec.data(), map.data(), map.size());
        fwrite(v.data(), sizeof(float), n_, fd);
        // std::vector<float> v2 = transform(n_, &(data[i*n_]), map.data());
        // fwrite(v2.data(), sizeof(float), n_, fd);
    }
    printf("done.\n");
    
    printf("Processing subwords... \n");
    uint8_t val = 0;
    std::vector<float> vmm(nsubs_bucket*2, 0.0);
    int32_t ii = 0;
    for(auto it = ord_subs.begin(); it != ord_subs.end(); ++it, ii++)
    {
        int32_t i = it->second+nwords_;
        if(i > 0 && ii % 100000 == 0)
            printf("%d\n", ii);
        std::vector<float> v = transform(n_, &(data[i*n_]), map.data(), map.size());
        vmm[ii*2] = v[0];
        vmm[ii*2+1] = v[0];
        for(int j = 1; j < n_; ++j)
        {
            if(v[j] < vmm[ii*2])
                vmm[ii*2] = v[j];
            else if(v[j] > vmm[ii*2+1])
                vmm[ii*2+1] = v[j];        
        }
        float step = (vmm[ii*2+1]-vmm[ii*2])/15.0;
        for(int j = 0; j < n_; ++j)
        {
            if(j%2==0)
                val = uint8_t((v[j]-vmm[ii*2])/(step+1e-11) + 0.5);
            else
            {
                val += 16*uint8_t((v[j]-vmm[ii*2])/(step+1e-11) + 0.5);
                fwrite(&(val), sizeof(uint8_t), 1, fd);
            }
        }
    }
    printf("done.\n");

    printf("Processing min/max... ");
    fwrite(vmm.data(), sizeof(float), nsubs_bucket*2, fd);
    printf("done.\n");
    fclose(fd);
}


}
