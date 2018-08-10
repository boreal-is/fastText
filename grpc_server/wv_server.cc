/*
 *
 * Copyright 2015 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <sstream>
#include <grpcpp/grpcpp.h>
#include "../src/fasttext.h"

#include "WordVector.grpc.pb.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using WordVector::DetectLanguagesRequest;
using WordVector::DetectLanguagesReply;
using WordVector::GetVectorsRequest;
using WordVector::GetVectorsReply;

const std::string EOS = "</s>";
const std::string BOW = "<";
const std::string EOW = ">";

struct VectorResult {
   std::vector<float>  data;
   float freq;
   int32_t ndim;
};

class Dictionary
{
  public:
    Dictionary(std::string filename) {load(filename);}
    ~Dictionary() {};

    void load(std::string filename);
    uint32_t hash(std::string& str);
    int32_t find(const char* w, uint32_t h);
    int32_t find(std::string& w);
    void  computeSubwords(std::string word, std::vector<float>& r, float count);
    void addVector(std::vector<float>& v1, std::vector<float>& v2);
    void divVector(std::vector<float>& v, float div);
    std::vector<float> toFloat(uint8_t* v, float min, float max);
    VectorResult getWordInfo(std::string& word);

    int32_t nwords;
    int32_t nrwords;
    int32_t nwords_bucket;
    int32_t nsubs_bucket;
    int32_t ndim;
    int32_t nchars;
    uint8_t maxn;
    uint8_t minn;
    std::vector<char> chars;
    std::vector<char*> words;
    std::vector<int32_t> hash2id;
    std::vector<float> freqs;
    std::vector<float> top_words;
    std::vector<float> mins_maxs;
    std::vector<uint8_t> sub_vecs;
};

Dictionary* dict;
fasttext::FastText* ft;

VectorResult Dictionary::getWordInfo(std::string& word)
{
    //bool isdot = word_[0] == '.';
    //std::string word = isdot ? word_.substr(1) : word_;

    VectorResult ret;
    int32_t idx = hash2id[find(word)];
    ret.ndim = ndim;
    ret.freq = freqs[nrwords-1];
    ret.data.resize(ndim, 0.0);
    //if(!isdot && idx < nrwords)
    if(idx != -1 && idx < nrwords && idx < nrwords+nsubs_bucket)
    {
        ret.freq = freqs[idx];
        //memcpy(ret.data.data(), &(top_words[idx*2*ndim+(isdot ? 1 : 0)]), ndim*sizeof(float));
        memcpy(ret.data.data(), &(top_words[idx*ndim]), ndim*sizeof(float));
        // if(isdot)
        //     computeSubwords(BOW + word + EOW, ret.data, 1.0);
    }
    else 
        computeSubwords(BOW + word + EOW, ret.data, 0.0);
    return ret;
}

void Dictionary::load(std::string filename)
{
    FILE* fd = fopen (filename.c_str(), "rb");
    fread(&nwords, sizeof(int32_t), 1, fd);
    fread(&nrwords, sizeof(int32_t), 1, fd);
    fread(&nwords_bucket, sizeof(int32_t), 1, fd);
    fread(&nsubs_bucket, sizeof(int32_t), 1, fd);
    fread(&ndim, sizeof(int32_t), 1, fd);
    fread(&nchars, sizeof(int32_t), 1, fd);
    fread(&minn, sizeof(uint8_t), 1, fd);
    fread(&maxn, sizeof(uint8_t), 1, fd);
    hash2id.resize(nwords_bucket+nsubs_bucket);
    chars.resize(nchars);
    words.resize(nrwords);
    freqs.resize(nwords_bucket+nsubs_bucket);
    top_words.resize(nrwords*ndim);
    mins_maxs.resize(nsubs_bucket*2);
    sub_vecs.resize(nsubs_bucket*ndim);
    fread(hash2id.data(), sizeof(int32_t), nwords_bucket+nsubs_bucket, fd);
    fread(chars.data(), sizeof(char), nchars, fd);
    
    fread(freqs.data(), sizeof(int32_t), nwords_bucket+nsubs_bucket, fd);
    fread(top_words.data(), sizeof(float), nrwords*ndim, fd);
    fread(sub_vecs.data(), sizeof(uint8_t), nsubs_bucket*ndim/2, fd);
    fread(mins_maxs.data(), sizeof(float), nsubs_bucket*2, fd);
    words[0] = chars.data();
    size_t pos = 1;
    for(size_t i = 1; i < nrwords; ++i)
    {
        while(chars[pos] != '\0')
            ++pos;
        ++pos;
        words[i] = &(chars[pos]);
    }
    
    float size = 0.0;
}

uint32_t Dictionary::hash(std::string& str)  
{
    uint32_t h = 2166136261;
    for (size_t i = 0; i < str.size(); i++) 
    {
        h = h ^ uint32_t(str[i]);
        h = h * 16777619;
    }
    return h;
}

int32_t Dictionary::find(const char* w, uint32_t h) {
    int32_t id = h % nwords_bucket;
    while (hash2id[id] != -1 && hash2id[id] < nrwords && strcmp(words[hash2id[id]], w) != 0)
        id = (id + 1) % nwords_bucket;
    return id;
}

int32_t Dictionary::find(std::string& w) 
{
    return find(w.c_str(), hash(w));
}


void Dictionary::computeSubwords(std::string word, std::vector<float>& r, float count=0.0)
{
    for (size_t i = 0; i < word.size(); i++) 
    {
        std::string ngram;
        if ((word[i] & 0xC0) == 0x80) continue;
        for (size_t j = i, n = 1; j < word.size() && n <= maxn; n++) {
            ngram.push_back(word[j++]);
            while (j < word.size() && (word[j] & 0xC0) == 0x80) {
                ngram.push_back(word[j++]);
            }
            if (n >= minn && !(n == 1 && (i == 0 || j == word.size()))) 
            {
                int32_t h = hash2id[hash(ngram) % nsubs_bucket + nwords_bucket];
                std::vector<float> v = toFloat(&(sub_vecs.data()[h*ndim/2]), mins_maxs[h*2], mins_maxs[h*2+1]);
                addVector(r, v);
                count += 1;                
            }
        }
    }
    divVector(r,count);
}

void Dictionary::addVector(std::vector<float>& v1, std::vector<float>& v2)
{
    for(size_t i = 0; i < ndim; ++i)
        v1[i] += v2[i];
}

void Dictionary::divVector(std::vector<float>& v, float div)
{
    for(size_t i = 0; i < ndim; ++i)
        v[i] /= div;
}

std::vector<float> Dictionary::toFloat(uint8_t* v, float min, float max)
{
    float step = (max-min)/15.0;
    std::vector<float> r(ndim);
    for(size_t i = 0; i < ndim/2; ++i)
    {
        r[2*i] = (v[i]%16)*step+min;
        r[2*i+1] = (v[i]/16)*step+min;
    }
    return r;
}





class WordVectorImpl final : public WordVector::WordVector::Service {
    Status DetectLanguages(ServerContext* context, const DetectLanguagesRequest* request, 
                          DetectLanguagesReply* response) override 
    {
        int32_t ntexts = request->texts_size();
        for(int32_t i = 0; i < ntexts; ++i)
        {
            std::vector<std::pair<float,std::string> > predictions(1);
            std::string s = ""+request->texts(i);
            std::stringstream ss(s);
            ft->predict(ss, 1, predictions, 0.0);
            WordVector::DetectedLanguage* value = response->add_results();
            if(predictions.size() > 0)
                if(predictions[0].second.substr(0,9) == "__label__")
                    value->set_language(predictions[0].second.substr(9));
                else
                    value->set_language(predictions[0].second);
            else
                value->set_language("unk");
            value->set_text(request->texts(i));
        }
        
        return Status::OK;
    }
    
    Status GetVectors(ServerContext* context, const GetVectorsRequest* request, 
                          GetVectorsReply* response) override 
    {
        int32_t ntokens = request->tokens_size();
        //std::vector<float> temp(300,12.23);
        
        for(size_t i = 0; i < ntokens; ++i)
        {
            std::string token = request->tokens(i);
            VectorResult result = dict->getWordInfo(token);
            printf("%s %f %d\n", token.c_str(), result.freq, result.ndim);
            WordVector::Vector* vec = response->add_vectors();
            vec->set_frequency(result.freq);                
            vec->set_data(result.data.data(), result.ndim*sizeof(float));
        }
        
        return Status::OK;
    }
};

void RunServer(std::string url ) {
    
    std::string server_address(url.c_str());
    WordVectorImpl service;

    ServerBuilder builder;
    // Listen on the given address without any authentication mechanism.
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    // Register "service" as the instance through which we'll communicate with
    // clients. In this case it corresponds to an *synchronous* service.
    builder.RegisterService(&service);
    // Finally assemble the server.
    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout << "Server listening on " << server_address << std::endl;

    // Wait for the server to shutdown. Note that some other thread must be
    // responsible for shutting down the server for this call to ever return.
    server->Wait();
}

int main(int argc, char** argv) {

    std::string path = "wiki_data_en.bin";
    std::string url = "localhost:50051";
    std::string lang_model_path = "lid.176.bin";
    if(argc > 1)
        path = argv[1];
    if(argc > 2)
        url = argv[2];
    if(argc > 3)
        lang_model_path = argv[3];
    dict = new Dictionary(path);
    ft = new fasttext::FastText();
    ft->loadModel(lang_model_path);
    RunServer(url);

  return 0;
}
