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
#include <cmath>
#include <grpc++/grpc++.h>

#include "WordVector.grpc.pb.h"


using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using WordVector::DetectLanguagesRequest;
using WordVector::DetectLanguagesReply;




class WordVectorClient {
 public:
  WordVectorClient(std::shared_ptr<Channel> channel)
      : stub_(WordVector::WordVector::NewStub(channel)) {
        
  }

  // Assembles the client's payload, sends it and presents the response back
  // from the server.
  std::string SayHello(const std::string& word1, const std::string& word2) {
    // Data we are sending to the server.
    DetectLanguagesRequest request;
    WordVector::GetVectorsRequest gv_request;
    request.add_texts();
    request.set_texts(0, "Bonjour Robert, comment vas-tu?");
    request.add_texts();
    request.set_texts(1, "Hi Robert, how are you?");
    request.add_texts();
    request.set_texts(2, "a$qwq$$ ad# sd%*fajn laskfd dfskdfs");
    request.add_texts();
    request.set_texts(3, "");
    request.add_texts();
    request.set_texts(3, "Merhaba Robert, nasılsın?");
    gv_request.add_tokens();
    gv_request.set_tokens(0, word1);
    gv_request.add_tokens();
    gv_request.set_tokens(1, word2);

    // Container for the data we expect from the server.
    DetectLanguagesReply reply;
    WordVector::GetVectorsReply gv_reply;

    // Context for the client. It could be used to convey extra information to
    // the server and/or tweak certain RPC behaviors.
    ClientContext context;
    ClientContext gv_context;

    // The actual RPC.
    Status status = stub_->DetectLanguages(&context, request, &reply);
    Status gv_status = stub_->GetVectors(&gv_context, gv_request, &gv_reply);

    // Act upon its status.
    if (status.ok()) 
    {
        for(size_t i = 0; i < reply.results_size(); ++i)
        {
            WordVector::DetectedLanguage lang = reply.results(i);
            printf("%s => %s (%d)\n", lang.text().c_str(), lang.language().c_str(), reply.results_size());
        }
        printf("Call to DetectLanguages OK\n");
    } 
    else 
    {
        std::cout << status.error_code() << ": " << status.error_message()
                  << std::endl;
        printf("RPC failed\n");
    }
    if (gv_status.ok()) 
    {
        printf("frequency %f\n", gv_reply.vectors()[0].frequency());//  .results()[0].text().c_str());
        
        const float* v1 = reinterpret_cast<const float*>(gv_reply.vectors()[0].data().data());
        const float* v2 = reinterpret_cast<const float*>(gv_reply.vectors()[1].data().data());
        double mult = 0.0, sz1 = 0.0, sz2 = 0.0;
        printf(">     ...      ...\n");
        for(int i = 280; i < 300; ++i)
        {
            printf("> %f %f\n", v1[i], v2[i]);
            mult+=v1[i]*v2[i];
            sz1+=v1[i]*v1[i];
            sz2+=v2[i]*v2[i];
        }
        printf("a*b = %f (%f %f)\n", mult/sqrt(sz1*sz2), sz1, sz2);
        printf("Call to GetVectors OK\n");
    } 
    else 
    {
        std::cout << status.error_code() << ": " << status.error_message()
                  << std::endl;
        printf("gv RPC failed\n");
    }
    return std::string("");
  }

 private:
  std::unique_ptr<WordVector::WordVector::Stub> stub_;
};

int main(int argc, char** argv) {
    // Instantiate the client. It requires a channel, out of which the actual RPCs
    // are created. This channel models a connection to an endpoint (in this case,
    // localhost at port 50051). We indicate that the channel isn't authenticated
    // (use of InsecureChannelCredentials()).
    
    std::string word1, word2, url;
    if(argc > 1)
        word1 = argv[1];
    else
        word1 = "hello";
    if(argc > 2)
        word2 = argv[2];
    else
        word2 = "world";
    if(argc > 3)
        url = argv[3];
    else
        url = "localhost:50052";
    WordVectorClient wv(grpc::CreateChannel(url, grpc::InsecureChannelCredentials()));
    std::string reply = wv.SayHello(word1,word2);

  

  return 0;
}
