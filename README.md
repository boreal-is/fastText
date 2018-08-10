## Table of contents

* [Introduction](#introduction)
* [Approach](#approach)
* [Approach](#Installation and usage)
* [License](#license)

## Introduction

[fastText](https://fasttext.cc/) is a library for efficient learning of word representations and sentence classification. This fork provides a way to minimize the model size without loosing too much information about the vectors. The small memory footprint enables to use fasttext word to vector conversion and language identification as a service on a small server. A grpc client is also included to build such a service.

## Approach

The size of kept words can be limited (by default to 50000). All the subwords are kept but they are replaced by a representation where we kept the minimum and maximum value of the vector and where all the other component are represented as the range between min and max on 4 bits. According to our experience, the incurred lost of precision seems to have minimal practical impacts.

## Installation and usage
Install the modified fasttext
```
$ cd fastText
$ make
```

To generate the binary file used in the grpc server :
```
$ ./fasttext generate-compact ../fasttext_format_reader/wiki.fr.bin
```

A prior transformation can also be applied on the vector :
```
$ ./fasttext generate-compact ../fasttext_format_reader/wiki.fr.bin ../fasttext_format_reader/transformation.bin
```
Where the file transformation.bin is raw float buffer of a (dim x dim)-matrix and dim is the dimension of the word vector.

To build the grpc service :
```
$ cd grpc_server
$ make
```

And to run the service without DetectLanguages:
```
$ ./wv_server ../wiki_data.bin 192.168.2.46:50052
```
Where wiki_data.bin is the generated binary file in the first step.

To run the service with DetectLanguages:
```
$ ./wv_server ../wiki_data_fr.bin 192.168.2.46:50052 lid.176.bin
```
Where lid.176.bin can be download at [https://s3-us-west-1.amazonaws.com/fasttext-vectors/supervised_models/lid.176.bin].

To try the service with the provided client execute:
```
$ ./wv_client salut vieux 192.168.2.46:50052
```

## License

fastText is BSD-licensed. We also provide an additional patent grant.
