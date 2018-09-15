# paper

## Recent Trends in Deep Learning BasedNatural Language Processing: [English](https://arxiv.org/pdf/1708.02709.pdf), [Korean](https://ratsgo.github.io/natural%20language%20processing/2017/08/16/deepNLP/)

What is different from Machine learning on NLP to Deep Learning on NLP

    * shallow models(e.g. SVM and logistic regression) -> dense vector representations
    * hand-crafted features ->  multi-level automatic feature


1. Distributed Represntation
    * Word embedding :  
        Measuring similarity between vectors is possible, using measures such as cosine similarity. 
        Word Embedding are often used as the first data processing layer in a deep learning model.
    * Word2vec(Apply the CBOW or Skip-gram)
        
        * Frameworks providing embedding tools and methods
        
          [S-Space(Java)](https://github.com/fozziethebeat/S-Space), 
          [Semanticvectors(Java)](https://github.com/semanticvectors/), 
          [Gensim(Python)](https://radimrehurek.com/gensim/), 
          [Pydsm(Python)](https://github.com/jimmycallin/pydsm/), 
          [Dissect(Python)](http://clic.cimec.unitn.it/composes/toolkit/), 
          [Fasttext(Python)](https://fasttext.cc/)
          
    * Character Embeddings
         
        For tasks such as POS-tagging and NER, intra-word morphological and shape information can also be very useful. A common phenomenon for languages with large vocabularies is the unknown word issue or out-of-vocabulary word(OOV)
issue.

2. Convolution Neural Networks


   A look-up table was used to transform each word into a vector of user-defined dimensions. Thus, an input sequence {s1, s2, ...sn} of n words was transformed into a series of vectors {ws1, ws2, ...wsn} by applying the look-up table to each of its words
 
   * Basic CNN
   
       [for korea](http://docs.likejazz.com/cnn-text-classification-tf/), [source](https://github.com/likejazz/cnn-text-classification-tf), korea paper ([SNU : Large-Scale Text Classification Methodology
with Convolutional Neural Network](https://bi.snu.ac.kr/Publications/Conferences/Domestic/KIISE2015W_JoHY.pdf))
       
      
      * Sentence Modeling
