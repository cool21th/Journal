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
      
         Sequential convolutions help in improved mining of the sentence to grasp a truly abstract representations comprising rich semantic information. 

      * window approach
         
         many NLP tasks, such as NER, POS tagging, and SRL, require word-based predictions
         To adapt CNNsfor such tasks, a window approach is used, which assumes that the tag of a word primarily depends on its neighboring words.
         The ultimate goal of word-level classification is generally to assign a sequence of labels to the entire sentence. 


   * Applications
   
      This simple network, however, had many shortcomings with the CNN’s inability to model long distance dependencies standing as the main issue
      
      [Kalchbrenner et al. [44]](http://www.aclweb.org/anthology/P14-1062), who published a prominent paper where they proposed a dynamic convolutional neural network (DCNN) for semantic modeling of sentences.
      
      Tasks involving sentiment analysis also require effective extraction of aspects along with their sentiment polarities. [Ruder et al.](http://www.aclweb.org/anthology/S16-1053) applied a CNN where in the input they concatenated an aspect vector with the word embeddings to get competitive results.
      
      [Denil et al.](https://arxiv.org/pdf/1406.3830.pdf) applied DCNN to map meanings of words that constitute a sentence to that of documents for summarization.
      
      In the domain of QA, [Yih et al.]((http://www.iro.umontreal.ca/~lisa/pointeurs/ir0895-he-2.pdf)) proposed to measure the semantic similarity between a question and entries in a knowledge base (KB) to determine what supporting fact in the KB to look for when answering a question
      
      [Dong et al.](http://www.aclweb.org/anthology/P15-1026) introduced a multi-column CNN (MCCNN) to analyze and understand questions from multiple aspects and create their representations.
      
      [Severyn and Moschitti](https://arxiv.org/pdf/1604.01178.pdf) also used CNN network to model optimal representations of question and answer sentences.
      
      [Chen et al.](https://pdfs.semanticscholar.org/ca70/480f908ec60438e91a914c1075b9954e7834.pdf) proposed a modified pooling strategy: dynamic multi-pooling CNN (DMCNN) To overcome this loss of information for multiple-event modeling because traditional max-pooling strategies perform this in a translation invariant form
      
      Speech recognition also requires such invariance and, thus, [Abdel-Hamid et al.](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/CNN_ASLPTrans2-14.pdf?ranMID=24542&ranEAID=je6NUbpObpQ&ranSiteID=je6NUbpObpQ-I48RjjENmS.uCPbhVqwKhw&epi=je6NUbpObpQ-I48RjjENmS.uCPbhVqwKhw&irgwc=1&OCID=AID681541_aff_7593_1243925&tduid=(ir_d32d3c05N7d3d1b2f5d8a9996762b938b)(7593)(1243925)(je6NUbpObpQ-I48RjjENmS.uCPbhVqwKhw)()&irclickid=d32d3c05N7d3d1b2f5d8a9996762b938b) used a hybrid CNN-HMM model which provided invariance to frequency shifts along the frequency axis.
      
      [Palaz et al.](https://ronan.collobert.com/pub/matos/2015_cnnspeech_interspeech.pdf) performed extensive analysis of CNN-based speech recognition systems when given raw speech as input.
      
       [Tu et al.](http://www.aclweb.org/anthology/P15-2088) addressed this task by considering both the semantic similarity of the translation pair and their respective contexts.
       
       Overall, CNNs are extremely effective in mining semantic clues in contextual windows. However, they are very data heavy
models. 
      
      
      
      
      
       

      
      
