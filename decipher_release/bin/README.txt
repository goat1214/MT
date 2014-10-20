
Command line to invoke the decipherment program:
Java - jar Xmx[memory] Slice_v9_t.jar <Iteration> <Language Model> <Ranked List> <Cipher File> <Seed> <Output Ttable> <Number of Bigram Type> <Numver of Thread>

Example:
java -jar -Xmx18000m Slice_v9_t.jar 2000 *.lm *.dat example.cipher 200 es-en.ttable 1 1


Interation: Number of sampling iterations
Language Model: Language model file in arpa format. Each token should be converted to an integer. The language model file should have suffix .lm_<0,1,2...>. See the example file example.lm_0 for the format.
Ranked List: Presorted list for performaing efficient slice sampling. Can be generated from the language model file automatically using Build_List.jar. The file should have suffix .dat_<0,1,2 ...>. see example.dat_0 for details of the format. 
Cipher File: Cipher file that contains cipher bigrams. Each token has to be an integer. Each line contains a cipher bigram. Each line is separated into 3 columns: count | bigram | type, where the last column is optional.  See example.cipher for details.
Seed: seed for random number generator
Output Ttable: Output file containing result of decipherment <Cipher Token> ||| <Plain Token> ||| <Probability>
Number of Bigram Type: The program supports using different language models to decipher different types of dependency bigrams. Default is 1
Number of Thread: Number of threads to perform sampling

Command line to generate ranked list for sampling:
java Build_List.jar <Language Model> <K>

Language Model: Language model file in arpa format
K: contain top k items for each context. For details refer to "Large Scale Decipherment for Out-of-Domain Machine Translation"
