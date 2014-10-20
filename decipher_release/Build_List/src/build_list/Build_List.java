/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package build_list;

import java.util.*;
/**
 *
 * @author qdou
 * build sorted list for slice sampling
 * 
 */

class pair {
    int key;
    float value;
    public pair(int k,float v) {
        key = k;
        value = v;
    }
}



public class Build_List {
    static HashMap<Integer,Float> prob_dict = new HashMap<Integer,Float>();
    static Comparator<pair> comparator = new Comparator<pair>()
    {
      // This is where the sorting happens.
      public int compare(pair p1, pair p2)
      {
        if(p2.value >= p1.value) {
            return 1;
        } else {
            return 0;
        }
      }
    };    
    
    static ArrayList<pair> start_list = new ArrayList<pair>();
    static ArrayList<pair> end_list = new ArrayList<pair>();
    
    static void genStartList(List<pair> start_list,int[] vocab) {
      for(int key : vocab) {
        float prob = lm.get_ngram_prob(lm.pseu_start, key) + 
                     lm.uni_gram_prob[key][1];
        start_list.add(new pair(key,prob));
      }
      // sort the list
      Collections.sort(start_list,comparator);
    }
    
    static void genEndList(List<pair> start_list,int[] vocab) {
      for(int key : vocab) {
        float prob = lm.get_ngram_prob(key,lm.pseu_end) + 
                     lm.uni_gram_prob[key][0];
        start_list.add(new pair(key,prob));
      }
      // sort the list
      Collections.sort(start_list,comparator);        
    }
    
    static void mergeList(int word, ArrayList<pair> list, 
                          ArrayList<pair> context_list, Set<Integer> context, 
                          int mode, int max) {
        if (context == null) {
          System.err.println(word + " no context ");
        }
        List<pair> final_list = new LinkedList<pair>();
        int list_pos = 0;
        int context_list_pos = 0;
        int count = 0;
        float l_prob = Float.NEGATIVE_INFINITY;
        while(count < max) {
          l_prob = Float.NEGATIVE_INFINITY;
          count++;
          
          while (list_pos < list.size() && context!=null && 
                 context.contains(list.get(list_pos).key)) {
            list_pos++;
          }
          
          if (list_pos < list.size()) {
            int key = list.get(list_pos).key;
            if (mode == 2) {
              l_prob = lm.get_ngram_prob(word,key) + 
                       lm.get_ngram_prob(key, lm.pseu_end); 
            } else {
              l_prob = lm.get_ngram_prob(key,word) + 
                       lm.get_ngram_prob(lm.pseu_start, key);                 
            }
          }
          
          if(context_list_pos < context_list.size()) {
            float context_prob = context_list.get(context_list_pos).value;
            if(context_prob > l_prob) {
              final_list.add(new pair(context_list.get(context_list_pos).key,
                                      context_prob));
              context_list_pos++;
              continue;
            }
          }
          
          if(list_pos < list.size()) {
            final_list.add(new pair(list.get(list_pos).key,l_prob));
            list_pos++;       
          }
          
        }
        
        // print out list
        for(pair p : final_list) {
          if(mode == 2) {
            System.out.println(word + "|</s>" + " " + p.key + " " + p.value);     
          } else {
            System.out.println("<s>|" + word + " " + p.key + " " + p.value);             
          }
        }
    }
    
    // generate sorted list
    static void genContextList(int word, int max) {
      ArrayList<pair> pre_context_list = new ArrayList<pair>();
      if (lm.pre_context.containsKey(word)) {
        for(int key : lm.pre_context.get(word)) {
          float prob = lm.get_ngram_prob(key, word) + 
                       lm.get_ngram_prob(lm.pseu_start, key); 
          pre_context_list.add(new pair(key,prob));
        }
        Collections.sort(pre_context_list,comparator);
      }
      
      ArrayList<pair> post_context_list = new ArrayList<pair>();
      if (lm.post_context.containsKey(word)) {
        for(int key : lm.post_context.get(word)) {
          float prob = lm.get_ngram_prob(word, key) + 
                       lm.get_ngram_prob(key, lm.pseu_end); 
          post_context_list.add(new pair(key,prob));
        }
        Collections.sort(post_context_list,comparator);
      } 
      
      mergeList(word,start_list,pre_context_list,lm.pre_context.get(word),1,max);
      mergeList(word,end_list,post_context_list,lm.post_context.get(word),2,max);
    }
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        // TODO code application logic here
        lm.load_lm(args[0]);
        System.err.println("vocab: " + lm.vocab_size);
        genStartList(start_list,lm.hidden_vocab);
        genEndList(end_list,lm.hidden_vocab);
        System.err.println("loading lm complete");
        System.err.println("pre_context size: " + lm.pre_context.size());
        System.err.println("post_context size: " + lm.post_context.size());
        // print out list
        /*for (pair p : start_list) {
            System.out.println(p.key + " " + p.value);
        }*/
        int counter = 0;
        for(int word : lm.hidden_vocab) {
          genContextList(word,Integer.parseInt(args[1]));
          counter++;
          if(counter%1000 == 0){
            System.err.println(counter + "processed");  
          }
        }
    }
}
