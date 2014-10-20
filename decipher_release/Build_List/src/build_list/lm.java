/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package build_list;

import java.util.*;
import java.io.*;

/**
 *
 * @author qdou
 */


public class lm {
   
    public static float[][] slice_list=null;
    //public static HashMap<Integer,HashMap<Integer,Float>> bi_gram_prob=new HashMap<Integer,HashMap<Integer,Float>>(); 
    public static HashMap<Integer,Float>[] bi_gram_prob=null; //new HashMap<Integer,HashMap<Integer,Float>>();
    public static HashMap<Integer,Set<Integer>> pre_context = null;
    public static HashMap<Integer,Set<Integer>> post_context = null;    
    public static float[][] uni_gram_prob=null;
    static int[] hidden_vocab=null; 
    static int vocab_size=0;
    //static HashMap<String,Integer> vocab_index=new HashMap<String,Integer>();
    //static HashMap<String,Integer> observed_vocab=new HashMap<String,Integer>();
    public static int pseu_start,pseu_end;
    public static int max_id=0;
    static long mask=Integer.MAX_VALUE>>1;
    
    public static void init_special_ids(int s){
        
        pseu_start=0;
        pseu_end=s;
    }
    
    public static int get_context_id(int w1,int w2){
        return w1+w2;
    }
    
    public static int get_token_id(String token){
        if(token.equals("<s>"))
            return pseu_start;
        else if(token.equals("</s>"))
            return pseu_end;
        else{
            return Integer.parseInt(token);
        }
    }
    
    public static float get_ngram_prob(int id1,int id2){
       try{
               return bi_gram_prob[id1].get(id2);
       }catch(NullPointerException e){
           return uni_gram_prob[id2][0]+uni_gram_prob[id1][1];
       }
       
          
       //   return ngram_prob.get(ngram & mask)[0]+ngram_prob.get((ngram>>30)&mask)[1];  
       
    }
    
    /*public static float get_ngram_prob_unk(int id1,int id2){

       try{
           return bi_gram_prob[id1].get(id2);
       }catch(Exception e){
           try{
               return uni_gram_prob[id2][0]+uni_gram_prob[id1][1];
           }catch(Exception e2){
               return -5;
           }
       }
    }*/
    
    // more compact version of load_ngram
    public static void load_slice_list(String file){
        String line="";
        try{
          BufferedReader in =new BufferedReader(new FileReader(file));
          // allocate memory space for the list
          int max=2000;
          int rank=0;
          slice_list=new float[2*pseu_end][2*max];
          
          
          int context=0;
          while((line=in.readLine())!=null){ // context word probability
             String[] entries=line.split(" ");
             if (entries.length==1){
                 max=Integer.parseInt(entries[0]);                 
                 continue;
             }
             //entries[0]=entries[0].replaceAll("\\|\\|\\|", "\\|");
             
             //System.out.println(get_token_id(tokens[0])+" "+get_token_id(tokens[1]));
              String[] tokens=entries[0].split("\\|");
              context=get_context_id(get_token_id(tokens[0]),get_token_id(tokens[1]));            
              //if(slice_list[context]==null)
              //   slice_list[context]=new float[max*2];
             
             //String[] ctks=entries[0].split("--");
             //System.out.println(entries[0]+" "+(lm.get_ngram_prob(ctks[0]+" "+entries[1])
             //   +lm.get_ngram_prob(entries[1]+" "+ctks[1])));
             float prob=(float)Math.pow(10,lm.get_ngram_prob(get_token_id(tokens[0]),get_token_id(entries[1]))
                +lm.get_ngram_prob(get_token_id(entries[1]),get_token_id(tokens[1])));
            
             //float prob=Float.parseFloat(entries[2]);
             slice_list[context][rank]=get_token_id(entries[1]);
             slice_list[context][rank+1]=prob;
                 
             rank+=2;
             if(rank==max*2)
                rank=0;
          } 
          /*for(gram_prob g: dict.get("v_q")){
            System.out.println(g.word+" "+g.prob);
          }*/
        }catch(Exception e){
            System.out.println(e+" "+line);
        }        
    }
    
   
    public static void get_vocab_size(String file){
          try{
                        
          BufferedReader in =new BufferedReader(new FileReader(file));
          String line="";
          while((line=in.readLine())!=null){
             if(line.contains("\t")){
               String[] entries=line.split("\t");
               String[] tokens=entries[1].split(" ");
               if(tokens.length==1 && !tokens[0].equals("<s>") && !tokens[0].equals("</s>")
                       && !tokens[0].equals("START") && !tokens[0].equals("END")){
                   int id=Integer.parseInt(tokens[0]);
                   vocab_size++;
                   if (id>max_id)
                       max_id=id;
               }else if (tokens.length==2)
                   break;
               
             }                   
          } 
          init_special_ids(max_id+1);
          System.err.println("max_id "+max_id);
          //System.out.println(hidden_vocab.size());
        }catch(Exception e){
            System.out.println(e);
        }        
        
    }
    
    public static void load_lm(String file){
          // get vocab size first         
          get_vocab_size(file);
                        
          try{
          // allocate memory              
          uni_gram_prob=new float[pseu_end+1][2];          
          bi_gram_prob=new HashMap[pseu_end];
          hidden_vocab=new int[vocab_size];
          pre_context = new HashMap<Integer,Set<Integer>>();
          post_context = new HashMap<Integer,Set<Integer>>();
          
          BufferedReader in =new BufferedReader(new FileReader(file));
          String line="";
          int counter=0;
          while((line=in.readLine())!=null){
             if(line.contains("\t")){
               String[] entries=line.split("\t");
               String[] tokens=entries[1].split(" ");
               if(tokens.length==1 && !tokens[0].equals("<s>") && !tokens[0].equals("</s>")
                       && !tokens[0].equals("START") && !tokens[0].equals("END")){                   
                   hidden_vocab[counter++]=get_token_id(tokens[0]);
               }
               
               if (tokens.length==2 && !entries[1].contains("s>")) { // keep track of word context
                      int id1=get_token_id(tokens[0]);
                      int id2=get_token_id(tokens[1]);
                      if (!pre_context.containsKey(id2)) {
                        pre_context.put(id2, new HashSet<Integer>());
                      } 
                      if (!post_context.containsKey(id1)) {
                        post_context.put(id1, new HashSet<Integer>());
                      }     
                      pre_context.get(id2).add(id1);
                      post_context.get(id1).add(id2);
               }
               if(entries.length==2){
                  if(tokens.length==2){
                    int id1=get_token_id(tokens[0]);
                    int id2=get_token_id(tokens[1]);
                    if(bi_gram_prob[id1]==null)
                      bi_gram_prob[id1]=new HashMap<Integer,Float>();
                    bi_gram_prob[id1].put(id2, Float.parseFloat(entries[0]));

                  }else{
                   uni_gram_prob[get_token_id(tokens[0])][0]=Float.parseFloat(entries[0]);
                  }
                                                                  
               }               
               else if(entries.length==3){
                  int id = get_token_id(entries[1]);
                  uni_gram_prob[id][0]=Float.parseFloat(entries[0]);
                  uni_gram_prob[id][1]=Float.parseFloat(entries[2]);
                        
               }
               else
                  System.out.println("file format error");
             }   
             
          }
          //long allocatedMemory = Runtime.getRuntime().totalMemory();
          //System.out.println("allocated :"+allocatedMemory/1024/1024);
          //System.out.println(hidden_vocab.length+" "+vocab_size);
        }catch(Exception e){
            System.out.println(e);
        }

    }
      
}
