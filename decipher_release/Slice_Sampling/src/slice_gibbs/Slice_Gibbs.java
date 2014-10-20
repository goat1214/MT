/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package slice_gibbs;

import java.util.*;
import java.util.concurrent.*;
import java.io.*;

/**
 *
 * @author qdou
 */

class example{
    int weight;
    int type;

    ArrayList<Integer> sentence;
    ArrayList<Integer> sample;

    public example(ArrayList<Integer> s,int w,int t){
        weight=w;
        sentence=s;
        type=t;
        sample=new ArrayList<Integer>();
      
    }
            
}

class trans_table{
    HashMap<Integer,Integer> members; // to quickly check if a given word is in the list
    ArrayList<Integer> id_word_map; // for quickly retrieve word
    
    public trans_table()
    {
        members=new HashMap<Integer,Integer>();
        id_word_map=new ArrayList<Integer>();
    }
}

class Candidate{
   int hidden;
   int count;
   float prob;
   public Candidate(int h,float p,int c){
       hidden=h;
       count=c;
       prob=p;
   }
}

class CandidateComparator implements Comparator<Candidate>{
    public int compare(Candidate c1,Candidate c2){
        if(c1.prob<c2.prob){
           return 1;
        }else if(c1.prob>c2.prob){
           return -1;
        }else{
           return 0;
        }
        
    }
}


public class Slice_Gibbs extends Thread{
    static ArrayList<example> corpus;
    static ConcurrentHashMap<Long,Integer> counts; // the cache
    static ConcurrentHashMap<Long,Long> final_counts; // store counts from last 100 iterations 
    static HashMap<Long,Integer> pseudo_counts; // pseudo_counts
    static HashMap<Long,Integer> o_counts;
    static HashMap<Long,Integer> pseudo_o_counts; // pseudo_o_counts
    static HashMap<Integer,Integer> observ_vocab;
    static HashMap<Integer,trans_table> Channel_List;
    static float sample_prob;
    static float ngram_prob;
    static int iter; // iterations to run
    Random dice;
    static Random global_dice;
    static long seed=0;
    static float prior=0;
    static float alpha=1.0f;
    double total_trial_count=0;
    float step_size=0;
    static int[] sample_num_dist;
    static double target_corpus_probability=0;
    //static int pseu_start,pseu_end;
    static CandidateComparator cmp=new CandidateComparator();
    static HashMap<Integer,ArrayList<Candidate>> ptable;
    static boolean burned_in=false;
    static float max_probability=Float.NEGATIVE_INFINITY;
    static lm[] lms=null; // language models
    static lm static_lm; // used for initialization
    static boolean is_viterbi=false;
    
    static int maxid;    
    int start_pos,end_pos; // define range of data to operate
    lm LM; // current LM for each thread
    // synchronization parts
    static Object cipher_locks[];
    static Object plain_locks[];    
    static Candidate lattice[][] = new Candidate[20][2]; 
    
    public long concat_numbers(int w1,int w2){
        long context=((long)w1)<<20 | ((long)w2);
        return context;
    }

    public static long concat_numbers_static(int w1,int w2){
        long context=((long)w1)<<20 | ((long)w2);
        return context;
    }
        
    // shuffle corpus based on thread count
    /*public static void shuffle(int t_count) {
        ArrayList<example> tmp_corpus=new ArrayList<example>();
        int block_size = corpus.size()/t_count;
        int start_pos = 0;
        int start_pos_list[]=new int[t_count]; // keep starting point of each thread
        for(int i=0;i<t_count;i++){
          start_pos_list[i]=start_pos;    
          start_pos += block_size;          
        }      
        // start shuffling
        for(int i=0;i<block_size;i++) {
          for(int j=0;j<t_count;j++) {
             tmp_corpus.add(corpus.get(start_pos_list[j]+i));               
          }  
        }
        // append any residual to end
        for(int i=block_size*t_count;i<corpus.size();i++){
            tmp_corpus.add(corpus.get(i));
        }
        corpus.clear();
        corpus= tmp_corpus;
        System.out.println("Corpus size after shuffling " + corpus.size());
    }*/
    public static void shuffle() {
        int n = corpus.size();
        Random random = new Random();
        for (int i=0;i<n;i++){
            int change=i+random.nextInt(n-i);
            example buffer=corpus.get(i);
            corpus.set(i, corpus.get(change));
            corpus.set(change, buffer);
        }
    }
    
    public static void load_ptable(String file){
       ptable=new HashMap<Integer,ArrayList<Candidate>>();
       String line="";
        try{
          BufferedReader in =new BufferedReader(new FileReader(file));
         
          while((line=in.readLine())!=null){ // context word probability
             String[] entries=line.split(" \\|\\|\\| ");
             String[] scores=entries[2].split(" ");
             //System.out.println(get_token_id(tokens[0])+" "+get_token_id(tokens[1]));
             Integer plain=Integer.parseInt(entries[1]);
             Integer cipher=Integer.parseInt(entries[0]);
             float score=Float.parseFloat(scores[1]);//+Float.parseFloat(scores[0]);
             int count=0;//Integer.parseInt(scores[2]);
             
             if(!ptable.containsKey(cipher))
               ptable.put(cipher, new ArrayList<Candidate>()); 
                 // ptable.put(cipher, plain);    
             ptable.get(cipher).add(new Candidate(plain,score,0));
             
          } 

        }catch(Exception e){
            System.out.println(e+" "+line);
        }
        for(Integer key:ptable.keySet()){
            ArrayList<Candidate> options=ptable.get(key);
            Collections.sort(options,cmp);
            /*for(int i = 0; i < 20 && i < options.size(); i++){
              System.out.println(key+" "+options.get(i).hidden+ " "+options.get(i).prob+" "+options.size());
            }*/
        }
      
    }
    
    
    
    float get_channel_prob(Integer hidden, Integer observe){
           Integer joint_count=0;
           Integer condition_count=0;
           Long joint=concat_numbers(hidden, observe); 
           Long condition=hidden.longValue();
           
           joint_count=counts.get(joint);
           condition_count=counts.get(condition);
           
           if(joint_count == null)
               joint_count=0;
           if(condition_count == null)
               condition_count=0;         
           return (alpha*prior+joint_count)/(alpha+condition_count);        
    }
    
    // use doubling to search for candidates
    int search_range(float[] candidates,
               float threshold, float d_prior){
        int location=2;
        int size=candidates.length;
        /*for(int i=0;i<size;i++){
            System.out.println(candidates.get(i).word+" "+candidates.get(i).prob);
        }*/
       
        
        while(location<=size && candidates[location-1]*d_prior>threshold){
            location+=4;
        }
        if(location>size){

            return size/2;
        }
        else{
            //System.out.println("cand prob: "+candidates[location-1]+"threshold: "+threshold);
            return (location-2)>>1;
        }
    }
    
    static int[] viterbi_sequence(example e){
        int[] sequence=new int[2];
        if(ptable==null||!ptable.containsKey(e.sentence.get(0))||!ptable.containsKey(e.sentence.get(1)))
        {
           int pos=0;
           for(Integer word : e.sentence){
             int hidden=0;
             if(ptable!=null && ptable.containsKey(word)){
               hidden=ptable.get(word).get(0).hidden;
               //hidden=ptable.get(word).get(Math.abs(global_dice.nextInt(ptable.get(word).size()))).hidden; 
               if(hidden>=static_lm.uni_gram_prob.length||static_lm.uni_gram_prob[hidden][0]==-10.0f){
                 int index=Math.abs(global_dice.nextInt(static_lm.hidden_vocab.length));  
                 hidden=static_lm.hidden_vocab[index];              
               }
             }else{
               int index=Math.abs(global_dice.nextInt(static_lm.hidden_vocab.length));   
               hidden=static_lm.hidden_vocab[index];
             }
             sequence[pos++]=hidden;
           }
        }else{ // find the viterbi sequence
           is_viterbi=true;
           //System.out.println("looking for viterbi sequence");
           ArrayList<Candidate> options=ptable.get(e.sentence.get(0));
           int i;
           for(i=0;i<options.size()&&i<20;i++){
              lattice[i][0].hidden=options.get(i).hidden;
              lattice[i][0].prob=(float)Math.log10(options.get(i).prob)+static_lm.get_ngram_prob(static_lm.pseu_start,options.get(i).hidden);              
           }
           options=ptable.get(e.sentence.get(1));     
           for(int j=0;j<options.size()&&j<20;j++){
              lattice[j][1].hidden=options.get(j).hidden;
              lattice[j][1].prob=Float.NEGATIVE_INFINITY;

              for(int k=0;k<i;k++){
                
                float prob=lattice[k][0].prob
                           +(float)Math.log10(options.get(j).prob)
                           +static_lm.get_ngram_prob(lattice[k][0].hidden,options.get(j).hidden)
                           +static_lm.get_ngram_prob(options.get(j).hidden,static_lm.pseu_end);
                if(prob>lattice[j][1].prob){
                  lattice[j][1].prob=prob;
                  lattice[j][1].count=k;
                }
              }
           }
           
           // retrive the sequence
           float best_prob=Float.NEGATIVE_INFINITY;
           for(int j=0;j<options.size()&&j<20;j++){
             if(lattice[j][1].prob>best_prob){
                best_prob=lattice[j][1].prob;
                sequence[1]=lattice[j][1].hidden;
                sequence[0]=lattice[lattice[j][1].count][0].hidden;
             }
           }
           //System.out.println(sequence[0] + " " + sequence[1]);           
        }
        

        return sequence;
    }
    
    
    static void init_sample(example e){
        Integer pre_hidden=static_lm.pseu_start;
        Integer hidden=0;
        Integer pre_obsv=static_lm.pseu_start;      
        
        is_viterbi=false;
        int [] sequence=viterbi_sequence(e);
        
        //System.out.println("source: "+e.sentence.get(0)+" "+e.sentence.get(1));
        //System.out.println(sequence[0]+" "+sequence[1]);
        
        int pos=0;
        for(Integer word : e.sentence){
           //pick a random word from vocab  
           //System.out.println(pre_obsv+" "+word);
           target_corpus_probability+=static_lm.get_ngram_prob_unk(pre_obsv, word);
           
           pre_obsv=word;

           hidden=sequence[pos];
           if(hidden>=static_lm.uni_gram_prob.length||static_lm.uni_gram_prob[hidden][0]==-10.0f){
               int index=Math.abs(global_dice.nextInt(static_lm.hidden_vocab.length));  
               hidden=static_lm.hidden_vocab[index];
              
           }
           /*
           if(ptable!=null && ptable.containsKey(word)){
             hidden=ptable.get(word).get(0).hidden; 

             if(hidden>=static_lm.uni_gram_prob.length||static_lm.uni_gram_prob[hidden][0]==-10.0f){
               int index=Math.abs(global_dice.nextInt(static_lm.hidden_vocab.length));        
               hidden=static_lm.hidden_vocab[index];
              
             }
           }else{
             int index=Math.abs(global_dice.nextInt(static_lm.hidden_vocab.length));        
             hidden=static_lm.hidden_vocab[index];
           }
           */
           
           e.sample.add(hidden);
           
           //String ngram=pre_hidden+" "+hidden;           
           
           // add observed word to observ_vocab
           observ_vocab.put(word, 1); 
           // increase counts
           Long[] contents={(long)hidden,concat_numbers_static(hidden, word)};
           for(Long x: contents){
             if(counts.containsKey(x)){
                 counts.put(x, counts.get(x)+e.weight);  
                 //System.out.println(x+" "+counts.get(x));
             }
             else{
                 counts.put(x, e.weight);
                 //System.out.println(x+" "+counts.get(x));
             }
             
             /*if(is_viterbi&&hidden==sequence[pos]){ // add psudo counts here
                pseudo_counts.put(x, counts.get(x));  
                pseudo_o_counts.put((long)word,o_counts.get((long)word));
             }*/
           }
           
           if(!Channel_List.containsKey(word)){               
               Channel_List.put(word, new trans_table());
           }
           trans_table table=Channel_List.get(word);
           if(!table.members.containsKey(hidden)){
             table.members.put(hidden, 1);
             table.id_word_map.add(hidden);
           }
           
           pre_hidden=hidden;
           pos++;
        }
        target_corpus_probability+=static_lm.get_ngram_prob_unk(pre_obsv, static_lm.pseu_end); 
        
        // also add the end probability
        //sample_prob+=lm.lm_map.get(hidden+" </s>");
        //ngram_prob+=lm.lm_map.get(hidden+" </s>");
        //System.out.println(e.sample+" "+e.sample_prob);
    }
    
    public static void add_pseudo_count(){
                  // add psuedo count
          for(Long key:pseudo_counts.keySet()){
            int count=pseudo_counts.get(key);
           
            counts.put(key,counts.get(key)+(int)(count*0.5));
          }
          
          for(Long key:pseudo_o_counts.keySet()){
            int count=pseudo_o_counts.get(key);
            o_counts.put(key,o_counts.get(key)+(int)(count*0.5));
            
          }   
          System.out.println("added "+pseudo_counts.size()+ " pseudo counts");          
          pseudo_counts.clear();
          pseudo_o_counts.clear();

    }
    
    public static void load_corpus(String file){
        corpus=new ArrayList<example>();

        o_counts=new HashMap<Long,Integer>();
        pseudo_o_counts=new HashMap<Long,Integer>();
        pseudo_counts=new HashMap<Long,Integer>();
        observ_vocab=new HashMap<Integer,Integer>();
        Channel_List = new HashMap<Integer,trans_table>();
        ngram_prob=sample_prob=0;
        global_dice=new Random(seed);
        maxid=0;
      
        for(int i=0;i<20;i++){
            lattice[i][0]=new Candidate(0,0.0f,0);
            lattice[i][1]=new Candidate(0,0.0f,0);
        }
        String line="";
        try{
          BufferedReader in =new BufferedReader(new FileReader(file));

          while((line=in.readLine())!=null){ // context word probability
               String[] entries=line.replaceAll("\"", "").split("\t"); 
               int weight=Integer.parseInt(entries[0]);
               int type=0;

               if(entries.length==3)
                 type=Integer.parseInt(entries[2]);
              
               static_lm=lms[type];
               //System.out.println("lm: "+type);
             
               String[] cipher_tokens=entries[1].split(" ");
               ArrayList<Integer> tmp_list=new ArrayList<Integer>();
               for(String token:cipher_tokens){
                   int i_token=static_lm.get_token_id(token);
                   if(i_token > maxid) {
                       maxid = i_token;
                   }
                   tmp_list.add(i_token);
                   // add observed freq
                   if(!o_counts.containsKey((long)i_token))
                       o_counts.put((long)i_token, weight);
                   else
                       o_counts.put((long)i_token, o_counts.get((long)i_token) +weight); 
               }
               example e=new example(tmp_list,weight,type);
               
               init_sample(e);
               corpus.add(e);
          }
          
          // intialize locks
          cipher_locks = new Object[maxid+1];
          for(int i = 0;i < cipher_locks.length; i++) {
             cipher_locks[i] = new Object();
          }
          int plain_locks_num=0;
          for(lm x:lms){
            if(x.uni_gram_prob.length>plain_locks_num)
                plain_locks_num=x.uni_gram_prob.length;
          }
          plain_locks = new Object[plain_locks_num];
          for(int i = 0; i < plain_locks.length; i++) {
             plain_locks[i] = new Object();
          }          
          
          System.out.println("loaded "+corpus.size()+" sentence");
          prior=1.0f/observ_vocab.size();
          System.out.println("Observe Vocab: "+observ_vocab.size());
          System.out.println("Target Corpus Prob: "+target_corpus_probability);
          //add_pseudo_count();  
          ptable=null;
          System.out.println(counts.size() +" " + o_counts.size() + " "+Channel_List.size());
          
          //System.exit(0);
        }catch(Exception e){
            System.out.println("corpus "+e);
        }        
    }
   
    
    int get_candidates(float y,int pre_hidden,int h,int post_hidden,int o,float current_prob){

          
          int context=lm.get_context_id(pre_hidden,post_hidden);
          //String context=pre_hidden+"--"+post_hidden;
          float candidate_score=1;
      
          step_size+=1;
          trans_table exist_trans=Channel_List.get(o);
    
          // try_count=0;              
              float[] candidates=LM.slice_list[context];
              int dict_size=candidates.length;
              float d_prior=(alpha*prior+0)/(alpha+0);
              
              int i=0;
        
              if (candidates[dict_size-1]*d_prior<y){  
                  int direction=dice.nextInt(2);
                  int range1=search_range(candidates,y,d_prior);      
                  //System.out.println(range1);
                 
                  //System.out.println(range1+" h:"+h+" o:"+o);
                  
                  int range2=exist_trans.id_word_map.size();
                  int range=range1+range2; // range is a combination of two sources, range could change during itration 
                  
                  while(true){ // find candidate randomly
                    
                      total_trial_count+=1;
                      if(direction==0){
                        i=dice.nextInt(range);
                      } else {
                        i=range-1-dice.nextInt(range);
                      }
                      
                      
                      if(i<range1){//try to retrieve from ngram list
                           //gram_prob tmp_cand=candidates.get(i);
                           int location=i<<1;
                           int new_hidden=(int)candidates[location];
                          
                           
                          
                           float cand_prob=candidates[location+1];
                           if(exist_trans.members.containsKey(new_hidden)){
                             continue;
                           }

                            
                           float channel= get_channel_prob(new_hidden,o);
                           candidate_score=cand_prob*channel;
                           if(candidate_score>y){                           
                              exist_trans.members.put(new_hidden, 1);
                              exist_trans.id_word_map.add(new_hidden); 
                              return new_hidden;                 
                           }                            
                        
                      }
                      else{ // try to retrieve from channel list
                         i-=range1;
                         int new_hidden=exist_trans.id_word_map.get(i);
                         if(new_hidden==h){
                            return new_hidden;
                         }
                         float channel= get_channel_prob(new_hidden,o);
                         candidate_score=(float)Math.pow(10,(LM.get_ngram_prob(pre_hidden,new_hidden)
                          +LM.get_ngram_prob(new_hidden,post_hidden)))*channel;

                         
                         if(candidate_score>y)
                            return new_hidden;
                         
                         if(!counts.containsKey(concat_numbers(new_hidden,o))){
                             exist_trans.members.remove(new_hidden);
                             exist_trans.id_word_map.remove(i);
                             range-=1;
                         }
                      }
                 }                  
              }
          
          
          int range=LM.hidden_vocab.length;
          
          while(true){ // find candidate randomly
            
              total_trial_count+=1;            
              i=dice.nextInt(range);
              int new_hidden=LM.hidden_vocab[i];
              if(h==new_hidden){
                  return new_hidden;
              }
              float channel= get_channel_prob(new_hidden,o);
              candidate_score=(float)Math.pow(10,(LM.get_ngram_prob(pre_hidden,new_hidden)
                +LM.get_ngram_prob(new_hidden,post_hidden)))*channel;
              if(candidate_score>y || h==new_hidden){
                   if(!exist_trans.members.containsKey(new_hidden)){
                         exist_trans.members.put(new_hidden, 1);
                         exist_trans.id_word_map.add(new_hidden);
                   }             
                  return new_hidden;              
              }

          }
         
    }
    
    int slice_gibbs_draw(int j, example e,int h,int o){
        //System.out.println("sampling "+ e.sample);
        
        int hidden=0,pre_hidden=0,post_hidden;
        int current_hidden=h;
   
        if (j==0) // beginning of sequence
           pre_hidden=LM.pseu_start;
        else
           pre_hidden=e.sample.get(j-1);
   
        if (j==e.sample.size()-1)
           post_hidden=LM.pseu_end;
        else
           post_hidden=e.sample.get(j+1);
        
        //double prob=ngram_prob;
        float current_ngram_prob=(float)Math.pow(10,(LM.get_ngram_prob(pre_hidden, current_hidden)
        +LM.get_ngram_prob(current_hidden,post_hidden))); 
        float old_channel=get_channel_prob(h,o);
        //(alpha*prior+counts.get(concat_numbers(h,o)))/
         //           (alpha+counts.get((long)h));
        // auxiliary variable
        //float random_num=dice.nextFloat();  
        double random_num=dice.nextDouble(); 
        float threshold=(float)(random_num*current_ngram_prob*old_channel);
       
        //System.out.println("t:"+random_num+" ngram_prob:"+current_ngram_prob+" old_channel:"+ old_channel);
        //System.out.println(o+" "+h+" T:"+threshold+" P:"+old_channel);
        return get_candidates(threshold,pre_hidden,h,post_hidden,o,current_ngram_prob);  
        //return h;
    }
    

 

    
    @Override
    public void run(){
        System.out.println("Thread "+this.getId()+" starting");
        
        dice=new Random(seed);
        
        float corpus_probability=0; 
        float old_cp=Float.NEGATIVE_INFINITY;
        boolean burn_in = false;
        for(int i=0;i<iter;i++){ // gibbs iteration
            corpus_probability=0;
            total_trial_count=0;
            step_size=0;
            if(!burn_in) {
                if(iter - i <= 100) {
                    burn_in = true;
                }
            }
            for(int k=start_pos;k<end_pos;k++){ // for each example
                example e=corpus.get(k);
                int weight=e.weight;
                LM = lms[e.type];
                  
                int pre_hidden=LM.pseu_start;
                for(int j=0;j<e.sentence.size();j++){ // for each word in the sequence
                    // remove old counts
                    int old_hidden=e.sample.get(j);
                    int old_observe=e.sentence.get(j);
                    int hidden = 0;
                    long trans_pair=concat_numbers(old_hidden,old_observe);
                    
                    synchronized(cipher_locks[old_observe]) {   
                      counts.put(trans_pair, counts.get(trans_pair)-weight);
                      
                      if(counts.get(trans_pair)==0)
                          counts.remove(trans_pair);
                      
                      synchronized(plain_locks[old_hidden]) {  
                        counts.put((long)old_hidden, counts.get((long)old_hidden)-weight);
                        if(counts.get((long)old_hidden)==0)
                          counts.remove((long)old_hidden);                                       
                      
                        // collect samples after burn in
                        /*if(burn_in) {
                            if(final_counts.containsKey(trans_pair)) {
                                final_counts.put(trans_pair, final_counts.get(trans_pair) + weight);
                            }else {
                                final_counts.put(trans_pair, (long)weight);
                            }
                            long h = (long)old_hidden;
                            if(final_counts.containsKey(h)) {
                                final_counts.put(h, final_counts.get(h) + weight);
                            }else {
                                final_counts.put(h, (long)weight);
                            }
                        }*/
                
                        hidden=slice_gibbs_draw(j,e,old_hidden,old_observe);
                        
                      }             
                      // update sample
                      // System.out.println("old sample: "+e.sample);
                      e.sample.set(j,hidden);
                      if(i%500==0){
                        corpus_probability += LM.get_ngram_prob(pre_hidden, old_hidden); 
                        corpus_probability += Math.log10(get_channel_prob(old_hidden,old_observe));
                      }
                      pre_hidden=hidden;
                     // System.out.println("new sample: "+e.sample);
                      // update counts
                      long new_hidden = hidden;
                      long new_pair = concat_numbers(hidden, old_observe);
                      
                      if(counts.containsKey(new_pair))
                        counts.put(new_pair, counts.get(new_pair)+e.weight);
                      else
                        counts.put(new_pair,e.weight);               
                     
                      synchronized(plain_locks[(int)new_hidden]) {
                        if(counts.containsKey(new_hidden))
                            counts.put(new_hidden, counts.get(new_hidden)+e.weight);
                        else
                            counts.put(new_hidden,e.weight);
                      }  
                    }// synce on cipher
                }
                if(i%500==0)   
                  corpus_probability += LM.get_ngram_prob(pre_hidden, LM.pseu_end);
                
            }
                        
            if(i%500==0){
              /*if(this.getId()==8){
                
                print_translation_table("viterbi50k.ptable."+seed);
              }*/
              System.out.println(i+" "+corpus_probability+" avg cand:"+total_trial_count/step_size);
            }
        }
        
    }
    
    // print out samples for debugging
    public static void print_samples(String file){
        try{
           PrintStream out=new PrintStream(new File(file));
           
           for(example e:corpus){
               
               
              out.println(""+e.sentence.get(0)+" "+e.sentence.get(1)+" ||| "+
                      e.sample.get(0)+" "+e.sample.get(1)+" ||| "+e.weight);
               
           }
        }catch(Exception e){
            System.err.println(e);
        }        
    }
    
    public static void print_translation_table(String file){
        Long token1=null;
        Long token0=null;    
        try{
           PrintStream out=new PrintStream(new File(file));
           //System.out.println("counts size: "+counts.size());
           for(Long key:counts.keySet()){
               
               token1=key & lm.mask;
               token0=(key>>20)& lm.mask;               
               if(token0!=0){
                   float channel_prob=(float)counts.get(key)/(float)counts.get(token0);
                   float reverse_channel=(float)counts.get(key)/(float)o_counts.get(token1);
                   //if(channel_prob >= 0.1 || reverse_channel >= 0.1) {
                       out.println(token1+" ||| "+token0+" ||| "+channel_prob+" "+reverse_channel);
                       
                   //}
               }
           }
        }catch(Exception e){
            
            System.out.println(e+" "+counts.get(token0)+" "+o_counts.get(token1));
        }
    }
    
    public static void print_translation_table_final(String file){
        Long token1=null;
        Long token0=null;    
        try{
           PrintStream out=new PrintStream(new File(file));
           //System.out.println("counts size: "+counts.size());
           for(Long key:final_counts.keySet()){
               
               token1=key & lm.mask;
               token0=(key>>20)& lm.mask;               
               if(token0!=0){
                   float channel_prob=(float)final_counts.get(key)/(float)final_counts.get(token0);
                   float reverse_channel=(float)final_counts.get(key)/(float)o_counts.get(token1)/100;
                   if(channel_prob >= 0.1 || reverse_channel >= 0.1) {
                       out.println(token1+" ||| "+token0+" ||| "+channel_prob+" "+reverse_channel);
                   }
               }
           }
        }catch(Exception e){
            
            System.out.println(e+" "+final_counts.get(token0)+" "+o_counts.get(token1));
        }
    }    
    
    
    /**
     * @param args the command line arguments
     */
    
    
    
    public static void main(String[] args) {
        // TODO code application logic here
        lms = new lm[Integer.parseInt(args[6])];
                // seed table
        load_ptable(args[5]);

         
         for(int i=0;i<Integer.parseInt(args[6]);i++) {
           lms[i]=new lm();
           lms[i].load_lm(args[1]+"_"+i);             
           System.out.println("vocab: "+lms[i].hidden_vocab.length);
           lms[i].load_slice_list(args[2]+"_"+i);
           
         }
         

        iter=Integer.parseInt(args[0]/*"4000"*/);
        long start_time=0;
        seed=Integer.parseInt(args[4]/*"100"*/);

        
        int t_count = 1;
        if(args.length>7)
          t_count = Integer.parseInt(args[7]);
        
        counts=new ConcurrentHashMap<Long,Integer>(100000,0.75f,20);
        final_counts=new ConcurrentHashMap<Long,Long>(100000,0.75f,20);

        load_corpus(args[3]);
        //System.exit(0);      
        
        start_time=System.currentTimeMillis();
        shuffle();
        
        Slice_Gibbs t_pool[] = new Slice_Gibbs[t_count];

        start_time=System.currentTimeMillis();
        
        // start threads for sampling
        int block_size = corpus.size()/t_count;
        int start_pos = 0;
        for(int i=0;i<t_pool.length;i++){
          t_pool[i]=new Slice_Gibbs();  
          t_pool[i].start_pos = start_pos;
          if(i!=t_pool.length-1)
            t_pool[i].end_pos = start_pos + block_size;
          else
            t_pool[i].end_pos = corpus.size();  
          start_pos += block_size;
        }
        
        for(int i=0;i<t_pool.length;i++){
          t_pool[i].start();
        }

        
        try{
          for(int i=0;i<t_pool.length;i++){
            t_pool[i].join();
          }
        }catch(Exception e){
          System.out.println("done");  
        }
        /*for(example e:corpus){
            System.out.println(e.sentence+" "+e.sample);
        }*/
        System.out.println("running time: "+(System.currentTimeMillis()-start_time)/1000.0);
        // output translation table
        print_translation_table(System.getenv("TMPDIR")+"/cipher.id.ptable.final");
        //print_samples(System.getenv("TMPDIR")+"/cipher.id.sample.final");
        //
                 
    }
}
