
# coding: utf-8

#### HMM Based Sentiment Tagger

# In[47]:

#Code to extract training data from file
import nltk
from collections import defaultdict
from itertools import repeat
tokenizer = nltk.TreebankWordTokenizer()    

alfa=0.1
zen=3

def prob(w,t,es,fs):
    if w in es[t]:
        return(float(es[t][w]+alfa)/(fs[w]+alfa*zen))
    else:
        return(float(alfa)/(fs[w]+alfa*zen))
    
def emisn(w,t,es,fs):
    if w in es[t]:
        return(float(es[t][w]+alfa)/(fs[t]+alfa*zen))
    else:
        return(float(alfa)/(fs[t]+alfa*zen))

def transn(t1,t2,ts,fs):
    if t2 in ts[t1]:
        return(float(ts[t1][t2]+alfa)/(fs[t1]+alfa*zen))
    else:
        return (float(alfa)/(fs[t1]+alfa*zen))
    
freqs = {}
tagfreqs={}
tags = defaultdict(dict)
trans = defaultdict(dict)

f=open('training_data.txt','rb')
text=f.read()
f.close()
lines = text.splitlines()
for line in lines:
    t=0
    if len(line.split(" "))<=1:
        continue
    tokens=line.split("\t")
    sen=tokens[0]
    tokens=tokens[1].split(" ")
    tokens.insert(0,"START")
    tokens.append("END")
    while t<len(tokens):
        try:
            freqs[tokens[t]]+=1
        except:
            freqs[tokens[t]]=1
        try:
            tagfreqs[sen]+=1
        except:
            tagfreqs[sen]=1
        try:
            tagfreqs["START"]+=1
        except:
            tagfreqs["START"]=1
        try:
            tags[sen][tokens[t]]+=1
        except:
            tags[sen][tokens[t]]=1
        t+=1
   
for line in lines:
    t=0
    if len(line.split(" "))<=1:
        continue
    tokens=line.split(" ")
    sen=tokens[0]
    st=["START"]
    tokens=tokens[1:]
    tokens.insert(0,"START")
    tokens.append("END")
    while t<len(tokens)-1:
        if tokens[t]=="START":
            z=max(tags.keys(), key=lambda tag:prob(tokens[t+1],tag,tags,freqs))
            try:
                trans["START"][z]+=1
            except:
                trans["START"][z]=1
        else:
            z=max(tags.keys(), key=lambda tag:prob(tokens[t+1],tag,tags,freqs))
            y=max(tags.keys(), key=lambda tag:prob(tokens[t],tag,tags,freqs))
            try:
                trans[y][z]+=1
            except:
                trans[y][z]=1            
        t+=1
    
print "Training Completed"


# In[48]:

#Perform sentiment tagging based on trained HMM Model using Viterbi Algorithm
from collections import Counter

def viterbi(fs,es,ts,ft):
    total=0
    cfm=defaultdict(dict)
    f=open('test_data_no_true_labels.txt','rb')
    fw=open('submit.csv','wb')
    fw.write("id,answer\n")
    text=f.read()
    f.close()
    lines = text.splitlines()
    for line in lines:
        t=0
        if len(line.split(" "))<=1:
            continue
        tokens=line.split("\t")
        tokens=tokens[1].split(" ")
        tokens.insert(0,"START")
        tokens.append("END")
                
        viterbi = [ ]
        backpointer = [ ]
        first_viterbi = { }
        first_backpointer = { }
        for tag in es:
            if tag == "START":
                continue
            first_viterbi[tag] = transn("START",tag,ts,fs)*emisn(tokens[0],tag,es,fs)
            first_backpointer[tag] = "START"
        viterbi.append(first_viterbi)
        #print "check1"
        backpointer.append(first_backpointer)
        for i in range(1, len(tokens)-1):
            this_viterbi = { }
            this_backpointer = { }
            prev_viterbi = viterbi[-1]
            for tag in es:
                if tag == "START":
                    continue
                best_previous = max(prev_viterbi.keys(), key = lambda prevtag:prev_viterbi[prevtag]*transn(prevtag,tag,es,fs)*emisn(tokens[i],tag,es,fs))
                this_viterbi[tag] = prev_viterbi[best_previous]*transn(best_previous,tag,ts,fs)*emisn(tokens[i],tag,es,fs)
                this_backpointer[ tag ] = best_previous
            viterbi.append(this_viterbi)
            backpointer.append(this_backpointer)
        
        prev_viterbi = viterbi[-1]
        best_previous = max(prev_viterbi.keys(),key = lambda prevtag: prev_viterbi[prevtag]*transn(prevtag,"END",ts,fs))
        prob_tagsequence = prev_viterbi[ best_previous ]*transn(best_previous,"END",ts,fs)
        best_tagsequence = [ "END", best_previous ]
        
        backpointer.reverse()
        current_best_tag = best_previous
        for bp in backpointer:
            best_tagsequence.append(bp[current_best_tag])
            current_best_tag = bp[current_best_tag]
        best_tagsequence.reverse()
    
        if best_tagsequence[-2]=="neg":
            fw.write(str(total)+",-1"+"\n")
        elif best_tagsequence[-2]=="pos":
            fw.write(str(total)+",1"+"\n")
        else:
            fw.write(str(total)+",0"+"\n")
        total+=1
    fw.close()


# In[49]:

viterbi(tagfreqs,tags,trans,freqs)




