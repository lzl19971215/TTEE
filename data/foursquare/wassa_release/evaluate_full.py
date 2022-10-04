#!/usr/bin/env python3
import sys, os
from io import open
import xml.etree.ElementTree as ET
import numpy as np
import pandas
from pandas import DataFrame
liste_gold={}
liste_pred={}
liste_gold_alloccurences={}
liste_pred_alloccurences={}
COLUMNS=["P", "R", "F1", "COMMON", "GOLD", "PREDICTED"]

def getScoresPerUnit(tpPerClass, fpPerClass, fnPerClass):
    allClasses = set(tpPerClass).union(set(fpPerClass)).union(set(fnPerClass))
    scoresPerClass = DataFrame(index=allClasses, columns=COLUMNS)
    for cl in allClasses:
        tp=0
        fp=0
        fn=0
        if cl in tpPerClass:
            tp = tpPerClass[cl]
        if cl in fnPerClass:
            fn = fnPerClass[cl]
        if cl in fpPerClass:
            fp = fpPerClass[cl]
        sc = getScores(tp, fp, fn, 1)
        scoresPerClass.loc[cl]=sc.values
    return scoresPerClass


def printScores(allscores):
    scores=allscores[0]
    scoresPerClass=allscores[1]
    scoresPerSentence = allscores[2]

    if scoresPerSentence:
        print("SCORES PER SENTENCE")
        print(scoresPerSentence.to_csv())
    if scoresPerClass:
        print("SCORES PER CLASS")
        print(scoresPerClass.to_csv())
    print("OVERALL")
    print(scores.to_latex())


def getScores(TP, FP, FN, TOTAL):
    if TP==0 and FP == 0:
        precision = 1    
    else:
        precision = TP / (TP + FP)        
    if TP == 0 and FN == 0:        
        recall = 1        
    else:
        recall =  TP / (TP + FN)
    if precision == 0 and recall == 0 and TP+FP+FN ==0:
        F1 = 1
    elif precision*recall == 0:
        F1 = 0
    else:
        F1 = 2*precision*recall/(precision+recall)

    columns = ["P", "R", "F1", "COMMON", "GOLD", "PREDICTED"]

    df = pandas.DataFrame(columns = columns)
    df.loc[0] = np.array([precision, round(recall, 4), round(F1, 4), TP, TP+FN, TP+FP])
    return df.loc[0]

def getFullAttStr(att):
    return att['category']+";"+att['polarity']+";"+att['target']+";"+att['from']+"-"+att['to']

def getS12AttStr(att):
    return att['category']+";"+getTermAttStr(att)

def getS13AttStr(att):
    return att['category']+";"+att['polarity']

def getTermAttStr(att):
    return att['target']+":"+att['from']+"-"+att['to']

def getTupleAndTripleOccs(f):
    """

    reads dataset in semeval format, and stores different information required for evaluation task:
        - aspects, 
        - aspect/offset pairs
        - aspect/polarity pairs
        - aspect/offset/polarity triples        
    """

    print(f)
    e = ET.parse(f)
    root = e.getroot()
    aspects = {}
    terms = {}
    aspect_terms = {}
    tuples = {}
    triples = {}
    aspect_pol = {}
    for child in root:
        #print(child.tag, child.attrib)
        sentid=""
        s = 0
        for c2 in child:
            for c3 in c2:                         
                if not "OutOfScope" in c3.attrib:
                    sentid=c3.attrib['id']
                    s+=1
                    if not sentid in tuples:
                        aspects[sentid] = set([])
                        tuples[sentid]=set([])
                        triples[sentid]=set([])
                        terms[sentid]=set([])
                        aspect_terms[sentid]=set([])
                        aspect_pol[sentid] = {}
                    for c4 in c3:
                      #  print(c4.tag, c4.attrib)
                        for c5 in c4:
                            #print("Opinion=",  str(c5.attrib))
                            if "category" in c5.attrib:
                                aspects[sentid].add(c5.attrib['category'])
                            #print(sentid, getTermAttStr(c5.attrib))
                            if 'target' in c5.attrib:
                                at = getS12AttStr(c5.attrib)
                                if c5.attrib['target'].strip() not in ["NULL", ""]:                  
                                    terms[sentid].add(getTermAttStr(c5.attrib))
                                    aspect_terms[sentid].add(at)
                                if not at in aspect_pol[sentid]:
                                    aspect_pol[sentid][at]=[]
                            if 'polarity' in c5.attrib:
                                #if len(aspect_pol[sentid][at]) ==0:
                                triples[sentid].add(getFullAttStr(c5.attrib))
                                tuples[sentid].add(getS13AttStr(c5.attrib))
                                aspect_pol[sentid][at].append(c5.attrib["polarity"])
                                    #print(c3.text, sentid, aspect_pol[sentid])
                                #label=(c5.attrib['category'], c5.attrib['polarity'])

    return dict(zip(["s1", "s2", "s13", "s123", "s12", "s3"], [aspects, terms, tuples, triples, aspect_terms, triples]))



def computeScores(liste_pred, liste_gold, perClass, perSentence, task):
    #print(liste_pred)
    FNperclass={}
    TPperclass={}
    FPperclass={}

    FNpersentence = {}
    TPpersentence = {}
    FPpersentence = {}

    TP=0
    FP=0
    FN=0
    TN=0
    TOTAL = 0    
    nbref=0
    macro_p = 0
    macro_r = 0
    acc = 0
    predicted = 0
    def updateScorePerClass(el_Set, toUpdate):
        for el in el_Set:
            if not el in toUpdate:
                toUpdate[el] = 0
            toUpdate[el]+=1
        return toUpdate
    for c in  liste_gold.keys():
        if c in liste_pred:           
            tpl = liste_pred[c].intersection(liste_gold[c])
            fpl = liste_pred[c].difference(liste_gold[c])
            fnl = liste_gold[c].difference(liste_pred[c])
            total=len(liste_pred[c].union(liste_gold[c]))    

        else:
            tpl = set()
            fpl = set()
            fnl = liste_gold[c]
            total =len(liste_gold[c])
        #print("MISSED", c, fnl)
#        GOLD +=len(liste_gold[c])
        TOTAL +=total
        TPpersentence[c] = len(tpl)
        FNpersentence[c] = len(fnl)
        FPpersentence[c] = len(fpl)
        if total>0:
            acc +=len(tpl)/total
        else:
            acc +=1
        if perClass:
            if task == "s13" or task == "s123":
                # store separately performance for each aspect and each polarity class
                TPperclass = updateScorePerClass([x.split(";")[0] for x in tpl], TPperclass)
                FNperclass = updateScorePerClass([x.split(";")[0] for x in fnl], FNperclass)
                FPperclass = updateScorePerClass([x.split(";")[0] for x in fpl], FPperclass)
                TPperclass = updateScorePerClass([x.split(";")[1] for x in tpl], TPperclass)
                FNperclass = updateScorePerClass([x.split(";")[1] for x in fnl], FNperclass)
                FPperclass = updateScorePerClass([x.split(";")[1] for x in fpl], FPperclass)
            elif task == "s12":
                # store only performance per aspect
                TPperclass = updateScorePerClass([x.split(";")[0] for x in tpl], TPperclass)
                FNperclass = updateScorePerClass([x.split(";")[0] for x in fnl], FNperclass)
                FPperclass = updateScorePerClass([x.split(";")[0] for x in fpl], FPperclass)
            elif not task == "s3":
                TPperclass = updateScorePerClass(tpl, TPperclass)
                FNperclass = updateScorePerClass(fnl, FNperclass)
                FPperclass = updateScorePerClass(fpl, FPperclass)
            else:
                #store performance per polarity class
                TPperclass = updateScorePerClass([x.split(";")[1] for x in tpl], TPperclass)
                FNperclass = updateScorePerClass([x.split(";")[1] for x in fnl], FNperclass)
                FPperclass = updateScorePerClass([x.split(";")[1] for x in fpl], FPperclass)

            
        tp=len(tpl)
        fp = len(fpl)
        fn=len(fnl)
        TP += tp
        FP += fp
        FN += fn

    if perClass and perSentence:
        return getScores(TP, FP, FN, TOTAL), getScoresPerUnit(TPperclass, FPperclass, FNperclass), getScoresPerUnit(TPpersentence, FPpersentence, FNpersentence)
    elif perSentence:
        return getScores(TP, FP, FN, TOTAL), None, getScoresPerUnit(TPpersentence, FPpersentence, FNpersentence)
    elif perClass:
        return getScores(TP, FP, FN, TOTAL), getScoresPerUnit(TPperclass, FPperclass, FNperclass), None
    else:
        return getScores(TP, FP, FN, TOTAL), None, None


import argparse
argparser = argparse.ArgumentParser(description='Full chain evaluation of semeval 2016 datasets')
    
argparser.add_argument('-prd', '--prediction', type=str,
                       help='Pathname to the file with predictions')

argparser.add_argument('-gld', '--gold', type=str,
                       help='Pathname to the file with gold annotations')


argparser.add_argument('-s1', '--slot1', action="store_true",
                       help=' perform evaluation for slot 1 (aspect detection)')

argparser.add_argument('-s2', '--slot2', action="store_true",
                       help=' perform evaluation for slot 2 (opinionated term extraction)')

argparser.add_argument('-s3', '--slot3', action="store_true",
                       help=' perform evaluation for slot 3 (polarity classification) WARNING : when using this option, make sure you perform polarity predictions on gold term/aspect annotations. The result will be equivalent to s123 evaluation.')

argparser.add_argument('-s13', '--slot13', action="store_true",
                       help=' perform evaluation for slot 1,3 (aspect,polarity)')
argparser.add_argument('-s12', '--slot12', action="store_true",
                       help=' perform evaluation for slot 1,2 (OTE, aspect)')

argparser.add_argument('-s123', '--slot123', action="store_true",
                       help=' perform evaluation for slot 1,2,3 (OTE,aspect,polarity)')

argparser.add_argument('-pc', '--perclass', action="store_true",
                       help=' output evaluation per class ')

argparser.add_argument('-ps', '--persentence', action="store_true",
                       help=' output evaluation per sentence ')


args = argparser.parse_args()
    
def main():
    index = []
    if args.slot1:
        index.append("s1")
    if args.slot2:
        index.append("s2")
    if args.slot3:
        index.append("s3")
    if args.slot12:
        index.append("s12")
    if args.slot123:
        index.append("s123")
    if args.slot13:
        index.append("s13")
    pred = getTupleAndTripleOccs(args.prediction)
    gold = getTupleAndTripleOccs(args.gold)
    if args.prediction1:
        pred1 = getTupleAndTripleOccs(args.prediction1)
    results = DataFrame(index=index, columns = COLUMNS)
    if args.perclass:
        resultsPerClass = DataFrame(columns=COLUMNS)
    if args.persentence :
        resultsPerSentece = DataFrame(columns=COLUMNS)
    for s in index:
        sres = computeScores(pred[s], gold[s], args.perclass, args.persentence, s)
        #print(sres[0], sres[0].index, sres[0].values)
        results.columns = sres[0].index
        results.loc[s] = sres[0].values
        if args.perclass:
            resultsPerClass = sres[1]
            resultsPerClass.loc["OVERALL"] = results.loc[s]
            print(resultsPerClass)
        if args.persentence:
            resultsPerSentece = sres[2]
            resultsPerSentece.loc["OVERALL"] = results.loc[s]
            print(resultsPerSentece)


        #resultsPerClass.index = index

    if not args.statistical_test:
        print(results.to_string())
if __name__ == '__main__':
    main() 
    
