This directory contains the release of a sample of English Foursquare restaurant reviews annotated with Aspect Based Sentiment annotations,
together with an evaluation script enabling assessment of performances of Aspect Based Sentiment detection systems.
These Foursquare reviews have been annotated using the SemEval2016 ABSA challenge annotation schema.
Link to  SemEval2016 challenge web page: http://alt.qcri.org/semeval2016/task5/
The annotation guidelines are available here: http://alt.qcri.org/semeval2016/task5/data/uploads/absa2016_annotationguidelines.pdf

The data/ directory contains two sub-directories, corresponding to two different formats of annotations :

- Semeval_format/
  It contains 3 files of annotated text compliant with SemEval2016 XML format and evaluation procedure (see http://alt.qcri.org/semeval2016/task5/ for details)
  Foursquare_testA.xml contains non-annotated data, Foursquare_testB.xml contains the same data annotated with opinion target expressions (OTE) together with their semantic aspects,
  and finally foursquare_gold.xml contains the same data annotated with the triplet OTE, semantic aspects, polarity.

- brat_format/
  It contains the same textual data annotated in the Brat format, as the brat tool (see http://brat.nlplab.org/)  was use to perform the annotations.
  It includes:
     - annotation.conf: the brat configuration used for the annotation (basically a transcription of SemEval2016 guidelines into Brat)
     - foursquare_raw_reviews.txt: the non-annotated Foursquare Reviews; one review per line, each review may contain more than one sentence;
     - foursquare_sentences.txt: the corresponding sentences, one sentence per line, that have been annotated with Brat
     - foursquare_sentences.ann: the Brat file containing the annotations of foursquare_sentences.txt
     - foursquare_sentences.rid: reviews IDs, each line corresponds to a sentence in foursquare_sentences.txt and contains the corresponding review line number (review ID) of foursquare_raw_reviews.txt.
    
EVALUATION SCRIPT

usage: evaluate_full.py [-h] [-prd PREDICTION] [-gld GOLD] [-s1] [-s2] [-s3]
                        [-s13] [-s12] [-s123] [-pc]

Full chain evaluation of semeval 2016 datasets

optional arguments:

 -h, --help            show this help message and exit
  -prd PREDICTION, --prediction PREDICTION
                        Pathname to the file with predictions
  -gld GOLD, --gold GOLD
                        Pathname to the file with gold annotations
  -s1, --slot1          perform evaluation for slot 1 (aspect detection)
  -s2, --slot2          perform evaluation for slot 2 (opinionated term
                        extraction)
  -s3, --slot3          perform evaluation for slot 3 (polarity
                        classification) WARNING : when using this option, make
                        sure you perform polarity predictions on gold
                        term/aspect annotations. The result will be equivalent
                        to s123 evaluation.
  -s13, --slot13        perform evaluation for slot 1,3 (aspect,polarity)
  -s12, --slot12        perform evaluation for slot 1,2 (OTE, aspect)
  -s123, --slot123      perform evaluation for slot 1,2,3
                        (OTE,aspect,polarity)
  -pc, --perclass       output evaluation per class
  -ps, --persentence    output evaluation per sentence

NOTE : when running script with -s3 option for polarity evaluation, the accuracy given by official semeval script would correspond to the recall of this script

If you use this data release for publications or presentations, please cite:

Caroline Brun, Vassilina Nikoulina. "Aspect Based Sentiment Analysis into the Wild", WASSA@EMNLP-2018 workshop: 9th Workshop on Computational Approaches to Subjectivity, Sentiment & Social Media Analysis. 

For any question, please contact : caroline.brun@naverlabs.com and vassilina.nikoulina@naverlabs.com
