===========================================================

   Documentation for Financial Phrase Bank v.1.0

===========================================================

Contents:

1. Introduction
2. Data
3. Acknowledgements
4. Contact Information
5. References

-----------------------------------------------------------

1. Introduction

The key arguments for the low utilization of statistical techniques in financial sentiment analysis have been the difficulty of implementation for practical applications and the lack of high quality training data for building such models. Especially in the case of finance and economic texts, annotated collections are a scarce resource and many are reserved for proprietary use only. To resolve the missing training data problem, we present a collection of ∼ 5000 sentences to establish human-annotated standards for benchmarking alternative modeling techniques. 

The objective of the phrase level annotation task was to classify each example sentence into a positive, negative or neutral category by considering only the information explicitly available in the given sentence. Since the study is focused only on financial and economic domains, the annotators were asked to consider the sentences from the view point of an investor only; i.e. whether the news may have positive, negative or neutral influence on the stock price. As a result, sentences which have a sentiment that is not relevant from an economic or financial perspective are considered neutral.

-----------------------------------------------------------

2. Data

This release of the financial phrase bank covers a collection of 4840 sentences. The selected collection of phrases was annotated by 16 people with adequate background knowledge on financial markets. Three of the annotators were researchers and the remaining 13 annotators were master’s students at Aalto University School of Business with majors primarily in finance, accounting, and economics.

Given the large number of overlapping annotations (5 to 8 annotations per sentence), there are several ways to define a majority vote based gold standard. To provide an objective comparison, we have formed 4 alternative reference datasets based on the strength of majority agreement: 

(i) sentences with 100% agreement [file=Sentences_AllAgree.txt]; 
(ii) sentences with more than 75% agreement [file=Sentences_75Agree.txt]; 
(iii) sentences with more than 66% agreement [file=Sentences_66Agree.txt]; and 
(iv) sentences with more than 50% agreement [file=Sentences_50Agree.txt].

All reference datasets are included in the release. The files are in a machine-readable "@"-separated format:

sentence@sentiment

where sentiment is either "positive, neutral or negative".

E.g.,  The operating margin came down to 2.4 % from 5.7 % .@negative


-----------------------------------------------------------

3. Acknowledgements

The development of the Financial Phrase Bank v.1.0 was supported by Emil Aaltonen Foundation and Academy of Finland (grant no: 253583). 

-----------------------------------------------------------

4. Contact Information

In case you have any questions regarding this phrase bank, please contact
Pekka Malo or Ankur Sinha for further information.

Pekka Malo	email: pekka.malo@aalto.fi
Ankur Sinha	email: ankur.sinha@aalto.fi

-----------------------------------------------------------

5. References

If you plan to use the dataset for research or academic purposes, please cite the following publication. For commercial or any other than academic use of the dataset, contact us for an appropriate license.

Malo, P., Sinha, A., Takala, P., Korhonen, P. and Wallenius, J. (2013): “Good debt or bad debt: Detecting semantic orientations in economic texts.” Journal of the American Society for Information Science and Technology. (in Press)
-----------------------------------------------------------

Pekka Malo
Ankur Sinha
Pyry Takala
Pekka Korhonen
Jyrki Wallenius

version 1.0
last modified 07/19/13