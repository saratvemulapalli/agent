1. I want to build a semantic search app. I have 5M japanese documentations.
(ask for clarify, directly give dense)

2. want to build a semantic search app. I have 5M english documentations.
- no more ask, directly give dense
- hallucination (83.4 BEIR)


1. single prompt
2. use tool to ingest knowledge
    - for case 2, we still see hallucination (~30-40% cheaper than dense at scale)


I want to build a semantic search app with 10M docs. 
 English. no budget limitation, but best search relevance
 I don't want to buy a GPU endpoint. host the sparse model on local nodes