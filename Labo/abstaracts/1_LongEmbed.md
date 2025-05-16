記録をしてみようと思ったきっかけ（https://iis-lab.org/misc/paperreading）
# TAMMICフォーマット


## Title
LongEmbed: Extending Embedding Models for Long Context Retrieval  

[Submitted on 18 Apr 2024 (v1), last revised 7 Nov 2024 (this version, v3)]

## Author
Dawei Zhu, Liang Wang

## Motivation
LLMに入れられる限界トークン数は1milionを超えるが、embeddingモデルはcontext windowが8kを超えることがない.<br>
先行研究では既存のLLMにplug-and-playの方法またはファインチューニングで4kから128kに拡張し、2milionにさえも可能だった。
このことから先行研究とは違い、ゼロからトレーニングせずに既存のモデルのcontext windowを拡張することを目的とする。

## Method
LONGEMBEDというベンチマークを導入。理由はdocumentの長さが不足していた点とターゲットとなる情報の分布にバイアスがかかっているため。
手法としてはplug-and-playで並列context window, position idsの再編成、position interpolation(おそらく位置の補完)



## Insight

## Contribution Summary
  
# KURRフォーマット
## keyword 
- context window  
一度にembeddingできるトークンの限界数

- contrasive loss
　contrasive learningというラベルなし学習において各パラメータの近さを測るもの。おそらくそれに用いられる目的関数  
　https://ai-scholar.tech/articles/contrastive-learning/UBCL

- APE(absolute position encoding)  
transformerで用いられる位置埋め込み(position embedding)の一種。
位置埋め込みとは単語の語順をtransformerは処理しないため、position embeddingとして与える必要がある。
https://paperswithcode.com/method/absolute-position-encodings
  
- RoPE(routary position embedding)  
position embeddingの一つ。APEでは長文に弱く、スケーリングや一般化が難しい。クエリとキーのベクトルに回転をかける手法（要調査）。現状では長文で良い精度を出している。

- passkey retrieval test  
長いコンテキストをモデルが記憶かつ取り出せるかのテスト。冗長な文にパスキーが入っており、それを抽出できるかテストするという手法。

## Unknown

## Reflection
最近のembedding modelはquery-documentのラベル付けがされたデータを活用して作られているが、それの実装を確認したい。(contrasive lossが使われているらしい)

## Reference
RoFormer: Enhanced Transformer with Rotary Position Embedding(RoPEについて)
あとは先行研究の確認



$
y = sin(x)
$