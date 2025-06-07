# TAMMICフォーマット
## Title
Lost in the Middle: How Language Models Use Long Contexts
(https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00638/119630/Lost-in-the-Middle-How-Language-Models-Use-Long )
February 23 2024

## Author
Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang

## Motivation
long contextにおいて様々なモデルが512～2048tokenほどのcontext windowを用いることができるようになった。
しかし、長文内の関連情報の位置によってタスクの精度は下がるのではないかという疑問から調べた。

## Method
openとcloseなlanguage modelのcontext windowや関連情報の位置を調整することで調べた。
まずRAGのようなタスクで複数の文章をモデルに与え調べた。
次にkey-value retrieve taskを行い、modelがretrieveする範囲を調べた。
使ったモデルは
open: MPT-30B-Instruct, LongChat-13B (16K)
closed: OpenAI’s GPT-3.5-Turbo,Anthropic’s Claude-1.3

## Insight
文章を与えてない場合では56.1%、関連情報を渡した場合はU字のグラフを描くような精度のグラフのようになった。とくに最初が高く、中間は文章を与えていない場合よりも精度が低かった（Lost in the middle）。
key-value retrieve taskではあるモデルは完璧に、あるモデルはまたU字カーブを描いた。
また、実験の結果はcontext windowの長さがかわるとU字カーブを描くので、accuracyとcontext window lengthはtrade-offの関係であると示していた。

（ココから精読）
従来のモデルとcontext windowを拡張したモデルで同じ長さで比較を行った。
context windowの拡張をしたものとしていないものの精度はあまり変わってなかった。最初と最後が精度がよく、真ん中が精度が悪かった。
つまりもともとのモデルでも"Lost in the middle"になっていた。

## Contribution Summary

# KURRフォーマット
## keyword 
- open end
  限りのない、機械学習の文脈だと無制限の文書（ネット）とか
- query aware contextualization
  queryをkey-valueまたはdocumentの前に置く方法
- closed book setting
  modelの内部知識だけでタスクを解かせる
- oracle setting 
  modelに一つ関連のある文章を与えてタスクを解かせる
## Unknown

## Reflection

## Reference