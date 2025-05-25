# TAMMICフォーマット
## Title
ROFORMER: ENHANCED TRANSFORMER WITH ROTARY POSITION EMBEDDING

## Author
Jianlin Su, Yu Lu

## Motivation
transformerは単語の位置情報を見ないため、意図的に位置情報を付加する必要がある。そのため、absolute position encodingや、relative position encoding、さらには複素解析空間を用いた手法まで提案されている。それにもかかわらず、一般にはposition informationをcontext representationに追加し、適していない線形なattention機構に使われている。

ここでRoPEというself-attentionにおいてトークンの相対的な位置を回転行列を用いて保存するrelative position encodingの新しい手法を提案する。またそれを用いたrotaty position embeddingを用いたtransformer, ReFormerの性能を比較する。

## Method

## Insight
RePEは現行のposition embeddingに対してcontext sequenceに柔軟であるという性質を持っている。ReFormerは長いcontext taskに対して経験的に柔軟であると考えられる。

## Contribution Summary

# KURRフォーマット
## keyword 

## Unknown

## Reflection

## Reference