記録をしてみようと思ったきっかけ（ https://iis-lab.org/misc/paperreading ）
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
GPT-4で100個の事実を生成し、PaulGrahhamEssayからcandidate documentを100個用意。各queryに対して100個のcandidate document, それを50セット作り、各{0.25, 0.5, 1,2,4,8,16,32}*1024tokenで行った。
ほかに現実のタスクに対応するためにlong-termQAとsummarization taskを作成した。


## Insight
plug-and-playでのcontext windowの拡張ではRoPEが良く、さらにfine tuningでさらに制度がよくなるような可能性を秘めている。

## Contribution Summary
Dawei Zhuはplug-and-playでcontext windowの拡張を動機とし、pallarel context, reorganize position ids, position interpolationの手法で行ったすべてのモデルでのより長いcontext windowでの正確さの向上を得られた。
  
# KURRフォーマット
## keyword 
#### context window  
一度にembeddingできるトークンの限界数

#### contrasive loss
　contrasive learningというラベルなし学習において各パラメータの近さを測るもの。おそらくそれに用いられる目的関数  
https://ai-scholar.tech/articles/contrastive-learning/UBCL

#### APE(absolute position encoding)  
transformerで用いられる位置埋め込み(position embedding)の一種。
位置埋め込みとは単語の語順をtransformerは処理しないため、position embeddingとして与える必要がある。最もよく見られる。BERTで使われている。<br>
absolute word positionをposition vectorに変え、transfomerに与える。
https://paperswithcode.com/method/absolute-position-encodings
  
#### RoPE(routary position embedding)  
position embeddingの一つ。最近広まっている。LLaMA,QWenで使われている。APEでは長文に弱く、スケーリングや一般化が難しい。クエリとキーのベクトルに回転をかける手法（要調査）。position informationに回転行列をかける。現状では長文で良い精度を出している。<br>
RoPEが行う操作は以下の通り。
$$ f(h,m) = [(h_0+ih_1)e^{im\theta_0}, (h_2+ih3)e^{im\theta_1}, \dots, (h_{d-2}+ih_{d-1})e^{im\theta_{d/2-1}}]   $$
ただし、$\theta_j=10000^{-2j/d}, j\in\{0,1,\dots,d/2-1\}, i=\sqrt{-1}$
xを直接ベクトルにするAPEと違い、RoPEでは各層でkey vectorとqueryが用いられている。位置mでのクエリ$q$と位置nでのkey $k$では、Attention Score $a(q,k)$は次のように定義される。
$$ a(\bm{q},\bm{k}) = Re\langle f\langle \bm{q},m\rangle , f\langle \bm{k}, n \rangle \rangle \\
= Re[\sum^{d/2-1}_{j=0}(q_{2j}+iq_{2j+1})(k_{2j}-ik_{2j+1})e^{i(m-n)\theta_j}] \\
:=g(\bm{q},\bm{k}, (m-n)\bm{\theta})$$
このときg()はabstract mapping functionであり、q,k,(m-n)$\theta$だけに依存する。


#### passkey retrieval test  
長いコンテキストをモデルが記憶かつ取り出せるかのテスト。冗長な文にパスキーが入っており、それを抽出できるかテストするという手法。

#### needle retrieval test  
needle(大事な情報)を長文中から持ってくるテスト。

#### divide and conquer<br>
  長いセンテンスを小さなチャンクに分け、それぞれ各チャンクを処理し、結果を集めるという手法。PCWという手法がある。<br>
  PCWの説明は以下<br>
  $L_0$をもともとのcontext lengthとし、target context lengthが$L_t$の長いドキュメント$\mathcal{D}$を
  $$ \mathcal{D}=\{x_1,x_2, \dots, x_{L_t}\} $$
  あとは$s=[L_t/L_0]$をcontext scaling factorとする。


#### position reorganization<br>
  長いinputの容量をうまく収めるためにposition idの位置を変える手法。SelfExtend、DA、GP、RPという手法がある。
  - GP
  - RP

#### position interpolation<br>
  position idの位置の補完。PIやNTK,reasonance RoPEの手法がある。

#### token compression　<br>
  双方向attentionには使いづらい

#### memory based transfomer
  処理が複雑

#### Benchmark
  - BEIR benchmark
  - LoCo benchmark
#### QA
  - Narrative QA<br>
  物語の集合でできてて、キャラとイベントについて書いてある。また、要点は各所にちりばめられている。
  - 2WikiMultihopQA<br>
  5個までの質問で構成されている（モデルがショートカットして答えを出さないように）。
  - QMSum<br>
  会議の議事録からなる選択と要約を行うためのデータセット
  - SummScreenFD
  TVの台本と人間の書いた要約からなるデータセット。QMSumよりちりばめられてて、簡潔な説明が多い。


   


## Unknown

## Reflection
最近のembedding modelはquery-documentのラベル付けがされたデータを活用して作られているが、それの実装を確認したい。(contrasive lossが使われているらしい)

## Reference
RoFormer: Enhanced Transformer with Rotary Position Embedding(RoPEについて)
あとは先行研究の確認
