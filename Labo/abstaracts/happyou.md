# Long Contextの課題について <br> ALRとSelf-Routeから期待する研究方法
## ALR
### 論文名
ALR^2 : A RETRIEVE-THEN-REASON FRAMEWORK FOR
LONG-CONTEXT QUESTION ANSWERING
（https://arxiv.org/pdf/2410.03227 ）
### 背景
最近になって長い文章をLLMに渡すことができるようになった。しかしながらreasoningにおいては長い文章につれて精度が下がっている。

### 目的・手法
reasoningにおいて一つのstepでreasoningさせるのではなく、文章をretirievalさせ、それらのreasoningを行う。その際にretrievalとreasoningするLLMのファインチューニングを行い、精度の改善を期待する。

### 評価
long contextでのHotPotQAとSQuADのベンチマークで最低でそれぞれ23.4と12.7EMよくなった。
direct answering promptingでは最低でそれぞれ8.4と7.9EM良くなった。
### 要約
長い文章でのreasoningの精度が低いという課題があり、それに対して二つのプロセスに分割しそれぞれファインチューニングすることでreasoningの精度が良くなった。

## Self-Route
### 論文名
Retrieval Augmented Generation or Long-Context LLMs?
A Comprehensive Study and Hybrid Approach
（ url: https://www.arxiv.org/pdf/2407.16833 ）

### 背景
最近では100万トークンを超えるような文章を直接LLMに渡すことが可能になった。しかしこのLong Context LLM(LC)は精度は高いがコストがかかり、一方でRAGは精度は低いがコストが安い。

### 目的・手法
RAGとLCの比較を公開されたデータセットを用いて行う。
long contextのneedle testにおいて評価を行う。

ベンチマークとしてLongBenchと$\infty$Benchを使用。
LongBenchは21個のデータセットからなる平均7k単語のベンチマーク。
$\infty$Benchは平均100kトークンのデータセットからなるベンチマーク。
RAGにはContrieverとDragonを用いている。1チャンク300単語で分割し、cosine類似度を用いてtop-k(大体5)で選ぶ。

### 評価
LCとRAGは60%のクエリにおいて同じ推論を行っていた。

Self-Routeは二つのステップ：RAG and RouteステップとLCステップからなる。
RAG and RouteステップはLLMにクエリと上位に選ばれたk個のチャンクを渡し、そのチャンクからLLMが回答できると考えるならばそのまま答えを出す。
LCステップは長文を全て渡してLLMに回答させる。

RAGがどのようなクエリを苦手とするか:
1. multi hop
2. 抽象的
3. 長く複雑
4. 分割されている

使ったデータセット:
1. NarrativeQA
2. Qasper
3. MultiFieldQA
4. HotpotQA
5. 2WikiMQA
6. MuSiQue
7. QMSum

各データセットから欠点の種類が見て取れた。この結果からRAGの未来への課題を理解できる。

- RAGの精度への影響
passkeytestではRAGはLCより精度が良いが、クエリに影響したり、とくに二つのチャンクを使うようなものには精度が悪い。

- LLMの内部知識について
LLMでは内部の知識を使うことがあるが、それを使わせないために「パッセージ内のものを使って」と指示をした。精度が悪くなったから、実際に使われていないのでないか？
あとは常識的に解けるような問題を提示させたりすることも必要。実際にそれを抜きでしたらまた精度が下がった。
新しい課題として内部知識の抑制が考えられる。

### 要約
RAGで回答できないものをLCで処理させるelf-routeを用いることでコストを抑えて精度を出すことを可能にした。
また、個人的に興味があるのは二つのモデルRAGとLCによる解答の違いを分析することでroutingまたは再帰的な処理を行い、その両方の利点を合わせたシステムを構成した点だ。このように分析結果、または特徴の違いからroutingすることは様々な課題に応用可能ではないのかという期待が生まれた。

## 今後の展望
ALRのようなプロセスの細分化や、Self-Routeのようなモデルの選択を行うことで既存の課題に対処したいと考えている。その為に既存の課題点のリストアップ（できればlong contextとRAG）を行い、成果が望めそうなものを行いたいと考えている。

１．（先行文献等の）背景
２．（先行文献等の）目的
３．（先行文献等の）提案手法
４．（先行文献等の）評価
５．自分の所感 (今後こうしていきたい等)