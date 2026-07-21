# CppExecutor移行測定

## 1. 目的

本書は、Cython CBFをC++へ機械的に書き換えるための性能比較ではない。現Cython Executorの
どの境界がend-to-end時間、copy、Python runtime依存を支配しているかを測定し、CppExecutorが
所有すべき範囲を決める。

値、interval、sequence、status、Diagnosticの同値性は性能値から分離し、既存conformance testを
正本とする。性能値は環境依存であり、pytestの合否条件やgolden値にはしない。

## 2. 測定方法

`benchmarks/native_executor_cpp_gate.py`は次を個別に反復測定する。

- `ExecutionPlan.run(executor=CythonExecutor())`全体
- `F64VectorSourceValues.emissions()`によるSource materialization
- 論理時刻の共通整数tick化とnative入力array packing
- Cython RATE/FRAME call
- Cython固定CBF batch call
- native出力から公開Emissionを構築してNoCollectへ渡すcollector境界
- Python heap peak
- 実装から静的に特定できるpayload copy回数、byte量、Python/native遷移数

各区間は別々に測定するため、中央値の和は厳密なinclusive profileではない。component中央値の和と
end-to-end中央値の差は`unattributed_p50_ns`として残し、負値を性能改善として解釈しない。

再実行command:

```bash
uv run python -m benchmarks.native_executor_cpp_gate \
  --sample-count 8192 \
  --block-sizes 64 256 1024 4096 \
  --channels 4 \
  --beams 2 \
  --warmups 5 \
  --repeats 20 \
  --cpu-model "Apple M4 Pro" \
  --output native_executor_cpp_gate.json
```

CLIは完全なlatency分布、環境、compiler flag、copy内訳をschema `0.1`のJSONとして出力する。

## 3. 測定環境

- 測定日時: 2026-07-21 17:05:37 UTC / 2026-07-22 02:05:37 JST
- CPU: Apple M4 Pro、12 logical CPU
- OS: macOS 26.5.2 arm64
- Python: CPython 3.13.14
- compiler: Apple `cc`、`-O3 -arch arm64`
- 入力: 8192 sample、4 channel、1 Hz論理格子
- CBF: 2 beam、非重複FRAME、NoCollect終端
- clock: `mach_absolute_time()`、報告resolution 42 ns

## 4. 結果

時間は各20回の中央値、throughputはend-to-end中央値から算出した。

| block | frame数 | end-to-end p50 ms | p99 ms | ksample/s | Source化 ms | time/pack ms | RATE/FRAME ms | CBF ms | collector復元 ms | native比率 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 128 | 36.972 | 44.603 | 221.6 | 16.074 | 16.499 | 0.313 | 0.019 | 1.398 | 0.900% |
| 256 | 32 | 36.662 | 43.934 | 223.4 | 15.906 | 16.380 | 0.285 | 0.020 | 1.167 | 0.831% |
| 1024 | 8 | 36.413 | 44.233 | 225.0 | 15.730 | 16.446 | 0.314 | 0.017 | 1.165 | 0.909% |
| 4096 | 2 | 36.465 | 45.306 | 224.7 | 15.768 | 16.287 | 0.299 | 0.019 | 1.149 | 0.873% |

component中央値でend-to-endの91.9%から92.8%を説明できた。残りは2.67 msから2.94 msで、
session生成、status集計、RunResult構築等を含む。Python heap peakは5.86 MiBから6.10 MiBだった。

block sizeを64から4096へ変えてもend-to-end throughputは約1.5%の範囲にあり、現構成では
FRAME数やCBF call粒度が支配要因ではない。Source Emission構築と論理時間/native array packingが
合計約32 msで、end-to-endの約87%から89%を占める。

## 5. copyと境界

8192 sample、4 channel、2 beam、非重複FRAMEではblock sizeにかかわらず次のpayload量になる。

| 箇所 | byte | 性質 |
|---|---:|---|
| Python Sourceからf64 array | 262,144 | Python/native入力境界 |
| RATE materialization | 262,144 | Scheduler内部 |
| FRAME materialization | 262,144 | Scheduler内部 |
| FRAME mallocからPython bytes | 262,144 | Scheduler/Kernel ABI境界 |
| CBF出力mallocからPython bytes | 131,072 | Kernel/Python境界 |
| 合計 | 1,179,648 | payload copy 5回 |

現ABIで直接削除可能なScheduler/KernelおよびKernel/Python境界copyは393,216 byte、全copy量の
33.3%である。実行ごとのPython/native遷移は4回、Stage単位のPython dispatchは1回である。
静的byte量はPython object、tuple、Diagnostic参照、allocator metadataを含まず、実測RSSではない。

## 6. 判断

### 6.1 CBF KernelのC++書き換えを先行しない

Cython CBF自体は8192 sample全体で中央値0.017 msから0.020 msだった。同じscalar loopをC++へ
移してもend-to-endへの寄与は1%未満であり、これだけをCppExecutor着手理由にしない。

### 6.2 owned C ABIを先に固定する

Schedulerが生成したFRAME bufferをPython bytesへcopyせず、所有権付き`BufferView`としてKernelへ
渡す。Kernel出力もPython bytesを経由せず次のnative Kernelまたはnative Sinkへ渡す。最初の
CppExecutor比較ではABI境界copyを393,216 byteから0へ、Stage Python dispatchを1から0へ
減らしたことを計数で確認する。

### 6.3 native ingressと整数tick入力をCppExecutor範囲へ含める

Python Sourceを毎run `Emission`へ展開してから有理数を整数tickへ戻す経路が最大要因である。
CppExecutorは、値、start tick、end tick、status、Diagnostic provenanceを既に持つnative ingressを
受理しなければ大きなend-to-end改善にならない。Python Source互換adapterは残すが、性能経路では
一度だけtickへ正規化したrun-localまたはstreaming input batchを受け取る。

### 6.4 公開Emission復元を観測境界まで遅延する

NoCollectでも現Executorは全CBF出力をPython tupleとEmissionへ復元してから破棄する。NoCollectと
native Sinkでは復元しない。Latest、Bounded、Extension、Python callbackなどPython値を要求する
境界だけで公開Emissionへ変換する。

## 7. 最小CppExecutorの実装順序

1. version付きC ABIへsession create/process/flush/destroy、owned input/output buffer、error code、
   Diagnostic provenanceを固定する。
2. prepacked f64 native ingress descriptorを追加し、logical timeをsigned i64 tickで受け取る。
3. `SOURCE → RATE → FRAME → native MAP → native Sink/NoCollect`の線形Executorを実装する。
4. 複数native MAPを同じowned buffer契約で連結し、Cython CBFとC++ Kernelを混在可能にする。
5. Python collector、Extension、callbackを明示Stage境界として追加する。
6. gap reset、INVALID partition、fan-out共有寿命を同じ状態機械へ追加する。

並列Stage、thread pool、SIMD固有最適化はこの線形経路の再測定後に判断する。

## 8. C++比較時の必須報告

- 同じ入力とFlowで値、interval、sequence、status、Diagnosticが一致すること
- end-to-end、Source/tick ingress、Scheduler、Kernel、collector境界のp50/p95/p99
- input sample/sとoutput frame/s
- Python/native遷移数とStage Python dispatch数
- payload copy回数、ABI境界copy byte、peak native buffer、Python heap peak
- Python adapter経路とnative ingress/native Sink経路を混同しないこと

性能改善率だけを報告せず、どの境界を除去した結果かをPlanと測定内訳から説明できることを
CppExecutorの移行gateとする。

## 9. 最小CppExecutor実装後の再測定

### 9.1 実装条件

`0.4.0.dev0`では、`f64_vector_source()`がSource宣言時にEmission列とnative-endian値、source timebase
上のsigned i64 tick、statusを一度だけpackする。CppExecutorはPortablePlanIRのopcode、RATE period、
FRAME size/hop、Kernel ABI、collector descriptorとprocess-local bindingからC++ sessionを生成する。

C++ sessionは入力と固定CBF係数を所有し、`run()`中はGILを解放する。NoCollectではCBFを実行するが
出力値をPythonへcopyしない。Latest/BoundedではC++側で保持対象を選択してから公開Emissionへ戻す。

再測定日時は2026-07-21 23:34:44 UTC / 2026-07-22 08:34:44 JST、machineと入力条件は第3節と
同一である。benchmark JSON schemaは`0.2`とした。

### 9.2 結果

| block | Cython NoCollect p50 ms | C++生成込み p50 ms | C++構築済みsession ms | NoCollect speedup | Cython Bounded ms | C++ Bounded ms | Bounded speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 18.120 | 0.509 | 0.071 | 35.6x | 18.039 | 1.691 | 10.7x |
| 256 | 17.605 | 0.486 | 0.071 | 36.2x | 17.802 | 1.424 | 12.5x |
| 1024 | 17.631 | 0.484 | 0.071 | 36.5x | 17.595 | 1.406 | 12.5x |
| 4096 | 17.600 | 0.504 | 0.071 | 34.9x | 17.535 | 1.374 | 12.8x |

CppExecutorの直近内部計測では、RATE/FRAMEが約0.056 msから0.084 ms、固定CBFが約0.011 msだった。
owned inputは392.1 KiB、NoCollectのoutput boundaryは0 byte、全frameを保持するBoundedでは128 KiB
だった。Python heap peakはNoCollectで約2.9 KiB、Boundedで約964 KiBから1050 KiBだった。

Cython値が初回の約36 msから約18 msへ下がったのは、Source EmissionをSource宣言時に一度だけ
正規化するよう変更したためである。Cython Executorはrunごとの有理時刻検査とnative array packing、
collector復元を引き続き行う。CppExecutorのsession生成込み約0.5 msと構築済み約0.071 msの差は、
主にowned input/Kernel bindingのsession copyとPlan validationである。

### 9.3 判定

最小線形経路について、次を満たした。

- PythonでFlowを記述・compileし、PortablePlanIRからC++ sessionを生成する
- C++がRATE、FRAME、CBF、collector保持選択を一つのstate machineとして運用する
- run中にGILを保持せず、native Stage内Python dispatchを0にする
- NoCollectの値copyを0 byteにする
- Python/Cython/C++で値、interval、sequence、status、Diagnosticを一致させる
- 同じsessionを再実行してもcursor、status、collector状態を持ち越さない

ただし、現runtimeは固定CBF ABIを直接認識する最小実証である。汎用function pointer ABI、複数native
Kernel chain、継続PlanSession、Extension/Python callback境界、gap reset、INVALID partition、
metadata table、fan-out共有寿命は未実装であり、CppExecutor v0.4の残件とする。
