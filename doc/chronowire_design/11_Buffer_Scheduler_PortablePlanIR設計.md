# Buffer、Scheduler、PortablePlanIR設計

## 1. 目的

本書は、途中Portのbuffer上限、fan-out時の値の寿命、exact mergeのready判定、Sourceの流量制御、およびそれらをExecutor間で共有するPortablePlanIR descriptorを定める。

Chronowireは、制御可能なSourceを無制限に先行実行して滞留をbuffer上限だけで抑え込まない。Schedulerは観測終端から必要な論理時間区間を逆伝播し、その区間を生成するために必要な経路だけを進める。buffer上限は一時的な実行揺らぎ、明示された動的出力、および制御不能な外部入力を安全に扱うための契約であり、誤ったSchedulingを隠すためには使用しない。

## 2. StreamItemの共有と所有権

### 2.1 PortBuffer

fan-outするPortは、consumerごとの値queueではなく、一つの読み取り専用`PortBuffer`へStreamItemを一度だけ格納する。

```text
Producer
   ↓
PortBuffer
   ├── Consumer A cursor
   ├── Consumer B cursor
   └── Consumer C cursor
```

各consumerは独立したcursorを持つ。StreamItemは、そのitemを必要とする全consumerのcursorが通過した時点でreclaimできる。consumer数が0の観測終端はcollectorまたはExtensionへの同期配送完了後にreclaimできる。

Extensionは同期callback中だけStreamItemを参照できる。callback終了後も値を保持するExtensionは、自身の責務でcopyまたは所有権取得を行い、PortBufferのconsumer cursorには含めない。

### 2.2 読み取り専用契約

StreamItemの値、interval、sequence、status、Diagnostic、metadataは論理的にimmutableとする。Python Executorでは契約を正本とし、debug modeでNumPyのwriteable flag、metadataの変更、入力値の変更を検査できるようにする。Native Executorの`BufferView`はread-only flagとownershipを明示する。

in-place実行は、compileが次のすべてを証明できる場合だけ許可する。

- 対象itemの未処理consumerが一つだけ
- collectorまたはExtensionの観測が完了している
- Backend間またはdevice間の共有参照がない
- Kernelがin-place対応を宣言している

### 2.3 EdgeAdapterBuffer

dtype、layout、device、Backend境界で変換が必要な場合は、共有`PortBuffer`とは別に`EdgeAdapterBuffer`を置く。Logical Graphのfan-outと物理変換bufferを混同しない。

## 3. Buffer分類と上限

PortablePlanIRは各runtime bufferを次のいずれかに分類する。

| 分類 | 用途 | capacityの決定方法 |
|---|---|---|
| `PORT_SHARED` | 通常Portとfan-out | compile時の滞留解析または明示`max_items` |
| `EDGE_ADAPTER` | dtype、layout、device変換 | adapterの入出力契約 |
| `FRAME_HISTORY` | `frame(size, hop)`の履歴 | `size`と入力batch上限 |
| `LATEST_STATE` | latest入力 | 確定済み最新値1件と必要なfuture pending |
| `REALTIME_INGRESS` | 制御不能な外部push入力 | Source作成時に必須の`max_items` |

capacityは少なくとも`max_items`を持つ。native memoryの割当と保護には`max_bytes`も使用できるが、item単位の意味論上限をbyte数だけで置き換えない。

通常の計算Portと`EDGE_ADAPTER`の既定overflow policyは`FAIL`とし、暗黙dropを禁止する。`LATEST_STATE`は古い確定値の置換を意味論として許可する。`REALTIME_INGRESS`は明示されたdrop policyを使用する。

Kernelが一回の実行で複数Emissionを生成できる場合、Kernel descriptorに一回あたりの`max_items`を必須とする。実行結果が宣言を超えた場合は契約違反として例外にし、途中までの出力を公開しない。

## 4. Sourceの分類

### 4.1 PULL_CONTROLLED

`scene_renderer`のようにScheduler要求から信号を生成できるSourceは`PULL_CONTROLLED`とする。

- Schedulerは必要な`start_time`と`duration`だけを要求する
- downstreamに未処理需要がない場合は生成しない
- downstream bufferがhigh-watermarkへ達した場合は追加要求しない
- exact mergeの一方が揃っている場合、揃っている経路を先行生成しない
- 通常動作ではdropしない

有限IterableをadapterでSource化する場合も、Executorが`next()`を呼ぶ時点を制御できるため`PULL_CONTROLLED`として扱う。ただし`next()`自体が外部イベントを待つIterableはリアルタイムSourceの代用にしない。

### 4.2 REALTIME_PUSH

オーディオdevice、sensor、network受信など、Schedulerが発生を停止できないSourceは`REALTIME_PUSH`とする。

- Source callbackとExecutorの間に`REALTIME_INGRESS` bufferを置く
- 一時的な処理時間の揺らぎを`max_items`件まで吸収する
- `BLOCK`は禁止し、device callbackをScheduler待ちにしない
- v0.1の既定overflow policyは`DROP_OLDEST`
- 連続性を優先する用途だけ、明示指定で`DROP_NEWEST`を許可する

`max_items`はフレームワーク固定値にせず、Source作成時またはSource binding時に必須指定する。処理が恒常的に入力rateへ追いつかない場合、bufferを増やして遅延を蓄積し続けず、指定policyで棄却して最新の処理可能位置へ回復する。

## 5. 論理時間frontierとdemand-driven Scheduler

### 5.1 demandの逆伝播

Schedulerは観測終端が必要とする論理intervalを起点に、依存経路へ需要を逆伝播する。

```text
観測終端が interval T を要求
        ↓
Nodeが各入力へ必要intervalを導出
        ↓
不足している入力経路だけを進める
        ↓
全入力がreadyならNodeを実行
```

各入力cursorは少なくとも次を追跡する。

- `required_interval`: Nodeが次に必要とするinterval
- `available_interval`: cursor位置にあるitemのinterval
- `producer_frontier`: producerが次に生成可能なinterval
- `eof`: producerがこれ以上生成できないか

Sourceをround-robinで無条件に進める方式は意味論の基準実装にしない。Schedulerはready Nodeを実行し、未充足需要を特定し、その需要を満たすSourceまたは上流Nodeだけを進める。

### 5.2 exact merge

compile時に、各同期入力のtime descriptorからinterval列の一致を検証する。time descriptorは少なくともtimebase、duration、period、phase、時間変換を持つ。

- 完全一致を証明できる場合は通常のexact mergeとする
- 不一致を証明できる場合は`POSSIBLE_INTERVAL_MISMATCH` warningを生成する
- 情報不足で証明できない場合もwarningを生成し、runtime frontierで検証する

warningは従来どおりcompileを停止しない。ただしruntimeは不一致を理由に一方の入力を無制限に先行させない。

mergeがinterval `T`を待っているとき、ある入力の`producer_frontier`または先頭itemが`T`を通過し、過去の`T`を今後生成できないことが確定した場合、そのNodeを`STALLED_EXACT_MERGE`とする。

- 該当Nodeへの需要伝播を停止する
- Node、全入力Port、必要interval、先行intervalをDiagnosticへ記録する
- そのNodeに依存しない観測終端は継続する
- 他に実行可能な需要がなければrunを終了する
- EOFまで無意味なitemを蓄積しない

`maximum_skew`はv0.1の完全一致mergeには導入しない。将来のtolerance付き同期または非同期時刻合わせの契約として検討する。

### 5.3 latest入力

latest入力は基準入力の`interval.start`以前に確定した最新値を使用する。基準時刻より古い値は最新値一件を残してreclaimし、基準時刻より未来の値はproducerのburst上限またはbufferの`max_items`を超えて先行取得しない。

## 6. realtime overflowとgap伝播

`REALTIME_INGRESS`で棄却が発生した場合、Executorはそれを単なるqueue操作ではなく論理時間の欠落として扱う。棄却位置には内部`GapMarker`を順序付きで記録する。

`GapMarker`は少なくとも次を持つ。

- Source IDとPort ID
- 欠落intervalの開始と終了
- 今回および累積のdrop件数
- buffer capacity
- overflow policy
- `INPUT_OVERRUN` Diagnostic

`GapMarker`自体は利用者値の`Emission`ではない。Scheduler、FRAME、RATE、merge、stateful Kernel sessionが欠落境界を認識するためのruntime control recordであり、PortablePlanIRにはgap処理policyを記録する。

`GapMarker`は`max_items`へ数える通常itemとしてingress ringへ格納しない。ingressは非dropのpending gap summaryを別に持ち、連続したdropを一つの欠落intervalと累積件数へcoalesceする。Executorは次の受理itemより前にそのsummaryを順序付きで取り出す。これにより、buffer満杯を理由に欠落情報自体が失われることを防ぐ。

v0.1の既定処理は次のとおりとする。

- 次に配送されるEmissionを`DEGRADED`とし、`INPUT_OVERRUN` Diagnosticを付加する
- Extensionと`RunResult`へdrop件数と欠落intervalを通知する
- FRAMEは欠落前の未完成履歴を暗黙に欠落後へ接続しない
- RATEは欠落後の入力intervalから発火境界を再確立する
- exact mergeは欠落intervalが生成不能であることを認識し、対応する待機itemを理由付きで解放して共通frontierへ進む
- stateful Kernelはgap policyに従ってsession stateをresetするか、gapを受理する

stateful Kernelのv0.1既定gap policyは`RESET`とする。`ACCEPT`はKernelが明示的に宣言し、欠落後の最初の出力へDiagnosticを伝播できる場合だけ許可する。欠落をまたいだ継続が安全でない場合は`INVALID`を生成するか例外で停止する。

## 7. PortablePlanIR descriptor

PortablePlanIRはPython objectを含まず、ID参照とserialization可能な固定descriptorだけで構成する。

### 7.1 NodeDescriptor

- `node_id`、`opcode`
- input Port ID列、output Port ID列
- config scope IDまたは安定digest
- execution domainとbinding slot
- `accepts_invalid`
- time transform ID
- 一回の実行あたりのEmission `max_items`
- state、workspace、gap policy

### 7.2 PortDescriptor

- `port_id`、producer Node ID、output index
- value schema ID
- dtype、shape、layout、device、opaque Python値の区別
- time descriptor ID
- sequence domain
- status、Diagnostic、metadataの搬送契約
- 共有Buffer ID

### 7.3 EdgeDescriptor

- `edge_id`
- source Port ID、target Node ID、target input index
- `SYNCHRONOUS`または`LATEST`
- requiredまたはoptional
- source Buffer ID
- EdgeAdapter IDまたはnull
- consumer cursor ID

### 7.4 BufferDescriptor

- `buffer_id`とbuffer分類
- producer IDとconsumer cursor ID列
- `max_items`、任意の`max_bytes`
- device、alignment
- overflow policy
- reclaim policy
- read-only、ownership、copy条件
- high-watermarkとlow-watermark

### 7.5 TimeDescriptor

- timebaseの整数分子・分母
- interval duration、period、phase
- identity、rate、frame、Kernel宣言のtime transform
- exactnessと有限・無限の生成範囲

有理数を実装言語固有の文字列表現だけで保存しない。分子・分母を正規化した整数として保存し、schemaで整数幅とoverflow条件を定める。

### 7.6 SourceDescriptorとBindingDescriptor

`SourceDescriptor`は`PULL_CONTROLLED`または`REALTIME_PUSH`、request単位、有限性、burst上限、ingress Buffer IDを持つ。`BindingDescriptor`はSource、CompiledKernel、Python callback、collector、Extensionのprocess-local実体へ結び付ける安定slot IDとABI/schema versionを持つ。

pointer、allocator instance、Python callable、Python class名はPortablePlanIRへ保存しない。

### 7.7 ExtensionDescriptor

Extension観測はcompile時に固定し、次を`ExtensionDescriptor`へ保存する。

- 利用者が明示した一意な`extension_id`
- observed Port ID
- `Always`、`Every`、`EveryLogicalTime`のtrigger descriptor
- priority
- failure policyとoverflow policy
- process-local実体を要求するbinding slot
- Extension ABI version

`extension_id`は観測契約の安定IDであり、binding slotとは同一概念ではない。Compilerは`extension_id="spectrum_snapshot"`から`binding_slot="extension:spectrum_snapshot"`を決定的に生成できる。対応する`BindingDescriptor(kind="extension")`はslotとABIだけを参照し、callback、path、Python objectを保持しない。

観測Portはrequired rootおよびFusion境界である。`PortBuffer.max_items`はScheduler内部の計算capacityであり、Extensionの履歴保持件数やCollector capacityへ流用しない。

## 8. compile時のbuffer planning

compileは次の順序でbufferを計画する。

1. Port、Edge、consumer cursorへ安定IDを割り当てる
2. time descriptorとSource modeを確定する
3. exact mergeのrate、phase、duration、time transformを検証する
4. frame履歴、latest保持、Kernel出力`max_items`を解析する
5. fan-out consumerごとの最大cursor遅延を解析する
6. `PORT_SHARED`と必要な`EDGE_ADAPTER`を割り当てる
7. 静的上限、明示`max_items`、backpressureのいずれでboundedになるかを記録する
8. 制御可能な経路が無制限先行しないことをScheduler契約として検証する
9. PortablePlanIRとcompile Diagnosticを生成する

静的上限を証明できず、`max_items`もbackpressureもない動的bufferはcompile errorとする。compile warningとruntime overflowを混同しない。

## 9. Executor間の同値性

Python、Cython、C++ Executorは、最適化方法や物理buffer実装が異なっても次を一致させる。

- Source requestの論理interval
- Node ready順と決定的な同順位規則
- Portごとの値、interval、sequence、status、Diagnostic、metadata
- consumer cursorが観測するitem列
- reclaim可能になる論理条件
- realtime drop policy、drop件数、欠落interval
- gap後のFRAME、RATE、merge、stateful Kernelの挙動
- `STALLED_EXACT_MERGE`とbuffer overflowの判定

参照カウント、ring buffer、arenaなどの物理方式はExecutor固有でよい。PortablePlanIRは物理pointerではなく、上記意味論を再現するためのdescriptorを保持する。
