# llama.cpp multi-backend build for gfx906 + Blackwell

CUDA (sm_120) / HIP (gfx906) / Vulkan の3バックエンドを **1つのバイナリ**に同居させた
llama.cpp + [llama-swap](https://github.com/mostlygeek/llama-swap) の Dockerfile。

想定ハード: RTX 5060 Ti x2 + Radeon Pro VII x2 + Radeon Instinct MI50 32GB x1。

## なぜ必要か

gfx906 (Vega20 / MI50 / Radeon Pro VII) は ROCm 6.4 以降 rocBLAS の Tensile から
カーネルが削除され、公式 ROCm では実質使えない。
[mixa3607/ML-gfx906](https://github.com/mixa3607/ML-gfx906) が TheRock ベースで
gfx906 対応 ROCm を再ビルドして配布しているが、配布イメージ
`mixa3607/llama.cpp-gfx906` は **HIP + RPC バックエンドのみ**で、NVIDIA GPU を扱えない。

ここでは同じ ROCm ベースに CUDA と Vulkan を足して、AMD と NVIDIA を
同一プロセスから同時に使えるようにしている。
後述の計測のとおり、**CUDA と Vulkan を足しても gfx906 の性能は落ちない。**

## 3種類の Dockerfile

| | ソース | バックエンド | KV cache 量子化 | ベース |
|---|---|---|---|---|
| `Dockerfile.upstream` | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) `b10470` | CUDA + HIP + Vulkan | 標準のみ | ROCm 7.14 (TheRock) |
| `Dockerfile.turboquant` | [AtomicBot-ai/atomic-llama-cpp-turboquant](https://github.com/AtomicBot-ai/atomic-llama-cpp-turboquant) `b10269-1.5.1` | CUDA + HIP + Vulkan | **turbo2/3/4** | ROCm 7.2.4 |
| `Dockerfile.beellama` | [Anbeeld/beellama.cpp](https://github.com/Anbeeld/beellama.cpp) `preview-v0.4.5` | CUDA + Vulkan | **KVarN + precision tail** | Ubuntu 24.04 + CUDA 13.3 |

`--spec-type` / `draft-mtp` / `draft-eagle3` / `ngram-*` はどちらでも使える
(上流に取り込み済み)。フォーク側だけの利点は TurboQuant の KV cache 量子化型で、
`turbo3` は約 3.5bit/要素。`q8_0` 比でおよそ 1/2.4、`f16` 比でおよそ 1/4.5 の KV サイズになる。
長コンテキストを VRAM に収めたい場合はフォーク版を選ぶ。
そうでなければ上流版のほうが速く、イメージも半分で済む。

## ビルド

```sh
docker build -t llama-swap-upstream:b10470   -f Dockerfile.upstream   .
docker build -t llama-swap-turboquant:b10269 -f Dockerfile.turboquant .
docker build -t llama-swap-beellama:v0.4.5   -f Dockerfile.beellama   .
```

ビルドコンテキストは使わないので `.` は何でもよい。主な `--build-arg`:

| ARG | 既定 | 用途 |
|---|---|---|
| `CUDA_ARCH` | `120` / `80;120` | NVIDIA の compute capability。RTX 50 系は 120、GA100 (CMP 170HX / A100) は 80。複数指定するときは `--build-arg CUDA_ARCH="80;120"` と引用符で囲む (囲まないと `docker` を叩くシェル側が `;` でコマンドを切る) |
| `AMDGPU_TARGET` | `gfx906` | AMD のターゲット |
| `LLAMACPP_TAG` / `TURBOQUANT_TAG` / `BEELLAMA_TAG` | 上表 | ソースのタグ |
| `LLAMA_SWAP_VERSION` | `199` | llama-swap のリリース版 |

成果物は `Dockerfile.upstream` で約 15GB、`Dockerfile.turboquant` で約 31GB
(CUDA フルツールキット + ROCm + ビルドツリーを1ステージに持つため)。
配布したい場合はランタイム用にマルチステージ化して `cuda-runtime` に落とすこと。

## 実行

```sh
docker run --rm \
  --device /dev/kfd --device /dev/dri --group-add video \
  --gpus all --security-opt seccomp=unconfined \
  -v /path/to/models:/models:ro -v /path/to/config.yaml:/app/config.yaml \
  -p 8080:8080 llama-swap-upstream:b10470
```

## 計測

Radeon Pro VII 1枚 / Qwen2.5-VL-7B-Instruct Q4_K_M /
`llama-bench --device ROCm0 -p 512 -n 128 -r 3`。値は t/s。

| ビルド | バックエンド | pp512 | tg128 |
|---|---|---|---|
| TurboQuant `b10018-1.1.2` | HIP+CUDA+Vulkan | 890.5 ± 20.6 | 79.0 ± 0.4 |
| TurboQuant `b10269-1.5.1` | HIP+CUDA+Vulkan | 986.8 ± 23.9 | 76.7 ± 0.4 |
| 上流 `b10470` (本リポジトリ) | HIP+CUDA+Vulkan | **1019.4 ± 53.8** | **79.3 ± 0.7** |
| `mixa3607/llama.cpp-gfx906:b10470-rocm-7.14` | HIP のみ | 969.5 ± 118.0 | 79.1 ± 0.8 |

pp512 は分散が大きいので、b10470 の2ビルドだけ `-r 8` で取り直した:

| ビルド | pp512 (`-r 8`) |
|---|---|
| 上流 `b10470` (HIP+CUDA+Vulkan) | 1029.8 ± 34.1 |
| `mixa3607` `b10470` (HIP のみ) | 1026.4 ± 36.7 |

差は 0.3% で標準偏差に埋もれる。両者は同じ上流タグ (`build: 34af94c`) なので、
**CUDA と Vulkan を追加してもバイナリサイズが増えるだけで gfx906 の速度は変わらない**、
というのがこの表の要点。

TurboQuant フォークは `b10018` → `b10269` で prefill が +10.8% 改善する一方、
tg は -2.8% 落ちる。上流 `b10470` は prefill が最も速く、tg も落ちない。

### 計測時の落とし穴

どちらも実際に踏んで数字を誤読したので記録しておく。

1. **`HIP_VISIBLE_DEVICES` は Vulkan を絞れない。**
   Vulkan バックエンドを同梱したビルドでは、`HIP_VISIBLE_DEVICES=0` で ROCm を1枚に
   絞っても Vulkan からは全 GPU が見えたままで、レイヤーが ROCm と Vulkan に分散して
   バックエンド跨ぎ転送が発生する。その状態だと pp512 が 975 → 506 まで落ちた。
   Vulkan を積んでいない `mixa3607` 版では起きないので、
   **自分のビルドだけが一方的に遅く見える**という質の悪い誤差になる。
   `llama-bench --device ROCm0` のようにデバイスを明示すること。
2. **他プロセスの VRAM 占有を確認する。**
   ComfyUI が同じ GPU に常駐していた状態では tg128 が 36.2 ± 23.0 のように暴れた。
   `rocm-smi --showpids` で確認してから測る。tg128 の標準偏差は正常なら 1% 未満なので、
   そこが膨らんでいたら測り直す。

### ROCm のバージョンについて

ROCm 7.14 のランタイムだけを 7.2.4 ビルドのバイナリに `LD_LIBRARY_PATH` で
差し替えて試した (soname は `librocblas.so.5` / `libhipblas.so.3` /
`libamdhip64.so.7` で一致するので差し替え自体は通る)。
結果は prefill が変わらず tg はむしろ不安定化した。
ROCm のバージョン自体は gfx906 の llama.cpp 性能にほとんど寄与しておらず、
上表の差はほぼ llama.cpp 側の変更によるもの。

なお HIP のコンパイルフラグには mixa3607 の preset と同じ
`-mllvm -amdgpu-sched-strategy=max-ilp` を入れている (gfx906 の prefill 対策)。

## AMD なし構成 (`Dockerfile.beellama`)

AMD を挿していない構成では、ROCm ベースイメージが丸ごと無駄になる。
`Dockerfile.beellama` は素の Ubuntu 24.04 + CUDA 13.3 から積み、
**CUDA + Vulkan の2バックエンド**だけをビルドする。

beellama.cpp を選ぶ理由は2つ。

1. **KVarN。** variance-normalized な KV cache 量子化 (`kvarn2`〜`kvarn8`) に加えて、
   直近 N トークンだけ F16/BF16 で残す precision tail を持つ。
   `-ctk` / `-ctv` を独立に指定できるので、V だけ低ビットに倒せる。
   TurboQuant の `turbo2/3/4` に対する代替で、公称値では
   KVarN3/KVarN2 + 1024トークン tail が BF16 比 20.7% (median KLD 0.002424)。
   `turbo3` の f16 比およそ 22% とほぼ同サイズだが、こちらは KLD が公開されている。
2. **Qwen3.8-Flash-Next (`qwen4exp`) を取り込み済み。**
   上流の [PR #27742](https://github.com/ggml-org/llama.cpp/pull/27742) は未マージなので、
   `Dockerfile.upstream` のタグ指定では同モデルを動かせない。

### 依存パッケージが既存2つと違う

上流は HTTP クライアントを curl から cpp-httplib + OpenSSL に移しており、
`preview-v0.4.5` のツリーには `find_package(CURL)` も `curl/curl.h` も残っていない
(`LLAMA_CURL` は `llama_option_depr` 行きで、渡しても無視される)。
そのため `libcurl4-openssl-dev` ではなく **`libssl-dev`** を入れている。
これが無いと configure が `OpenSSL not found, HTTPS support disabled` と警告し、
ビルドは通るが `-hf` / `--model-url` での HTTPS 取得が無効になる。

`GGML_CUDA_NO_MXFP4` もこのツリーには存在しないので指定していない。

### Vulkan を動かすのに `graphics` capability が要る

NVIDIA の Vulkan ICD は nvidia-container-toolkit がホストから注入するが、
既定の `compute,utility` では入らず、Vulkan バックエンドがデバイスを1枚も
見つけられない。Dockerfile 側で

```
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
```

を指定してある。コンテナ内で `vulkaninfo --summary` が GPU を列挙するか確認できる。

### 実行

```sh
docker run --rm --gpus all \
  -v /path/to/models:/models:ro -v /path/to/config.yaml:/app/config.yaml \
  -p 8080:8080 llama-swap-beellama:v0.4.5
```

`--device /dev/kfd` 等の AMD 向けオプションは不要。

### Qwen3.8-Flash-Next の 51B テーブルの置き場所

以下は `preview-v0.4.5` (`476aa6c`) のソースを読んで確認した内容。

51B の N-gram テーブルのテンソル名は **`per_layer_token_embd`** (`src/llama-arch.cpp:548`)。
`ple_key` / `ple_value` 等の `blk.N.ple_*` はどれも小さい射影行列で、別物。

そして通常は手で配置指定する必要がない。beellama はこのテンソルを
`TENSOR_READ_LAZY` 付きで確保し (`src/models/qwen4exp.cpp`)、mmap 経由で
必要な行だけをディスクから読む。制御は `--tensor-read-lazy`:

| 値 | 挙動 |
|---|---|
| `auto` (既定) | 4 GiB を超えるテンソルにだけ on |
| `on` | 常に on (mmap 必須) |
| `off` | 常駐させる |

51B テーブルは当然 4 GiB を超えるので、**既定のまま on になる**。
RAM は常駐先ではなくページキャッシュとして効く。

明示的に RAM 常駐にしたい場合だけ、

```
--tensor-read-lazy off -ot "per_layer_token_embd=CPU"
```

`-ot` の正規表現がどのテンソルにもマッチしなかった場合は黙って無視されるので、
起動ログの `tensor ... buffer type` 行で実際の割り当て先を必ず確認すること。

### KV cache の型

`--cache-type-k` / `--cache-type-v` が受ける型 (`common/arg.cpp`):

- 標準: `f32`, `f16`, `bf16`, `q8_0`, `q4_0`, `q4_1`, `iq4_nl`, `q5_0`, `q5_1`
- 低ビット追加分: `q6_0`, `q6_1`, `q3_0`, `q3_1`, `q2_0`, `q2_1`
- KVarN 擬似型: `kvarn2`, `kvarn3`, `kvarn4`, `kvarn5`, `kvarn6`, `kvarn8`

precision tail は `--kv-tail-tokens` (数値 / `auto` / 位置指定リスト) と
`--kv-tail-type` (`f16` または `bf16`)。SWA レイヤーだけ別扱いにするなら
`--cache-type-k-swa` / `--cache-type-v-swa`。

`Dockerfile.turboquant` からの移行について: **`turbo2` / `turbo3` / `turbo4` は
beellama でもそのまま受け付ける**が、v0.4.0 で削除済みとして
`q2_0` / `q3_0` / `q4_0` に読み替えられ、警告が出る (`common/arg.cpp`)。
KVarN を使いたいなら明示的に `kvarn3` 等へ書き換えること。

### 未検証

- **docker build 自体は未実行。** 検証環境から Docker Hub の blob CDN に到達できず、
  ベースイメージを pull できなかった。ソースの clone、CMake オプションの実在確認、
  CPU バックエンドでの `cmake` configure (exit 0、未使用変数なし) までは通してある。
  apt のパッケージ名、CUDA toolkit のインストール、CUDA/Vulkan の実コンパイルは未検証。
- 配布バイナリ (`ghcr.io/anbeeld/beellama.cpp`) に sm_80 が含まれるかは未確認。
  含まれているなら CMP 170HX でもビルド不要で済む。自前ビルドはその確認が取れるまでの保険。
- CMP 170HX の PCIe リンクは Gen1 x4 相当に絞られている。`--split-mode layer` の
  token generation は毎トークン hidden state しか流れないので耐えるが、
  モデルロードと prefill は明確に落ちる。`--tensor-split` の配分はそれを見込むこと。
- Vulkan を積んでいるので、下記「計測時の落とし穴 1」がこの構成でも当てはまる。
  CUDA と Vulkan の両方から同じ GPU が見えるため、`--device CUDA0,CUDA1,CUDA2` の
  ように明示しないとレイヤーがバックエンドを跨いで分散する。

## ライセンス

Dockerfile 部分は MIT。ビルドされる成果物はそれぞれ llama.cpp (MIT)、
llama-swap (MIT)、ROCm / CUDA の各ライセンスに従う。
