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

## 2種類の Dockerfile

| | ソース | KV cache turbo2/3/4 | ROCm ベース |
|---|---|---|---|
| `Dockerfile.upstream` | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) `b10470` | なし | 7.14 (TheRock) |
| `Dockerfile.turboquant` | [AtomicBot-ai/atomic-llama-cpp-turboquant](https://github.com/AtomicBot-ai/atomic-llama-cpp-turboquant) `b10269-1.5.1` | **あり** | 7.2.4 |

`--spec-type` / `draft-mtp` / `draft-eagle3` / `ngram-*` はどちらでも使える
(上流に取り込み済み)。フォーク側だけの利点は TurboQuant の KV cache 量子化型で、
`turbo3` は約 3.5bit/要素。`q8_0` 比でおよそ 1/2.4、`f16` 比でおよそ 1/4.5 の KV サイズになる。
長コンテキストを VRAM に収めたい場合はフォーク版を選ぶ。
そうでなければ上流版のほうが速く、イメージも半分で済む。

## ビルド

```sh
docker build -t llama-swap-upstream:b10470   -f Dockerfile.upstream   .
docker build -t llama-swap-turboquant:b10269 -f Dockerfile.turboquant .
```

ビルドコンテキストは使わないので `.` は何でもよい。主な `--build-arg`:

| ARG | 既定 | 用途 |
|---|---|---|
| `CUDA_ARCH` | `120` | NVIDIA の compute capability。RTX 50 系は 120 |
| `AMDGPU_TARGET` | `gfx906` | AMD のターゲット |
| `LLAMACPP_TAG` / `TURBOQUANT_TAG` | 上表 | ソースのタグ |
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

## ライセンス

Dockerfile 部分は MIT。ビルドされる成果物はそれぞれ llama.cpp (MIT)、
llama-swap (MIT)、ROCm / CUDA の各ライセンスに従う。
