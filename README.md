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

## 2種類の Dockerfile

| | ソース | KV cache turbo2/3/4 | ROCm ベース |
|---|---|---|---|
| `Dockerfile.upstream` | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) `b10470` | なし | 7.14 (TheRock) |
| `Dockerfile.turboquant` | [AtomicBot-ai/atomic-llama-cpp-turboquant](https://github.com/AtomicBot-ai/atomic-llama-cpp-turboquant) `b10269-1.5.1` | **あり** | 7.2.4 |

`--spec-type` / `draft-mtp` / `draft-eagle3` / `ngram-*` はどちらでも使える
(上流に取り込み済み)。フォーク側だけの利点は TurboQuant の KV cache 量子化型で、
`turbo3` は約 3.5bit/要素。`q8_0` 比でおよそ 1/2.4、`f16` 比でおよそ 1/4.5 の KV サイズになる。
長コンテキストを VRAM に収めたい場合はフォーク版を選ぶ。

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

生成物は 30GB 級になる (CUDA フルツールキット + ROCm + ビルドツリーを1ステージに持つため)。
配布したい場合はランタイム用にマルチステージ化して `cuda-runtime` に落とすこと。

## 実行

```sh
docker run --rm \
  --device /dev/kfd --device /dev/dri --group-add video --group-add render \
  --gpus all --security-opt seccomp=unconfined \
  -v /path/to/models:/models:ro -v /path/to/config.yaml:/app/config.yaml \
  -p 8080:8080 llama-swap-upstream:b10470
```

## 計測メモ

Radeon Pro VII 1枚 / Qwen2.5-VL-7B-Instruct Q4_K_M / `-ngl 99 -r 4`。

| ビルド | pp512 | pp2048 | tg128 |
|---|---|---|---|
| TurboQuant `b10018-1.1.2` | 811.7 | 880.1 | 76.4 |
| TurboQuant `b10269-1.5.1` | 974.5 | 957.8 | 74.1 |
| `mixa3607/llama.cpp-gfx906:b10470-rocm-7.14` (HIP のみ) | 1001.5 | 997.1 | 76.2 |

計測時の落とし穴を2つ記録しておく。どちらも実際に踏んで数字を誤読した。

1. **`HIP_VISIBLE_DEVICES` は Vulkan を絞れない。**
   Vulkan バックエンドを同梱したビルドでは、ROCm を1枚に絞っても Vulkan からは
   全 GPU が見えており、レイヤーが ROCm と Vulkan に分散してバックエンド跨ぎ転送が発生する。
   その状態だと pp512 が 975 → 506 まで落ちた。
   `llama-bench --device ROCm0` のようにデバイスを明示すること。
2. **他プロセスの VRAM 占有を確認する。**
   ComfyUI が同じ GPU に常駐していた状態では tg128 が 36.2 ± 23.0 のように暴れた。
   `rocm-smi --showpids` で確認してから測る。

また、ROCm 7.14 のランタイムだけを 7.2.4 ビルドのバイナリに
`LD_LIBRARY_PATH` で差し替えて試したが (soname は `librocblas.so.5` /
`libhipblas.so.3` / `libamdhip64.so.7` で一致する)、prefill は変わらず
tg はむしろ不安定化した。ROCm のバージョン自体は gfx906 の llama.cpp 性能に
ほとんど寄与していない。

## ライセンス

Dockerfile 部分は MIT。ビルドされる成果物はそれぞれ llama.cpp (MIT)、
llama-swap (MIT)、ROCm / CUDA の各ライセンスに従う。
