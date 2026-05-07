# hymotion 环境依赖

本文档记录本机 `hymotion` Conda 环境的依赖快照，便于复现和整理仓库运行环境。

## 环境概要

- 环境名：`hymotion`
- Python：`3.10.19`
- Conda channels：`defaults`
- 导出命令：`conda env export -n hymotion --no-builds`
- 导出日期：`2026-05-07`

## Conda 依赖

```yaml
dependencies:
  - arrow-cpp=23.0.1
  - aws-c-auth=0.9.4
  - aws-c-cal=0.9.13
  - aws-c-common=0.12.6
  - aws-c-compression=0.3.1
  - aws-c-event-stream=0.5.9
  - aws-c-http=0.10.7
  - aws-c-io=0.23.3
  - aws-c-mqtt=0.13.3
  - aws-c-s3=0.11.3
  - aws-c-sdkutils=0.2.4
  - aws-checksums=0.2.8
  - aws-crt-cpp=0.35.4
  - aws-sdk-cpp=1.11.720
  - blas=1.0
  - bottleneck=1.4.2
  - bzip2=1.0.8
  - c-ares=1.34.6
  - ca-certificates=2025.12.2
  - cramjam=2.11.0
  - et_xmlfile=2.0.0
  - expat=2.7.3
  - fastparquet=2025.12.0
  - fsspec=2026.1.0
  - gettext=0.25.1
  - gettext-tools=0.25.1
  - gflags=2.2.2
  - glog=0.5.0
  - icu=73.1
  - jansson=2.14
  - libabseil=20260107.0
  - libasprintf=0.25.1
  - libasprintf-devel=0.25.1
  - libbrotlicommon=1.2.0
  - libbrotlidec=1.2.0
  - libbrotlienc=1.2.0
  - libcurl=8.18.0
  - libcxx=20.1.8
  - libev=4.33
  - libevent=2.1.12
  - libexpat=2.7.3
  - libffi=3.4.4
  - libgettextpo=0.25.1
  - libgettextpo-devel=0.25.1
  - libgfortran=15.2.0
  - libgfortran5=15.2.0
  - libgrpc=1.78.0
  - libiconv=1.18
  - libidn2=2.3.8
  - libintl=0.25.1
  - libintl-devel=0.25.1
  - libkrb5=1.22.1
  - libnghttp2=1.67.1
  - libopenblas=0.3.31
  - libprotobuf=6.33.5
  - libre2-11=2025.11.05
  - libssh2=1.11.1
  - libthrift=0.22.0
  - libunistring=1.3
  - libxml2=2.13.9
  - libzlib=1.3.1
  - llvm-openmp=21.1.8
  - lmdb=0.9.31
  - lz4-c=1.9.4
  - ncurses=6.5
  - numexpr=2.14.1
  - openpyxl=3.1.5
  - openssl=3.5.5
  - orc=2.2.0
  - pip=25.3
  - pyarrow=23.0.1
  - python=3.10.19
  - python-dateutil=2.9.0post0
  - python-tzdata=2025.3
  - pytz=2025.2
  - re2=2025.11.05
  - readline=8.3
  - six=1.17.0
  - snappy=1.2.2
  - sqlite=3.51.1
  - tk=8.6.15
  - tzdata=2025b
  - utf8proc=2.6.1
  - wheel=0.45.1
  - xz=5.6.4
  - zlib=1.3.1
  - zstd=1.5.7
```

## Pip 依赖

```yaml
pip:
  - alembic==1.18.1
  - annotated-types==0.7.0
  - anyio==4.12.1
  - beautifulsoup4==4.14.3
  - certifi==2026.1.4
  - charset-normalizer==3.4.4
  - chumpy==0.70
  - click==8.3.1
  - colorlog==6.10.1
  - configer==1.3.1
  - configparser==7.2.0
  - contourpy==1.3.2
  - cycler==0.12.1
  - distro==1.9.0
  - einops==0.8.1
  - exceptiongroup==1.3.1
  - filelock==3.20.3
  - fonttools==4.62.1
  - freetype-py==2.5.1
  - gdown==5.2.1
  - gitdb==4.0.12
  - gitpython==3.1.46
  - google-ai-generativelanguage==0.6.15
  - google-api-core==2.29.0
  - google-api-python-client==2.188.0
  - google-auth==2.47.0
  - google-auth-httplib2==0.3.0
  - google-genai==1.59.0
  - googleapis-common-protos==1.72.0
  - grpcio==1.76.0
  - grpcio-status==1.71.2
  - h11==0.16.0
  - hf-xet==1.2.0
  - httpcore==1.0.9
  - httplib2==0.31.1
  - httpx==0.28.1
  - huggingface-hub==0.36.0
  - human-body-prior==0.8.5.0
  - idna==3.11
  - imageio==2.37.2
  - jinja2==3.1.6
  - jiter==0.12.0
  - joblib==1.4.2
  - kiwisolver==1.5.0
  - lazy-loader==0.4
  - mako==1.3.10
  - markupsafe==3.0.3
  - matplotlib==3.10.7
  - mpmath==1.3.0
  - networkx==3.4.2
  - numpy==2.2.6
  - numpyencoder==0.3.0
  - openai==2.15.0
  - opencv-python==4.13.0.90
  - optuna==4.2.1
  - packaging==26.0
  - pandas==2.2.3
  - pillow==12.1.1
  - platformdirs==4.5.1
  - proto-plus==1.27.0
  - protobuf==5.29.5
  - pyasn1==0.6.2
  - pyasn1-modules==0.4.2
  - pydantic==2.12.5
  - pydantic-core==2.41.5
  - pyglet==2.1.12
  - pyopengl==3.1.0
  - pyparsing==3.3.2
  - pyrender==0.1.45
  - pysocks==1.7.1
  - pyyaml==6.0.3
  - regex==2026.1.15
  - requests==2.32.5
  - rsa==4.9.1
  - safetensors==0.7.0
  - scikit-image==0.25.2
  - scikit-learn==1.6.1
  - scipy==1.15.3
  - seaborn==0.13.2
  - sentencepiece==0.2.1
  - sentry-sdk==2.50.0
  - setuptools==75.8.0
  - shellingham==1.5.4
  - smmap==5.0.2
  - smplx==0.1.28
  - sniffio==1.3.1
  - soupsieve==2.8.3
  - sqlalchemy==2.0.46
  - sympy==1.14.0
  - tenacity==9.1.2
  - tensorboardx==2.6.4
  - threadpoolctl==3.6.0
  - tifffile==2025.5.10
  - timm==1.0.15
  - tokenizers==0.22.2
  - tomli==2.4.0
  - torch==2.9.1
  - torch-dct==0.1.6
  - torchaudio==2.9.1
  - torchdiffeq==0.2.5
  - torchgeometry==0.1.2
  - torchvision==0.24.1
  - tqdm==4.66.1
  - transformers==4.57.6
  - transforms3d==0.4.2
  - trimesh==4.6.8
  - typer-slim==0.21.1
  - typing-extensions==4.15.0
  - typing-inspection==0.4.2
  - uritemplate==4.2.0
  - urllib3==2.6.3
  - wandb==0.24.0
  - websockets==15.0.1
```

## 复现方式

如果希望在另一台机器复现接近的环境，可以先写成 `environment.yml` 再创建：

```bash
conda create -n hymotion python=3.10.19
conda activate hymotion
conda install arrow-cpp=23.0.1 pyarrow=23.0.1 fastparquet=2025.12.0 openpyxl=3.1.5
pip install alembic==1.18.1 chumpy==0.70 human-body-prior==0.8.5.0 matplotlib==3.10.7 numpy==2.2.6 opencv-python==4.13.0.90 pandas==2.2.3 pyrender==0.1.45 requests==2.32.5 scikit-learn==1.6.1 scipy==1.15.3 seaborn==0.13.2 smplx==0.1.28 torch==2.9.1 torchaudio==2.9.1 torchvision==0.24.1 transformers==4.57.6 trimesh==4.6.8 wandb==0.24.0
```

如果后续希望直接用于环境恢复，建议额外保存一份标准 `environment.yml`，而不是只保留 Markdown。