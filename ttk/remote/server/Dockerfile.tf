# Example Dockerfile for TF executor.
# Base image version is an example — actual image configured via docker.images.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
FROM tensorflow/tensorflow:2.15.0-gpu
# Install executor deps (tf base already has tensorflow; add per-op deps here)
RUN pip install numpy pyyaml
COPY ttk /executor/ttk
WORKDIR /executor
ENTRYPOINT ["python", "-m", "ttk.remote.server.executor_main"]
