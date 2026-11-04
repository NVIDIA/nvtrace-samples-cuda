# Third-Party License Texts

Preserve this file with [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
when distributing the corresponding code. Component copyright statements and
special notices appear below; identical license terms are reproduced once in
this file or referenced from [LICENSE](LICENSE), with references from each
applicable component.

The dependency versions and selected DuckDB extensions are identified in the
notices file. Test and benchmark terms apply when that code is distributed.
Daemon and PerfWorks runtime dependencies, copyright attributions and license
terms are listed in [the daemon notices](https://github.com/NVIDIA/nsight-perf-daemon/blob/main/THIRD_PARTY_NOTICES.md).
Redistribute those notices along with any redistribution of the corresponding
components.

## Components

### cxxopts 3.3.1

Pinned upstream source archive. License: `MIT`.

```text
Copyright (c) 2014 Jarryd Beck
```

Applicable terms: [license text 1](#license-text-1).

### FlatBuffers 23.5.26

Pinned upstream source archive. License: `Apache-2.0`.

```text
Copyright 2021 Google Inc. All rights reserved.
Copyright 2023 Google Inc. All rights reserved.
Copyright 2014 Google Inc. All rights reserved.
Copyright 2017 Google Inc. All rights reserved.
Copyright 2022 Google Inc. All rights reserved.
Copyright 2015 Google Inc. All rights reserved.
Copyright 2016 Google Inc. All rights reserved.
Copyright 2018 Dan Field
Copyright 2018 Google Inc. All rights reserved.
Copyright 2020 Google Inc. All rights reserved.
```

Applicable terms: [license text 2](#license-text-2).

### Protocol Buffers 3.21.12

Pinned upstream source archive. License: `BSD-3-Clause`.

```text
Copyright 2008 Google Inc.  All rights reserved.
Copyright 2015 Google Inc.  All rights reserved.
Copyright 2014 Google Inc.  All rights reserved.
Copyright (c) 2006, Google Inc.
All rights reserved.
Copyright 2012 Google Inc.  All rights reserved.
Copyright 2005 Google Inc.
All rights reserved.
Copyright 2009 Google Inc.  All rights reserved.
```

Applicable terms: [license text 3](#license-text-3).

### Perfetto schema baseline v58.3

Documented schema baseline 11fbaed836361258f23604daf67c837f3034e663. License: `Apache-2.0`.

```text
Copyright (c) 2017, The Android Open Source Project
Copyright 2015 The Chromium Authors
Copyright 2025 The syntaqlite Authors. All rights reserved.
Copyright 2002 Niels Provos <provos@citi.umich.edu>
All rights reserved.
Copyright (C) 2020 The Android Open Source Project
Copyright (C) 2018 The Android Open Source Project
Copyright (C) 2019 The Android Open Source Project
Copyright (C) 2017 The Android Open Source Project
```

Applicable terms: [license text 2](#license-text-2).

### Google Benchmark 1.9.5

Pinned upstream source archive. License: `Apache-2.0`.

```text
Copyright 2015 Google Inc. All rights reserved.
Copyright 2018 Google Inc. All rights reserved.
Copyright 2016 Ismael Jimenez Martinez. All rights reserved.
Copyright 2021 Google Inc. All rights reserved.
Copyright 2017 Roman Lebedev. All rights reserved.
```

Applicable terms: [license text 2](#license-text-2).

### GoogleTest 1.14.0

Pinned upstream source archive. License: `BSD-3-Clause`.

```text
Copyright 2008, Google Inc.
All rights reserved.
Copyright 2005, Google Inc.
All rights reserved.
Copyright 2007, Google Inc.
All rights reserved.
Copyright 2008 Google Inc.
All Rights Reserved.
Copyright 2015, Google Inc.
All rights reserved.
```

Applicable terms: [license text 3](#license-text-3).

### NVTX C/C++

Supplied NVIDIA NVTX import. License: `Apache-2.0 WITH LLVM-exception`.

```text
Copyright (c) 2009-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Copyright (c) 2020-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
```

Applicable terms: [license text 2](#license-text-2), [license text 4](#license-text-4).

## License texts

### License text 1

MIT.

```text
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

### License text 2

Apache-2.0: see the full text in [LICENSE](LICENSE).

The same license text applies to the Apache-2.0 components identified above.
Preserve LICENSE with this file when redistributing those components.

### License text 3

BSD-3-Clause.

```text
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above
copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the
distribution.
Neither the name of Google Inc. nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### License text 4

LLVM exception.

```text
---- LLVM Exceptions to the Apache 2.0 License ----

As an exception, if, as a result of your compiling your source code, portions
of this Software are embedded into an Object form of such source code, you
may redistribute such embedded portions in such Object form without complying
with the conditions of Sections 4(a), 4(b) and 4(d) of the License.

In addition, if you combine or link compiled forms of this Software with
software that is licensed under the GPLv2 ("Combined Software") and if a
court of competent jurisdiction determines that the patent provision (Section
3), the indemnity provision (Section 9) or other Section of the License
conflicts with the conditions of the GPLv2, you may retroactively and
prospectively choose to deem waived or otherwise exclude such Section(s) of
the License, but only in their entirety and only with respect to the Combined
Software.
```

