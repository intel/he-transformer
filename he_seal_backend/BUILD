# Copyright 2018-2020 Intel Corporation.




cc_library(
    name = "he_seal_backend",
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.hpp"]),
    deps = [
        "@seal",
        "@ngraph",
        "@com_google_protobuf",
    ],
)

cc_library(
    name = "dummy",
    srcs = ["dummy.cpp"],
    deps = [
        "@seal",
    ],
)
