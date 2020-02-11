
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_proto_library")
load("@rules_java//java:defs.bzl", "java_binary", "java_lite_proto_library", "java_proto_library")
load("@rules_proto//proto:defs.bzl", "proto_library")

proto_library(
    name = "message_proto",
    srcs = ["message.proto"],
    deps = ["@com_google_protobuf//:timestamp_proto"],
)

cc_proto_library(
    name = "mesasge_cc_proto",
    deps = [":message_proto"],
)
