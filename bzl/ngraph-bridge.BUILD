load("@rules_foreign_cc//tools/build_defs:cmake.bzl", "cmake_external")
load("@rules_foreign_cc//:workspace_definitions.bzl", "rules_foreign_cc_dependencies")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD/MIT-like license (for zlib)

# TODO: build zlib

genrule(
    name = "ngraph-bridge",
    outs = ["build_cmake/artifacts"],
    cmd = "python3 build_ngtf.py --use_grappler_optimizer",
)

#cc_library(
#    name = "ngraph-bridge",
#    deps = [":ngraph-bridge-build"]
#)
